import json
import logging
import os
import re
import time

import pandas as pd

import cfmid_adapter as cfmid_module
from cfmid_adapter import CFMIDAdapter
from config import (
    API_KEY,
    BASE_URL,
    CONTEXT_K,
    FAISS_DB_DIR,
    MODEL,
    MCTS_ITERATIONS,
    RETRIEVAL_K,
    TEMPERATURE,
    TOP_K,
)
from custom_retriever import FaissDiceRetriever
from evaluator import MolecularEvaluator, get_inchikey_match, robust_standardize
from langchain_openai import ChatOpenAI
from mcts_engine import MCTSEngine
from post_processor import MolecularPostProcessor
from prompt_engine import build_molrag_prompt, format_docs_for_context, get_top_peaks_string

logger = logging.getLogger(__name__)
worker_resources = {}


def init_worker(worker_id_queue, search_mode):
    from rdkit import RDLogger
    import warnings
    import faiss

    try:
        worker_id = worker_id_queue.get(timeout=10)
        cfmid_module.CURRENT_WORKER_ID = worker_id
    except Exception:
        logger.exception("[Worker %s] FATAL: Failed to assign Worker ID", os.getpid())
        raise RuntimeError(f"Worker {os.getpid()} could not get an ID.")

    faiss.omp_set_num_threads(1)
    RDLogger.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore")

    worker_resources["retriever"] = FaissDiceRetriever.load_from_dir(FAISS_DB_DIR, k=RETRIEVAL_K)
    worker_resources["cfmid"] = CFMIDAdapter(docker_image="wishartlab/cfmid:latest")
    worker_resources["llm"] = ChatOpenAI(
        model=MODEL,
        openai_api_base=BASE_URL,
        openai_api_key=API_KEY,
        temperature=TEMPERATURE,
        request_timeout=60,
        max_retries=3,
    )
    worker_resources["evaluator"] = MolecularEvaluator(use_mces=True)
    worker_resources["post_processor"] = MolecularPostProcessor()
    worker_resources["prompt_template"] = build_molrag_prompt(top_k=TOP_K)
    worker_resources["search_mode"] = search_mode


def process_single_sample(row_data):
    retriever = worker_resources["retriever"]
    cfmid = worker_resources["cfmid"]
    llm = worker_resources["llm"]
    evaluator = worker_resources["evaluator"]
    post_processor_inst = worker_resources["post_processor"]
    prompt_template = worker_resources["prompt_template"]
    search_mode = worker_resources["search_mode"]

    cfmid.reset_status()
    idx = row_data["index"]

    try:
        def process_formula(val):
            return str(val).strip() if not pd.isna(val) else ""

        target_formula = process_formula(row_data.get("formula"))

        raw_adduct = str(row_data.get("adduct", "M+")).strip()
        adduct_map = {
            "M+": "[M]+", "[M]+": "[M]+",
            "M+H": "[M+H]+", "[M+H]+": "[M+H]+",
            "M+Na": "[M+Na]+", "[M+Na]+": "[M+Na]+",
            "M-H": "[M-H]-", "[M-H]-": "[M-H]-",
        }
        current_adduct = adduct_map.get(raw_adduct, "[M]+")

        true_smiles = row_data.get("true_smiles", "")
        target_fp_list = row_data.get("target_fp_list", [])
        target_peaks_cleaned = row_data.get("target_peaks_cleaned", [])

        if not target_fp_list and not target_peaks_cleaned:
            return None

        target_top_peaks_str = get_top_peaks_string(target_peaks_cleaned, top_k=10)
        retrieval_query = {"fps": target_fp_list, "formula": target_formula}
        retrieved_docs = retriever.invoke(json.dumps(retrieval_query))

        def simple_count(f):
            return len(re.findall(r"[A-Z]", str(f)))

        target_cnt = simple_count(target_formula)
        for doc in retrieved_docs:
            doc.metadata["size_diff"] = abs(simple_count(doc.metadata.get("formula", "")) - target_cnt)
            if "tier_rank" not in doc.metadata:
                doc.metadata["tier_rank"] = 99

        retrieved_docs.sort(key=lambda x: (x.metadata["tier_rank"], -x.metadata["score"]))
        final_context_docs = retrieved_docs[:CONTEXT_K]

        scores = [d.metadata.get("score", 0.0) for d in final_context_docs]
        max_retr_sim = max(scores) if scores else 0.0
        min_retr_sim = min(scores) if scores else 0.0
        ref_smiles_list = [d.metadata.get("smiles") for d in final_context_docs if d.metadata.get("smiles")]

        context_str = format_docs_for_context(final_context_docs, target_fp_list)
        prompt_value = prompt_template.invoke({
            "context": context_str,
            "target_formula": target_formula,
            "target_top_peaks": target_top_peaks_str,
            "top_k": TOP_K,
        })

        mcts_engine = None
        if target_peaks_cleaned:
            mcts_engine = MCTSEngine(
                llm=llm,
                cfmid_adapter=cfmid,
                target_formula=target_formula,
                target_peaks=target_peaks_cleaned,
                target_fp=target_fp_list,
                reference_smiles=ref_smiles_list,
                adduct=current_adduct,
                top_k=TOP_K,
                true_smiles=true_smiles,
                search_mode=search_mode,
            )

        initial_candidates = []
        initial_prompt_str = prompt_value.to_string()
        initial_response_str = ""
        max_init_retries = 3

        for attempt in range(max_init_retries):
            try:
                current_temp = TEMPERATURE + (attempt * 0.3)
                bound_llm = llm.bind(temperature=min(1.0, current_temp))

                initial_response_msg = bound_llm.invoke(prompt_value)
                initial_response_str = initial_response_msg.content
                _, candidates = post_processor_inst.process_generations(
                    [initial_response_msg], target_formula, target_fp_list=target_fp_list
                )

                if not candidates:
                    continue

                valid_batch = []
                for smi in candidates:
                    if mcts_engine:
                        if mcts_engine._validate_formula(smi, strict_h=False):
                            valid_batch.append(smi)
                    else:
                        valid_batch.append(smi)

                if valid_batch:
                    initial_candidates = valid_batch[:TOP_K]
                    break
            except Exception:
                logger.warning(
                    "Initial LLM generation failed for sample=%s formula=%s attempt=%s/%s",
                    idx,
                    target_formula,
                    attempt + 1,
                    max_init_retries,
                    exc_info=True,
                )
                time.sleep(1)
                continue

        initial_best_smiles = initial_candidates[0] if initial_candidates else ""
        best_pred_smiles = initial_best_smiles
        candidates_smiles = initial_candidates
        mcts_status = "Skipped"

        if mcts_engine:
            try:
                is_initialized = False

                if initial_candidates:
                    if mcts_engine.initialize_with_candidates(initial_candidates):
                        is_initialized = True
                    else:
                        mcts_status = "Init Failed (Formula Check)"

                if not is_initialized:
                    rag_seeds = [d.metadata.get("smiles") for d in final_context_docs if d.metadata.get("smiles")]
                    if rag_seeds:
                        try:
                            from mcts_engine import MCTSNode

                            if mcts_engine.root is None:
                                mcts_engine.root = MCTSNode("VIRTUAL_ROOT")
                            mcts_engine.root.visits = 1
                            mcts_engine.root.q_value = 0.0
                            mcts_engine.root.children = []

                            for rs in rag_seeds[:3]:
                                node = MCTSNode(rs, parent=mcts_engine.root)
                                score = mcts_engine._simulate_hybrid(node)
                                node.visits = 1
                                node.q_value = score
                                mcts_engine.root.children.append(node)
                                if score > mcts_engine.root.q_value:
                                    mcts_engine.root.q_value = score

                            mcts_engine.root.is_expanded = True
                            if mcts_engine.root.children:
                                is_initialized = True
                                mcts_status = "Fallback to RAG"
                        except Exception:
                            logger.warning("RAG fallback initialization failed for sample=%s", idx, exc_info=True)

                if is_initialized:
                    mcts_engine.search(n_iterations=MCTS_ITERATIONS)
                    raw_refined = mcts_engine.get_refined_top_k()

                    if raw_refined:
                        top1_candidate = raw_refined[0]
                        candidates_smiles = raw_refined
                        best_pred_smiles = top1_candidate

                        is_perfect = mcts_engine._validate_formula(top1_candidate, strict_h=True)
                        base_status = "Refined" if "Fallback" not in mcts_status else "Fallback -> Refined"
                        mcts_status = f"{base_status} (Success)" if is_perfect else f"{base_status} (Formula Mismatch)"
                    else:
                        mcts_status += " (No result)"
                else:
                    mcts_status = "Failed (Init & Fallback)"

            except Exception as e:
                logger.exception("MCTS failed for sample=%s formula=%s", idx, target_formula)
                mcts_status = f"Error: {str(e)}"
        elif not initial_candidates:
            mcts_status = "Skipped (No Peaks & LLM Failed)"

        if cfmid.has_timeout:
            mcts_status = "Failed (Timeout)"

        is_correct = False
        std_true = robust_standardize(true_smiles)
        if std_true and candidates_smiles:
            for cand in candidates_smiles:
                std_cand = robust_standardize(cand)
                if std_cand and (std_cand == std_true or get_inchikey_match(std_cand, std_true)):
                    is_correct = True
                    break

        if is_correct:
            _write_success_prompt_trace(idx, true_smiles, initial_prompt_str, initial_response_str, mcts_engine)

        metrics = mcts_engine.get_search_metrics() if mcts_engine else {"valid_node_rate": 0.0, "avg_reward_delta": 0.0}
        m_top1 = evaluator.compute_metrics(true_smiles, best_pred_smiles)
        m_topk = evaluator.compute_topk_metrics(true_smiles, candidates_smiles)

        return {
            "index": idx,
            "identifier": row_data.get("identifier", idx),
            "true_smiles": true_smiles,
            "pred_top1": best_pred_smiles,
            "pred_topk": candidates_smiles,
            "initial_guess_top1": initial_best_smiles,
            "mcts_status": mcts_status,
            **m_top1,
            **m_topk,
            "max_retrieved_similarity": max_retr_sim,
            "min_retrieved_similarity": min_retr_sim,
            "valid_node_rate": metrics["valid_node_rate"],
            "avg_reward_delta": metrics["avg_reward_delta"],
        }

    except Exception as e:
        logger.exception("Unhandled error in sample=%s", idx)
        from tqdm import tqdm

        tqdm.write(f"Error in sample {idx}: {e}")
        return None


def _write_success_prompt_trace(idx, true_smiles, initial_prompt_str, initial_response_str, mcts_engine):
    mcts_prompt_str = mcts_engine.successful_mcts_prompt if mcts_engine else None
    mcts_resp_str = mcts_engine.successful_mcts_response if mcts_engine else None
    safe_smiles = re.sub(r'[\\/*?:"<>|]', "_", true_smiles)
    if not safe_smiles:
        safe_smiles = f"unknown_{idx}"

    file_name = f"{idx}_{safe_smiles}.txt"
    file_path = os.path.join("data/prompt", file_name)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("========== 1. Initial RAG Prompt ==========\n")
            f.write(initial_prompt_str or "N/A")
            f.write("\n\n========== 2. Initial RAG Response ==========\n")
            f.write(initial_response_str or "N/A")
            f.write("\n\n========== 3. MCTS Prompt (First Success) ==========\n")
            f.write(mcts_prompt_str or "N/A (Correct in initial phase or RAG fallback)")
            f.write("\n\n========== 4. MCTS Response (First Success) ==========\n")
            f.write(mcts_resp_str or "N/A")
    except Exception:
        logger.warning("Failed to write successful prompt trace for sample=%s path=%s", idx, file_path, exc_info=True)
