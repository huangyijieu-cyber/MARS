import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

SEARCH_MODE = 3

import json
import pandas as pd
import re
import numpy as np
import ast
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import signal
import psutil
import subprocess
import time

from config import *
from data_loader import MolecularDataLoader
from custom_retriever import FaissDiceRetriever
from prompt_engine import build_molrag_prompt, format_docs_for_context, get_top_peaks_string
from cfmid_adapter import CFMIDAdapter
from mcts_engine import MCTSEngine
from post_processor import MolecularPostProcessor
from langchain_openai import ChatOpenAI
import cfmid_adapter

from evaluator import MolecularEvaluator, robust_standardize, get_inchikey_match

worker_resources = {}
global_executor = None

def init_worker(worker_id_queue):
    import os
    from rdkit import RDLogger
    import warnings
    import faiss

    try:
        worker_id = worker_id_queue.get(timeout=10)
        cfmid_adapter.CURRENT_WORKER_ID = worker_id
    except Exception as e:
        print(f"[Worker {os.getpid()}] FATAL: Failed to assign Worker ID: {e}")
        raise RuntimeError(f"Worker {os.getpid()} could not get an ID.")

    faiss.omp_set_num_threads(1)
    RDLogger.DisableLog('rdApp.*')
    warnings.filterwarnings("ignore")

    retriever = FaissDiceRetriever.load_from_dir(FAISS_DB_DIR, k=RETRIEVAL_K)
    worker_resources['retriever'] = retriever

    worker_resources['cfmid'] = CFMIDAdapter(docker_image="wishartlab/cfmid:latest")

    worker_resources['llm'] = ChatOpenAI(
        model=MODEL,
        openai_api_base=BASE_URL,
        openai_api_key=API_KEY,
        temperature=TEMPERATURE,
        request_timeout=60,
        max_retries=3
    )

    worker_resources['evaluator'] = MolecularEvaluator(use_mces=True)
    worker_resources['post_processor'] = MolecularPostProcessor()
    worker_resources['prompt_template'] = build_molrag_prompt(top_k=TOP_K)

def process_single_sample(row_data):
    retriever = worker_resources['retriever']
    cfmid_adapter = worker_resources['cfmid']
    llm = worker_resources['llm']
    evaluator = worker_resources['evaluator']
    post_processor_inst = worker_resources['post_processor']
    prompt_template = worker_resources['prompt_template']

    cfmid_adapter.reset_status()

    idx = row_data['index']

    try:
        def process_formula(val): return str(val).strip() if not pd.isna(val) else ""
        target_formula = process_formula(row_data.get('formula'))

        raw_adduct = str(row_data.get('adduct', 'M+')).strip()
        adduct_map = {
            "M+": "[M]+", "[M]+": "[M]+",
            "M+H": "[M+H]+", "[M+H]+": "[M+H]+",
            "M+Na": "[M+Na]+", "[M+Na]+": "[M+Na]+",
            "M-H": "[M-H]-", "[M-H]-": "[M-H]-"
        }
        current_adduct = adduct_map.get(raw_adduct, "[M]+")

        true_smiles = row_data.get('true_smiles', '')
        target_fp_list = row_data.get('target_fp_list', [])
        target_peaks_cleaned = row_data.get('target_peaks_cleaned', [])

        if not target_fp_list and not target_peaks_cleaned:
            return None

        target_top_peaks_str = get_top_peaks_string(target_peaks_cleaned, top_k=10)

        retrieval_query = {
            "fps": target_fp_list,
            "formula": target_formula
        }
        retrieved_docs = retriever.invoke(json.dumps(retrieval_query))

        def simple_count(f): return len(re.findall(r'[A-Z]', str(f)))
        target_cnt = simple_count(target_formula)
        for doc in retrieved_docs:
            doc.metadata['size_diff'] = abs(simple_count(doc.metadata.get('formula', '')) - target_cnt)
            if 'tier_rank' not in doc.metadata: doc.metadata['tier_rank'] = 99

        retrieved_docs.sort(key=lambda x: (x.metadata['tier_rank'], -x.metadata['score']))
        final_context_docs = retrieved_docs[:CONTEXT_K]

        scores = [d.metadata.get('score', 0.0) for d in final_context_docs]
        max_retr_sim = max(scores) if scores else 0.0
        min_retr_sim = min(scores) if scores else 0.0
        ref_smiles_list = [d.metadata.get('smiles') for d in final_context_docs if d.metadata.get('smiles')]

        context_str = format_docs_for_context(final_context_docs, target_fp_list)
        prompt_input = {
            "context": context_str,
            "target_formula": target_formula,
            "target_top_peaks": target_top_peaks_str,
            "top_k": TOP_K
        }
        prompt_value = prompt_template.invoke(prompt_input)

        mcts_engine = None
        if target_peaks_cleaned:
            mcts_engine = MCTSEngine(
                llm=llm,
                cfmid_adapter=cfmid_adapter,
                target_formula=target_formula,
                target_peaks=target_peaks_cleaned,
                target_fp=target_fp_list,
                reference_smiles=ref_smiles_list,
                adduct=current_adduct,
                top_k=TOP_K,
                true_smiles=true_smiles,
                search_mode=SEARCH_MODE
            )

        initial_candidates = []
        max_init_retries = 3

        initial_prompt_str = prompt_value.to_string()
        initial_response_str = ""

        for attempt in range(max_init_retries):
            try:
                current_temp = TEMPERATURE + (attempt * 0.3)
                bound_llm = llm.bind(temperature=min(1.0, current_temp))

                initial_response_msg = bound_llm.invoke(prompt_value)
                initial_response_str = initial_response_msg.content

                generations_list = [initial_response_msg]

                _, candidates = post_processor_inst.process_generations(
                    generations_list, target_formula, target_fp_list=target_fp_list
                )

                if not candidates: continue

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
                    rag_seeds = [d.metadata.get('smiles') for d in final_context_docs if d.metadata.get('smiles')]
                    if rag_seeds:
                        try:
                            from mcts_engine import MCTSNode
                            if mcts_engine.root is None: mcts_engine.root = MCTSNode("VIRTUAL_ROOT")
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
                            pass

                if is_initialized:
                    mcts_engine.search(n_iterations=MCTS_ITERATIONS)
                    raw_refined = mcts_engine.get_refined_top_k()

                    if raw_refined:
                        top1_candidate = raw_refined[0]
                        candidates_smiles = raw_refined
                        best_pred_smiles = top1_candidate

                        is_perfect = mcts_engine._validate_formula(top1_candidate, strict_h=True)
                        base_status = "Refined" if "Fallback" not in mcts_status else "Fallback -> Refined"

                        if is_perfect:
                            mcts_status = f"{base_status} (Success)"
                        else:
                            mcts_status = f"{base_status} (Formula Mismatch)"
                    else:
                        mcts_status += " (No result)"
                else:
                    mcts_status = "Failed (Init & Fallback)"

            except Exception as e:
                mcts_status = f"Error: {str(e)}"
        else:
            if not initial_candidates:
                mcts_status = "Skipped (No Peaks & LLM Failed)"

        if cfmid_adapter.has_timeout:
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
            mcts_prompt_str = mcts_engine.successful_mcts_prompt if mcts_engine else None
            mcts_resp_str = mcts_engine.successful_mcts_response if mcts_engine else None
            safe_smiles = re.sub(r'[\\/*?:"<>|]', "_", true_smiles)
            if not safe_smiles: safe_smiles = f"unknown_{idx}"
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
            except Exception as e:
                pass

        metrics = mcts_engine.get_search_metrics() if mcts_engine else {"valid_node_rate": 0.0, "avg_reward_delta": 0.0}

        m_top1 = evaluator.compute_metrics(true_smiles, best_pred_smiles)
        m_topk = evaluator.compute_topk_metrics(true_smiles, candidates_smiles)

        result_record = {
            "index": idx,
            "identifier": row_data.get('identifier', idx),
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
            "avg_reward_delta": metrics["avg_reward_delta"]
        }
        return result_record

    except Exception as e:
        tqdm.write(f"Error in sample {idx}: {e}")
        return None

def kill_all_children():
    try:
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        if children:
            for child in children:
                try: child.kill()
                except: pass
            _, alive = psutil.wait_procs(children, timeout=2)
    except Exception: pass

def signal_handler(signum, frame):
    global global_executor
    if global_executor:
        try: global_executor.shutdown(wait=False, cancel_futures=True)
        except: global_executor.shutdown(wait=False)

    try: CFMIDAdapter.cleanup_pool()
    except: pass

    kill_all_children()
    os._exit(1)

def main():
    global global_executor
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    os.makedirs("data/prompt", exist_ok=True)

    loader = MolecularDataLoader()
    df = loader.load_test_data(TEST_FILE_PATH, limit=MAX_TEST_SAMPLES)

    tasks = []
    for idx, row in df.iterrows():
        import selfies as sf
        raw_smi = row.get('smiles', '')
        raw_sf = row.get('selfies', '')
        true_smi = ""

        if raw_smi and isinstance(raw_smi, str):
            true_smi = raw_smi
        elif raw_sf and isinstance(raw_sf, str):
            try: true_smi = sf.decoder(raw_sf)
            except: pass

        task = row.to_dict()
        task['index'] = idx
        task['true_smiles'] = true_smi

        if isinstance(row.get('fps'), str): task['target_fp_list'] = json.loads(row['fps'])
        else: task['target_fp_list'] = row.get('fps', [])

        p_raw = row.get('peaks')
        p_list = []
        if isinstance(p_raw, str): p_list = json.loads(p_raw)
        else: p_list = p_raw

        cleaned = []
        if p_list:
            for p in p_list:
                try:
                    if float(p[1]) > 0: cleaned.append((float(p[0]), float(p[1])))
                except: pass
        task['target_peaks_cleaned'] = cleaned
        tasks.append(task)

    valid_finished_count = 0
    sum_top1_acc = 0.0
    sum_topk_acc = 0.0
    sum_valid_node_rate = 0.0
    sum_avg_reward_delta = 0.0

    finished_indices = set()

    if os.path.exists(OUTPUT_FILE_PATH):
        try:
            print(f"Checking for existing checkpoint: {OUTPUT_FILE_PATH}")
            existing_df = pd.read_csv(OUTPUT_FILE_PATH)
            if not existing_df.empty and 'index' in existing_df.columns:
                finished_indices = set(existing_df['index'].astype(int).unique())
                print(f"-> Found {len(finished_indices)} finished samples.")

                for _, row in existing_df.iterrows():
                    status = str(row.get('mcts_status', 'N/A'))
                    is_timeout = "Timeout" in status
                    is_sys_err = any(x in status for x in ["Init Failed", "Skipped", "Error"])

                    if not is_timeout and not is_sys_err:
                        valid_finished_count += 1
                        sum_top1_acc += row.get('top1_mol_accuracy', 0)
                        sum_topk_acc += row.get('topk_mol_accuracy', 0)
                        sum_valid_node_rate += float(row.get('valid_node_rate', 0.0))
                        sum_avg_reward_delta += float(row.get('avg_reward_delta', 0.0))

                original_len = len(tasks)
                tasks = [t for t in tasks if t['index'] not in finished_indices]
                print(f"-> Resuming execution. Remaining tasks: {len(tasks)} / {original_len}")
            else:
                print("-> Checkpoint empty or invalid. Starting fresh.")
        except Exception as e:
            print(f"-> Error reading checkpoint: {e}. Starting fresh.")
    else:
        print("-> No checkpoint found. Starting fresh.")

    if not tasks:
        print("All tasks completed.")
        return

    MAX_WORKERS = PWORKERS
    print(f"Starting execution with {MAX_WORKERS} workers (Mode: {SEARCH_MODE})...")

    try:
        CFMIDAdapter.prepare_pool(
            num_workers=MAX_WORKERS,
            cpus=DOCKER_CPUS,
            mem=DOCKER_MEM,
            base_dir=POOL_DIR_BASE,
            image="wishartlab/cfmid:latest"
        )
    except Exception as e:
        print(f"FATAL: Failed to initialize Docker pool: {e}")
        CFMIDAdapter.cleanup_pool()
        return

    try:
        m = multiprocessing.Manager()
        id_queue = m.Queue()
        for i in range(MAX_WORKERS):
            id_queue.put(i)
    except Exception as e:
        print(f"FATAL: Failed to create Manager Queue: {e}")
        CFMIDAdapter.cleanup_pool()
        return

    try:
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=init_worker,
            initargs=(id_queue,)
        ) as pool:
            global_executor = pool

            future_to_idx = {pool.submit(process_single_sample, t): t['index'] for t in tasks}

            pbar = tqdm(
                as_completed(future_to_idx),
                total=len(tasks),
                desc=f"Processing",
                mininterval=0.5
            )

            for future in pbar:
                res = future.result()
                if res:
                    single_df = pd.DataFrame([res])
                    header_needed = not os.path.exists(OUTPUT_FILE_PATH)
                    single_df.to_csv(OUTPUT_FILE_PATH, mode='a', header=header_needed, index=False)

                    status = str(res.get('mcts_status', 'N/A'))
                    is_timeout = "Timeout" in status
                    is_sys_err = any(x in status for x in ["Init Failed", "Skipped", "Error"])

                    if not is_timeout and not is_sys_err:
                        valid_finished_count += 1
                        sum_top1_acc += res.get('top1_mol_accuracy', 0)
                        sum_topk_acc += res.get('topk_mol_accuracy', 0)
                        sum_valid_node_rate += float(res.get('valid_node_rate', 0.0))
                        sum_avg_reward_delta += float(res.get('avg_reward_delta', 0.0))

                    cur_t1 = sum_top1_acc / valid_finished_count if valid_finished_count > 0 else 0.0
                    cur_tk = sum_topk_acc / valid_finished_count if valid_finished_count > 0 else 0.0
                    cur_vnr = sum_valid_node_rate / valid_finished_count if valid_finished_count > 0 else 0.0
                    cur_delta = sum_avg_reward_delta / valid_finished_count if valid_finished_count > 0 else 0.0

                    display_status = status[:25] + ".." if len(status) > 25 else status
                    pbar.set_postfix_str(f"T1:{cur_t1:.4f} TK:{cur_tk:.4f} VNR:{cur_vnr:.2f} dR:{cur_delta:.3f} | {display_status}")

    finally:
        print("Cleaning up resources...")
        global_executor = None
        try: CFMIDAdapter.cleanup_pool()
        except: pass
        try: m.shutdown()
        except: pass

    try:
        final_df = pd.read_csv(OUTPUT_FILE_PATH)
        print(f"\nExperiment Finished. Total Samples: {len(final_df)}")

        def is_valid_sample(status_str):
            s = str(status_str)
            if "Timeout" in s: return False
            if any(x in s for x in ["Init Failed", "Skipped", "Error"]): return False
            return True

        if 'mcts_status' in final_df.columns:
            valid_df = final_df[final_df['mcts_status'].apply(is_valid_sample)]

            if not valid_df.empty:
                top1_agg = {
                    'top1_valid_mols': valid_df['top1_valid_mols'].mean(),
                    'top1_mces_dist': valid_df['top1_mces_dist'].mean(),
                    'top1_tanimoto_sim': valid_df['top1_tanimoto_sim'].mean(),
                    'top1_mol_accuracy': valid_df['top1_mol_accuracy'].mean()
                }
                topk_agg = {
                    'topk_valid_mols': valid_df['topk_valid_mols'].mean(),
                    'topk_mces_dist': valid_df['topk_mces_dist'].mean(),
                    'topk_tanimoto_sim': valid_df['topk_tanimoto_sim'].mean(),
                    'topk_mol_accuracy': valid_df['topk_mol_accuracy'].mean()
                }
                process_agg = {
                    'valid_node_rate': valid_df['valid_node_rate'].mean() if 'valid_node_rate' in valid_df.columns else 0.0,
                    'avg_reward_delta': valid_df['avg_reward_delta'].mean() if 'avg_reward_delta' in valid_df.columns else 0.0
                }

                def fmt(d): return {k: round(float(v), 4) for k, v in d.items()}
                print("\n" + "="*40)
                print(f"Final Metrics (Valid Samples: {len(valid_df)}, Mode: {SEARCH_MODE})")
                print("="*40)
                print(f"Process: {fmt(process_agg)}")
                print(f"Top1: {fmt(top1_agg)}")
                print(f"Topk: {fmt(topk_agg)}")
            else:
                print("No valid samples found for metrics calculation.")
    except Exception as e:
        print(f"Could not load final results for summary: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()