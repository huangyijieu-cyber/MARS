import math
import json
import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
from langchain_core.prompts import ChatPromptTemplate
from prompt_engine import build_unified_mcts_prompt

from evaluator import robust_standardize, get_inchikey_match

logger = logging.getLogger(__name__)

class MCTSNode:
    def __init__(self, smiles: str, parent=None):
        self.smiles = smiles
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.reward = None
        self.is_expanded = False
        self.q_value = 0.0
        self.pred_peaks_cache: List[Dict[str, Any]] = []

    def uct_score(self, c_param=1.414):
        if self.visits == 0: return float('inf')
        exploitation = self.q_value 
        if self.parent:
            exploration = c_param * math.sqrt(math.log(self.parent.visits + 1e-6) / self.visits)
        else:
            exploration = 0
        return exploitation + exploration

class MCTSEngine:
    def __init__(self, llm, cfmid_adapter, target_formula, target_peaks, target_fp: List[int] = None, reference_smiles: List[str] = None, adduct="[M]+", top_k=5, true_smiles: str = "", search_mode: int = 3):
        self.llm = llm
        self.cfmid = cfmid_adapter
        self.target_formula = target_formula
        self.target_peaks = target_peaks
        self.target_fp_list = target_fp
        self.reference_smiles = reference_smiles or []
        self.adduct = adduct
        self.top_k = top_k
        self.search_mode = search_mode 
        
        self.true_smiles = true_smiles
        self.std_true_smiles = robust_standardize(true_smiles) if true_smiles else ""
        self.successful_mcts_prompt = None
        self.successful_mcts_response = None
        
        self.total_llm_proposals = 0
        self.valid_formula_proposals = 0
        self.reward_deltas = []
        
        self.ref_spectra_cache = [] 
        if self.search_mode == 3 and self.reference_smiles:
            for ref_smi in self.reference_smiles[:3]: 
                try:
                    peaks = self.cfmid.predict_spectrum(ref_smi, adduct=self.adduct)
                    if peaks:
                        self.ref_spectra_cache.append({"smiles": ref_smi, "peaks": peaks})
                except Exception:
                    logger.warning("Failed to cache reference spectrum for smiles=%r", ref_smi, exc_info=True)

        self.target_fp_bv = None
        if self.target_fp_list:
            try:
                self.target_fp_bv = DataStructs.ExplicitBitVect(4096)
                for i, enumerate_bit in enumerate(self.target_fp_list):
                    if enumerate_bit > 0: self.target_fp_bv.SetBit(i)
            except Exception:
                logger.warning("Failed to build target fingerprint bit vector", exc_info=True)

        self.visited_states: Dict[str, Dict[str, Any]] = {}
        self.solver_llm = self.llm.bind(temperature=0.7) 
        
        if self.search_mode == 1:
            from prompt_engine import build_mcts_baseline1_prompt
            self.unified_prompt = build_mcts_baseline1_prompt()
        elif self.search_mode == 2:
            from prompt_engine import build_mcts_baseline2_prompt
            self.unified_prompt = build_mcts_baseline2_prompt()
        else:
            self.unified_prompt = build_unified_mcts_prompt()
            
        self.root = None

    def _canonize(self, smiles):
        if not smiles: return ""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return ""
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        except Exception:
            logger.debug("Failed to canonicalize smiles=%r", smiles, exc_info=True)
            return ""

    def _parse_formula_to_dict(self, formula):
        if not formula: return {}
        matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
        counts = {}
        for elem, num in matches:
            counts[elem] = int(num) if num else 1
        return counts

    def _validate_formula(self, smiles, strict_h=True):
        if not smiles or smiles == "VIRTUAL_ROOT": return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return False
            calc = rdMolDescriptors.CalcMolFormula(mol)
            target_counts = self._parse_formula_to_dict(self.target_formula)
            calc_counts = self._parse_formula_to_dict(calc)
            if not strict_h:
                target_counts.pop('H', None)
                calc_counts.pop('H', None)
            return target_counts == calc_counts
        except Exception:
            logger.debug("Formula validation failed for smiles=%r target_formula=%s", smiles, self.target_formula, exc_info=True)
            return False

    def _compute_formula_score(self, smiles):
        if not smiles: return 0.0
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return 0.0
            
            curr_formula = rdMolDescriptors.CalcMolFormula(mol)
            curr_counts = self._parse_formula_to_dict(curr_formula)
            target_counts = self._parse_formula_to_dict(self.target_formula)
            
            all_elements = set(curr_counts.keys()) | set(target_counts.keys())
            total_diff = 0
            
            for elem in all_elements:
                diff = abs(curr_counts.get(elem, 0) - target_counts.get(elem, 0))
                if elem == 'H':
                    total_diff += diff * 0.5 
                else:
                    total_diff += diff * 1.0 
            
            score = 1.0 / (1.0 + total_diff)
            return score
        except Exception:
            logger.debug("Formula score calculation failed for smiles=%r target_formula=%s", smiles, self.target_formula, exc_info=True)
            return 0.0

    def _calculate_hybrid_reward(self, smiles: str, pred_peaks_dicts: List[Dict[str, Any]]) -> float:
        spec_score = 0.0
        if pred_peaks_dicts:
            spec_score = self.cfmid.compute_similarity(pred_peaks_dicts, self.target_peaks)
        if not pred_peaks_dicts and len(self.target_peaks) > 0:
            spec_score = 0.0
            
        fp_score = 0.0
        if self.target_fp_bv:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    cand = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
                    fp_score = DataStructs.TanimotoSimilarity(cand, self.target_fp_bv)
            except Exception:
                logger.warning("Failed to compute fingerprint reward for smiles=%r", smiles, exc_info=True)
        else:
            fp_score = spec_score 

        formula_score = self._compute_formula_score(smiles)

        W_SPEC = 0.5
        W_FORM = 0.3
        W_FP = 0.2
        
        return (W_SPEC * spec_score) + (W_FORM * formula_score) + (W_FP * fp_score)

    def _get_reference_hints(self, missing_mz_list: List[float], tolerance=0.5) -> str:
        if not missing_mz_list or not self.ref_spectra_cache:
            return "No specific reference hints available."
        hints = []
        for miss_mz in missing_mz_list:
            found_refs = []
            for ref in self.ref_spectra_cache:
                for p in ref["peaks"]:
                    if abs(p['mz'] - miss_mz) < tolerance and p['intensity'] > 5.0:
                        found_refs.append(ref["smiles"])
                        break
            if found_refs:
                hints.append(f"Peak m/z {miss_mz:.1f} is MISSING in current, but FOUND in Reference: {found_refs[0]}.")
        return "\n".join(hints) if hints else "References do not contain the missing peaks."

    def _compare_spectra_text(self, target_peaks, pred_peaks_dicts, tolerance=0.5):
        if not pred_peaks_dicts: return "Predicted spectrum is empty.", []
        max_t = max([x[1] for x in target_peaks]) if target_peaks else 1.0
        t_norm = [(x[0], (x[1]/max_t)*100.0) for x in target_peaks]
        
        matched_report = []
        extra_report = []
        missing_report = []
        missing_mzs = []
        t_matched_indices = set()
        p_matched_indices = set()
        
        for i, (tmz, tint) in enumerate(t_norm):
            if tint < 5.0: continue
            best_match_idx = -1
            best_match_diff = float('inf')
            for j, p_dict in enumerate(pred_peaks_dicts):
                pmz = p_dict['mz']
                pint = p_dict['intensity']
                if pint < 5.0: continue
                diff = abs(tmz - pmz)
                if diff < tolerance and diff < best_match_diff:
                    best_match_diff = diff
                    best_match_idx = j
            if best_match_idx != -1:
                t_matched_indices.add(i)
                p_matched_indices.add(best_match_idx)
                p_data = pred_peaks_dicts[best_match_idx]
                struct_info = p_data.get('smiles', 'Unknown')
                if struct_info == "Unknown": struct_info = "Structure Unidentified"
                matched_report.append(f"m/z {tmz:.1f} (Target Int: {tint:.0f}%) -> Matches Pred m/z {p_data['mz']:.1f} (Pred Int {p_data['intensity']:.0f}%). Pred Fragment: {struct_info}")
            else:
                if tint > 10.0: 
                    missing_report.append(f"m/z {tmz:.1f} (Int: {tint:.0f}%)")
                    missing_mzs.append(tmz)

        for j, p_dict in enumerate(pred_peaks_dicts):
            if j not in p_matched_indices and p_dict['intensity'] > 10.0:
                struct_info = p_dict.get('smiles', 'Unknown')
                if struct_info == "Unknown": struct_info = "Structure Unidentified"
                extra_report.append(f"m/z {p_dict['mz']:.1f} (Int: {p_dict['intensity']:.0f}%). Fragment: {struct_info}")

        matched_report.sort(key=lambda x: float(re.search(r'Int: (\d+)', x).group(1)), reverse=True)
        missing_report.sort(key=lambda x: float(re.search(r'Int: (\d+)', x).group(1)), reverse=True)
        extra_report.sort(key=lambda x: float(re.search(r'Int: (\d+)', x).group(1)), reverse=True)

        num_target_total = len(target_peaks)
        num_pred_total = len(pred_peaks_dicts)
        num_missing = len(missing_report)
        
        if num_pred_total < num_target_total:
            limit_extra = len(extra_report)
        else:
            limit_extra = num_missing
            if limit_extra == 0 and len(extra_report) > 0:
                limit_extra = 3

        report_lines = ["[Analysis Report]"]
        report_lines.append("1. MATCHED Peaks (Correct Structures - KEEP THESE):")
        if matched_report: report_lines.extend(["   - " + s for s in matched_report]) 
        else: report_lines.append("   - None.")
        report_lines.append(f"\n2. EXTRA Peaks (Hallucinated Structures - REMOVE THESE) [Showing Top {limit_extra}]:")
        if extra_report: report_lines.extend(["   - " + s for s in extra_report[:limit_extra]])
        else: report_lines.append("   - None.")
        report_lines.append("\n3. MISSING Peaks (Target Signals - NEED TO ADD):")
        if missing_report: report_lines.extend(["   - " + s for s in missing_report])
        else: report_lines.append("   - None.")

        return "\n".join(report_lines), missing_mzs

    def initialize_with_candidates(self, candidates: List[str]):
        self.root = MCTSNode("VIRTUAL_ROOT")
        self.root.visits = 1
        self.root.q_value = 0.0
        
        unique_cands = []
        seen = set()
        for s in candidates:
            mol = Chem.MolFromSmiles(s)
            if mol:
                c = self._canonize(s)
                if c and c not in seen:
                    seen.add(c)
                    unique_cands.append(s)
        
        for s in unique_cands:
            node = MCTSNode(s, parent=self.root)
            self._simulate_hybrid(node)
            node.visits = 1
            node.q_value = node.reward
            self.root.children.append(node)
            if node.reward > self.root.q_value:
                self.root.q_value = node.reward

        self.root.is_expanded = True
        return len(self.root.children) > 0

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_expanded and node.children:
            node = max(node.children, key=lambda c: c.uct_score())
        return node

    def _expand(self, node: MCTSNode):
        if node.smiles == "VIRTUAL_ROOT": return []
        
        if self.search_mode == 3:
            if not node.pred_peaks_cache:
                node.pred_peaks_cache = self.cfmid.predict_spectrum(node.smiles, adduct=self.adduct)
            if node.pred_peaks_cache and 'smiles' not in node.pred_peaks_cache[0]:
                node.pred_peaks_cache = self.cfmid.annotate_peaks(node.smiles, node.pred_peaks_cache, adduct=self.adduct)
        
        if self.search_mode == 1:
            msg = self.unified_prompt.invoke({"formula": self.target_formula, "smiles": node.smiles})
        elif self.search_mode == 2:
            from prompt_engine import get_top_peaks_string
            t_str = get_top_peaks_string(self.target_peaks, top_k=15)
            msg = self.unified_prompt.invoke({"formula": self.target_formula, "smiles": node.smiles, "target_spectrum_raw": t_str})
        else:
            comp_text, missing_mzs = self._compare_spectra_text(self.target_peaks, node.pred_peaks_cache)
            ref_hints = self._get_reference_hints(missing_mzs)
            msg = self.unified_prompt.invoke({
                "formula": self.target_formula,
                "smiles": node.smiles,
                "spectrum_comparison": comp_text,
                "reference_hints": ref_hints
            })
        
        children = []
        max_llm_retries = 3
        
        for attempt in range(max_llm_retries):
            try:
                current_temp = 0.7 + (attempt * 0.1) 
                res = self.llm.bind(temperature=min(1.0, current_temp)).invoke(msg)
                text = res.content
                
                json_str = text
                if "```json" in text: json_str = re.search(r'```json(.*?)```', text, re.DOTALL).group(1)
                elif "```" in text: json_str = re.search(r'```(.*?)```', text, re.DOTALL).group(1)
                
                parsed = json.loads(json_str.strip())
                candidate_list = []
                if isinstance(parsed, list): candidate_list = parsed
                elif isinstance(parsed, dict): candidate_list = [parsed.get("smiles")]
                
                found_valid = False
                for new_smiles in candidate_list:
                    if not new_smiles or not isinstance(new_smiles, str): continue
                    
                    self.total_llm_proposals += 1
                    if self._validate_formula(new_smiles, strict_h=True):
                        self.valid_formula_proposals += 1
                    
                    if self.std_true_smiles and self.successful_mcts_prompt is None:
                        std_new = robust_standardize(new_smiles)
                        if std_new:
                            is_match = (std_new == self.std_true_smiles) or get_inchikey_match(std_new, self.std_true_smiles)
                            if is_match:
                                self.successful_mcts_prompt = msg.to_string() 
                                self.successful_mcts_response = text          

                    mol = Chem.MolFromSmiles(new_smiles)
                    if mol is not None:
                        c_new = self._canonize(new_smiles)
                        c_curr = self._canonize(node.smiles)
                        existing = {self._canonize(c.smiles) for c in node.children}
                        if c_new and c_new != c_curr and c_new not in existing:
                            child = MCTSNode(new_smiles, parent=node)
                            node.children.append(child)
                            children.append(child)
                            found_valid = True
                if found_valid: break
            except Exception:
                logger.warning(
                    "MCTS expansion failed for node=%r attempt=%s/%s",
                    node.smiles,
                    attempt + 1,
                    max_llm_retries,
                    exc_info=True,
                )
                continue

        node.is_expanded = True
        return children

    def _simulate_hybrid(self, node: MCTSNode) -> float:
        if node.reward is not None: return node.reward
        canon = self._canonize(node.smiles)
        
        if canon in self.visited_states:
            cached = self.visited_states[canon]
            base_score = cached["reward"]
            bonus = 0.05
            final = min(1.0, base_score + bonus)
            node.reward = final
            node.q_value = final
            node.pred_peaks_cache = cached["peaks"]
            return final
        
        pred = self.cfmid.predict_spectrum(node.smiles, adduct=self.adduct)
        node.pred_peaks_cache = pred
        score = self._calculate_hybrid_reward(node.smiles, pred)
        
        self.visited_states[canon] = {"reward": score, "peaks": pred}
        node.reward = score
        node.q_value = score
        return score

    def _backpropagate(self, node: MCTSNode, score: float):
        while node is not None:
            node.visits += 1
            if node.visits > 1:
                node.q_value = 0.5 * score + 0.5 * node.q_value
            score = node.q_value
            node = node.parent

    def search(self, n_iterations=10):
        if not self.root: return
        for i in range(n_iterations):
            if self.cfmid.has_timeout:
                break 

            leaf = self._select(self.root)
            if leaf.smiles != "VIRTUAL_ROOT":
                children = self._expand(leaf)
                
                if self.cfmid.has_timeout: break
                
                if children:
                    child = children[0]
                    score = self._simulate_hybrid(child)
                    
                    if leaf.reward is not None:
                        self.reward_deltas.append(score - leaf.reward)
                    
                    if self.cfmid.has_timeout: break
                    self._backpropagate(child, score)
                else:
                    self._backpropagate(leaf, 0.0)
            else: pass

    def get_refined_top_k(self):
        if not self.visited_states: return []
        
        candidates = []
        for smi, data in self.visited_states.items():
            is_perfect_formula = self._validate_formula(smi, strict_h=True)
            score = data["reward"]
            candidates.append({
                "smiles": smi,
                "score": score,
                "is_perfect": is_perfect_formula
            })
            
        candidates.sort(key=lambda x: (x['is_perfect'], x['score']), reverse=True)
        return [x['smiles'] for x in candidates[:self.top_k]]
        
    def get_search_metrics(self):
        vnr = (self.valid_formula_proposals / self.total_llm_proposals) if self.total_llm_proposals > 0 else 0.0
        avg_delta = (sum(self.reward_deltas) / len(self.reward_deltas)) if self.reward_deltas else 0.0
        return {"valid_node_rate": vnr, "avg_reward_delta": avg_delta}
