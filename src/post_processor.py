import json
import re
import logging
from evaluator import robust_standardize
from rdkit import Chem

logger = logging.getLogger(__name__)

class MolecularPostProcessor:
    def __init__(self):
        pass

    def _extract_text_safe(self, gen_item):
        if isinstance(gen_item, str): return gen_item
        if hasattr(gen_item, 'content'): return gen_item.content
        if isinstance(gen_item, dict): return gen_item.get('text', '')
        return str(gen_item)

    def _clean_llm_json_output(self, text):
        if not text: return []
        text = str(text).strip()
        
        match = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
        if not match:
            match = re.search(r"```\s*(\[.*?\])\s*```", text, re.DOTALL)
            
        candidate_str = match.group(1) if match else text
        
        try:
            if not candidate_str.strip().startswith('['): 
                s = candidate_str.find('[')
                e = candidate_str.rfind(']')
                if s != -1 and e != -1: candidate_str = candidate_str[s:e+1]

            res = json.loads(candidate_str)
            return res if isinstance(res, list) else [str(res)]
        except Exception:
            logger.warning("Failed to parse LLM output as JSON; falling back to quoted-string extraction.", exc_info=True)
            return re.findall(r'"([^"]+)"', candidate_str)

    def process_generations(self, generations_list, target_formula=None, **kwargs):
        raw_strings = []
        for gen in generations_list:
            text_output = self._extract_text_safe(gen)
            candidates = self._clean_llm_json_output(text_output)
            raw_strings.extend([str(c).strip() for c in candidates if c])

        unique_proposed = set(raw_strings)
        num_proposed = len(unique_proposed)

        unique_valid_candidates = []
        
        for cand in unique_proposed:
            mol = Chem.MolFromSmiles(cand)
            if mol:
                canon_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                if canon_smi not in unique_valid_candidates:
                    unique_valid_candidates.append(canon_smi)
        
        num_valid = len(unique_valid_candidates)
        stats = {
            "proposed": num_proposed,
            "valid": num_valid
        }
        
        return stats, unique_valid_candidates
