import faiss
import numpy as np
import pandas as pd
import json
import os
import re
import pickle
from collections import defaultdict
from typing import List, Any, Dict, Set
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from rdkit import Chem 

class FaissDiceRetriever(BaseRetriever):
    packed_data: Any
    df_metadata: Any
    db_popcounts: Any
    
    hetero_sigs_map: Dict[str, List[int]]
    
    k: int = 5
    
    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _parse_formula_counts(formula: str) -> dict:
        if not isinstance(formula, str) or pd.isna(formula):
            return {}
        matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
        counts = defaultdict(int)
        for elem, num in matches:
            if elem in ['C', 'H']: continue 
            count = int(num) if num else 1
            counts[elem] += count
        return dict(counts)

    @staticmethod
    def _get_signature_from_counts(counts: dict) -> str:
        if not counts: return "HYDROCARBON"
        sorted_keys = sorted(counts.keys())
        return "".join([f"{k}{counts[k]}" for k in sorted_keys])

    @staticmethod
    def _get_signature(formula: str) -> str:
        c = FaissDiceRetriever._parse_formula_counts(formula)
        return FaissDiceRetriever._get_signature_from_counts(c)

    @staticmethod
    def _parse_sig_to_counts(sig: str) -> dict:
        if sig == "HYDROCARBON": return {}
        matches = re.findall(r'([A-Z][a-z]*)(\d+)', sig)
        return {elem: int(num) for elem, num in matches}

    @classmethod
    def load_from_dir(cls, data_dir, k=20):
        
        meta_path = os.path.join(data_dir, "metadata.parquet")
        sigs_cache_path = os.path.join(data_dir, "hetero_sigs_map.pkl")
        
        hetero_sigs_map = defaultdict(list)
        
        if os.path.exists(sigs_cache_path):
            with open(sigs_cache_path, "rb") as f: 
                hetero_sigs_map = pickle.load(f)
        else:
            print("Building hetero-indices from metadata (First run)...")
            df_meta = pd.read_parquet(meta_path)
            formulas = df_meta['formula'].fillna("").astype(str).values
            
            for idx, formula in enumerate(formulas):
                sig = cls._get_signature(formula)
                hetero_sigs_map[sig].append(idx)
            
            print("Saving caches...")
            with open(sigs_cache_path, "wb") as f: 
                pickle.dump(hetero_sigs_map, f)

        packed_path = os.path.join(data_dir, "packed_features.npy")
        packed_data = np.load(packed_path)
        
        pop_path = os.path.join(data_dir, "popcounts.npy")
        db_popcounts = np.load(pop_path)
        
        df_metadata = pd.read_parquet(meta_path)
        
        return cls(
            packed_data=packed_data,
            df_metadata=df_metadata,
            db_popcounts=db_popcounts,
            hetero_sigs_map=hetero_sigs_map,
            k=k
        )

    def _perform_subset_search(self, target_fp_list, candidate_indices, limit, tier_rank, tier_name) -> List[Document]:
        if len(candidate_indices) == 0 or limit <= 0:
            return []
            
        candidate_indices = np.array(candidate_indices, dtype=np.int64)
        subset_packed = self.packed_data[candidate_indices]
        subset_popcounts = self.db_popcounts[candidate_indices]
        
        try:
            query_arr = np.array([target_fp_list], dtype=np.uint8)
            query_popcount = np.sum(query_arr)
            query_packed = np.packbits(query_arr, axis=1)
            
            d = query_arr.shape[1]
            temp_index = faiss.IndexBinaryFlat(d)
            temp_index.add(subset_packed)
            
            buffer_factor = 10
            search_k = min(limit * buffer_factor + 500, len(subset_packed))

            hamming_distances, relative_indices = temp_index.search(query_packed, search_k)
            
            docs = []
            count = 0 

            for rank, rel_idx in enumerate(relative_indices[0]):
                if rel_idx == -1: continue
                
                global_idx = candidate_indices[rel_idx]
                meta = self.df_metadata.iloc[global_idx]
                retrieved_smiles = meta.get('smiles', '')

                h_dist = hamming_distances[0][rank]
                denom = query_popcount + subset_popcounts[rel_idx]
                dice_score = 0.0 if denom == 0 else 1.0 - (h_dist / denom)
                
                doc = Document(
                    page_content=retrieved_smiles,
                    metadata={
                        "score": float(dice_score),
                        "index": int(global_idx),
                        "formula": meta.get('formula', ''),
                        "smiles": retrieved_smiles,
                        "identifier": meta.get('identifier', ''),
                        "tier_rank": int(tier_rank),
                        "tier_name": tier_name
                    }
                )
                docs.append(doc)
                count += 1
                
                if count >= limit:
                    break
            
            docs.sort(key=lambda x: x.metadata['score'], reverse=True)
            return docs
            
        except Exception as e:
            print(f"Subset search error: {e}")
            return []

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        try:
            input_data = json.loads(query)
            target_fp_list = input_data.get("fps", [])
            target_formula = input_data.get("formula", "")
        except: return []

        target_counts = self._parse_formula_counts(target_formula)
        target_keys = set(target_counts.keys())
        target_total = sum(target_counts.values())
        
        pool_tier1 = [] 
        pool_tier2 = [] 
        pool_tier3 = [] 
        pool_tier4 = [] 
        
        for sig, indices in self.hetero_sigs_map.items():
            ref_counts = self._parse_sig_to_counts(sig)
            ref_keys = set(ref_counts.keys())
            
            if ref_keys == target_keys:
                ref_total = sum(ref_counts.values())
                if ref_counts == target_counts:
                    pool_tier1.extend(indices) 
                elif ref_total > target_total:
                    pool_tier2.extend(indices) 
                else:
                    pool_tier3.extend(indices) 
            elif ref_keys.issubset(target_keys):
                pool_tier4.extend(indices) 
            
        final_results = []
        remaining_k = self.k
        
        stages = [
            (pool_tier1, 0, "Exact"),
            (pool_tier2, 1, "More"),
            (pool_tier3, 2, "Less"),
            (pool_tier4, 3, "Subset")
        ]
        
        selected_indices = set()

        for pool, rank, name in stages:
            if remaining_k <= 0: break
            if not pool: continue
            
            docs = self._perform_subset_search(
                target_fp_list, 
                pool, 
                remaining_k, 
                rank, 
                name
            )
            
            for d in docs:
                final_results.append(d)
                selected_indices.add(d.metadata['index'])
            
            remaining_k = self.k - len(final_results)

        if remaining_k > 0:
            try:
                query_arr = np.array([target_fp_list], dtype=np.uint8)
                query_packed = np.packbits(query_arr, axis=1)
                
                d = query_arr.shape[1]
                temp_global = faiss.IndexBinaryFlat(d)
                temp_global.add(self.packed_data)
                
                safe_buffer_k = max(2000, self.k * 5)
                
                D, I = temp_global.search(query_packed, safe_buffer_k)
                
                tier5_docs = []
                for r, idx in enumerate(I[0]):
                    if idx == -1: continue
                    if idx in selected_indices: continue
                    
                    meta = self.df_metadata.iloc[idx]
                    smi = meta.get('smiles', '')

                    h_dist = D[0][r]
                    denom = np.sum(query_arr) + self.db_popcounts[idx]
                    dice = 0.0 if denom == 0 else 1.0 - h_dist/denom
                    
                    tier5_docs.append(Document(
                        page_content=smi,
                        metadata={
                            "score": float(dice),
                            "index": int(idx),
                            "formula": meta.get('formula',''),
                            "smiles": smi,
                            "tier_rank": 4, 
                            "tier_name": "Global"
                        }
                    ))
                    if len(tier5_docs) >= remaining_k: break
                
                tier5_docs.sort(key=lambda x: x.metadata['score'], reverse=True)
                final_results.extend(tier5_docs)
                
            except Exception as e:
                print(f"Global fallback error: {e}")

        return final_results[:self.k]