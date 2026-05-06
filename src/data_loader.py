import pandas as pd
import numpy as np
import json
import ast
import os

class MolecularDataLoader:
    def __init__(self):
        pass
    
    def _safe_parse_list(self, val):
        if isinstance(val, str):
            try:
                return json.loads(val)
            except:
                try:
                    return ast.literal_eval(val)
                except:
                    return []
        return val if isinstance(val, list) else []

    def _read_file(self, path, limit=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        if path.endswith('.tsv'):
            sep = '\t'
        else:
            sep = ','
            
        if limit:
            df = pd.read_csv(path, sep=sep, nrows=limit)
        else:
            df = pd.read_csv(path, sep=sep)
        return df

    def load_test_data(self, test_csv_path, limit=None):
        print(f"Loading Test Data from {test_csv_path}...")
        
        df = self._read_file(test_csv_path, limit=limit)
            
        if 'mzs' in df.columns:
            df['mzs'] = df['mzs'].apply(self._safe_parse_list)
        if 'intensities' in df.columns:
            df['intensities'] = df['intensities'].apply(self._safe_parse_list)
        
        print(f"Test Data Loaded. Samples: {len(df)}")
        return df