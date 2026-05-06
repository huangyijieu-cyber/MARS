import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import selfies as sf
from tqdm import tqdm

INPUT_RAW_DATA = "data/molecules/candidate_pools/MassSpecGym_retrieval_molecules_4M.tsv"
OUTPUT_CLEANED_DATA = "data/raw_kb.tsv"

def process_molecule(row):
    smiles = row['smiles']
    inchikey = row['inchikey']
    
    result = {
        'smiles': smiles,
        'inchikey': inchikey,
        'identifier': inchikey,
        'fps': None,
        'formula': None,
        'selfies': None
    }
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return result
            
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
        result['fps'] = json.dumps(list(fp))
        
        result['formula'] = rdMolDescriptors.CalcMolFormula(mol)
        
        try:
            result['selfies'] = sf.encoder(smiles)
        except:
            result['selfies'] = ""
            
    except Exception as e:
        pass
        
    return result

print("Generating fingerprints and metadata...")
df_raw = pd.read_csv(INPUT_RAW_DATA, sep='\t')

tqdm.pandas(desc="Processing molecules")
processed_data = df_raw.progress_apply(process_molecule, axis=1)

df_cleaned = pd.DataFrame(processed_data.tolist())

df_cleaned = df_cleaned.dropna(subset=['fps'])

df_cleaned.to_csv(OUTPUT_CLEANED_DATA, sep='\t', index=False)
print(f"Data preparation completed! Saved to {OUTPUT_CLEANED_DATA}")