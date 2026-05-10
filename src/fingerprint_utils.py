from rdkit import Chem
from rdkit.Chem import AllChem
import logging

logger = logging.getLogger(__name__)

def get_substructure_smiles(mol, atom_idx, radius):
    if radius == 0:
        return mol.GetAtomWithIdx(atom_idx).GetSymbol()
    
    try:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
        amap = {}
        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
        smi = Chem.MolToSmiles(submol, isomericSmiles=False, canonical=True)
        return smi
    except Exception:
        logger.debug("Failed to extract substructure atom_idx=%s radius=%s", atom_idx, radius, exc_info=True)
        return ""

def explain_shared_bits(retrieved_mol, shared_bits_list):
    if retrieved_mol is None:
        return []
    
    try:
        bi = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(retrieved_mol, radius=2, nBits=4096, bitInfo=bi)
        
        explained_fragments = set()
        
        for bit in shared_bits_list:
            if bit in bi:
                atom_idx, radius = bi[bit][0]
                frag_smi = get_substructure_smiles(retrieved_mol, atom_idx, radius)
                if len(frag_smi) > 1: 
                    explained_fragments.add(frag_smi)
        
        sorted_frags = sorted(list(explained_fragments), key=len, reverse=True)
        return sorted_frags[:8]
    except Exception:
        logger.warning("Failed to explain shared fingerprint bits", exc_info=True)
        return []
