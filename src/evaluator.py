import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np
from typing import List, Optional
import pulp

def robust_standardize(smiles: str, include_chirality: bool = False) -> str:
    if not smiles or not isinstance(smiles, str):
        return ""
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: 
            return ""
        
        try:
            lfc = rdMolStandardize.LargestFragmentChooser()
            mol = lfc.choose(mol)
        except Exception:
            pass 

        try:
            uc = rdMolStandardize.Uncharger()
            mol = uc.uncharge(mol)
        except Exception:
            pass

        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=include_chirality)
    except Exception:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=include_chirality)
        except:
            pass
        return ""

def get_inchikey_match(smiles1: str, smiles2: str) -> bool:
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if not mol1 or not mol2: return False
        
        key1 = Chem.MolToInchiKey(mol1)
        key2 = Chem.MolToInchiKey(mol2)
        
        return key1.split('-')[0] == key2.split('-')[0]
    except:
        return False

class MyopicMCES():
    def __init__(
        self,
        ind: int = 0,
        solver: str = pulp.listSolvers(onlyAvailable=True)[0],
        threshold: int = 15,
        always_stronger_bound: bool = True,
        solver_options: dict = None
    ):
        self.ind = ind
        self.solver = solver
        self.threshold = threshold
        self.always_stronger_bound = always_stronger_bound
        if solver_options is None:
            solver_options = dict(msg=0, timeLimit=10) 
        self.solver_options = solver_options

    def __call__(self, smiles_1: str, smiles_2: str) -> float:
        try:
            from myopic_mces.myopic_mces import MCES
            result = MCES(
                smiles1=smiles_1,
                smiles2=smiles_2,
                threshold=self.threshold,
                always_stronger_bound=self.always_stronger_bound,
                solver=self.solver,
                solver_options=self.solver_options
            )
            return float(result[1])
        except ImportError:
            print("Error: myopic_mces module not found.")
            return 100.0
        except Exception as e:
            return 100.0 

class MolecularEvaluator:
    def __init__(self, use_mces=True):
        self.mces_calculator = MyopicMCES() if use_mces else None

    def is_valid_smiles(self, smiles_str: str):
        if not isinstance(smiles_str, str): return False
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            return mol is not None
        except Exception:
            return False

    def compute_tanimoto(self, mol_pred, mol_true):
        try:
            fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, 2, nBits=2048)
            fp_true = AllChem.GetMorganFingerprintAsBitVect(mol_true, 2, nBits=2048)
            return DataStructs.TanimotoSimilarity(fp_pred, fp_true)
        except Exception:
            return 0.0

    def compute_mces(self, smiles_pred: str, smiles_true: str) -> float:
        if self.mces_calculator is None: return 100.0
        try:
            return self.mces_calculator(smiles_pred, smiles_true)
        except Exception:
            return 100.0

    def compute_metrics(self, true_smiles: str, pred_smiles: str):
        metrics = {
            "top1_valid_mols": 0,
            "top1_mol_accuracy": 0,
            "top1_tanimoto_sim": 0.0,
            "top1_mces_dist": 100.0 
        }

        std_pred = robust_standardize(pred_smiles, include_chirality=False)
        mol_pred_std = None
        if std_pred:
            mol_pred_std = Chem.MolFromSmiles(std_pred)
            
        if mol_pred_std:
            metrics["top1_valid_mols"] = 1
        else:
            return metrics

        std_true = robust_standardize(true_smiles, include_chirality=False)
        mol_true_std = None
        if std_true:
            mol_true_std = Chem.MolFromSmiles(std_true)

        if not mol_true_std:
            return metrics

        is_match = False
        if std_pred == std_true:
            is_match = True
        elif get_inchikey_match(std_pred, std_true):
            is_match = True

        if is_match:
            metrics.update({
                "top1_mol_accuracy": 1,
                "top1_tanimoto_sim": 1.0,
                "top1_mces_dist": 0.0
            })
            return metrics

        metrics["top1_tanimoto_sim"] = self.compute_tanimoto(mol_pred_std, mol_true_std)
        metrics["top1_mces_dist"] = self.compute_mces(std_pred, std_true)

        return metrics
    
    def compute_topk_metrics(self, true_smiles: str, pred_smiles_list: List[str]):
        metrics = {
            "topk_valid_mols": 0.0,
            "topk_mol_accuracy": 0,
            "topk_tanimoto_sim": 0.0,
            "topk_mces_dist": 100.0
        }

        if not pred_smiles_list:
            return metrics

        valid_preds = [] 
        for pred_smiles in pred_smiles_list:
            std_pred = robust_standardize(pred_smiles, include_chirality=False)
            if std_pred:
                mol = Chem.MolFromSmiles(std_pred)
                if mol:
                    valid_preds.append((std_pred, mol))
        
        metrics["topk_valid_mols"] = len(valid_preds) / len(pred_smiles_list)
        if not valid_preds:
            return metrics

        std_true = robust_standardize(true_smiles, include_chirality=False)
        mol_true_std = None
        if std_true:
            mol_true_std = Chem.MolFromSmiles(std_true)
        if not mol_true_std:
            return metrics

        hit = 0
        max_sim = 0.0
        mces_values = []

        for std_pred, mol_pred_std in valid_preds:
            is_match = False
            if std_pred == std_true:
                is_match = True
            elif get_inchikey_match(std_pred, std_true):
                is_match = True

            if is_match:
                hit = 1
                max_sim = 1.0
                mces_values.append(0.0)
            else:
                sim = self.compute_tanimoto(mol_pred_std, mol_true_std)
                max_sim = max(max_sim, sim)
                dist = self.compute_mces(std_pred, std_true)
                mces_values.append(dist)

        metrics["topk_mol_accuracy"] = hit
        metrics["topk_tanimoto_sim"] = max_sim
        
        if hit == 1:
            metrics["topk_mces_dist"] = 0.0
        elif mces_values:
            metrics["topk_mces_dist"] = min(mces_values)

        return metrics