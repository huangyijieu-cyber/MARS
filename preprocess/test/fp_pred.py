import pandas as pd
from pathlib import Path
import copy
import numpy as np
import torch
from tqdm import tqdm
import selfies as sf
from rdkit import Chem
import json
import sys
import os

sys.path.append("./mist/src") 

from mist.utils.plot_utils import *
import mist.subformulae.assign_subformulae as assign_subformulae
import mist.models.base as base
import mist.data.datasets as datasets
import mist.data.featurizers as featurizers

class MISTPredictor:
    def __init__(self, fp_ckpt, res_dir, mgf_input, labels):
        self.fp_ckpt = fp_ckpt
        self.res_dir = Path(res_dir)
        self.mgf_input = mgf_input
        self.labels = labels
        self.res_dir.mkdir(parents=True, exist_ok=True)
        self.subform_dir = self.res_dir / "subforms_fp"
        self.subform_dir.mkdir(exist_ok=True, parents=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_dataset = None

        self.load_model()

    def assign_subformulae(self):
        print(f"Starting subformula assignment, output directory: {self.subform_dir}")
        assign_subformulae.assign_subforms(
            spec_files=self.mgf_input,
            labels_file=self.labels,
            output_dir=self.subform_dir,
            mass_diff_thresh=20,
            max_formulae=50,
            num_workers=32,
            feature_id="FEATURE_ID",
            debug=False
        )

    def load_model(self):
        print(f"Loading model weights: {self.fp_ckpt}")
        fp_model = torch.load(self.fp_ckpt, map_location=self.device)
        main_hparams = fp_model["hyper_parameters"]
        self.kwargs = copy.deepcopy(main_hparams)
        self.kwargs['device'] = str(self.device)
        self.kwargs['num_workers'] = 0
        self.kwargs['subform_folder'] = self.subform_dir
        self.kwargs['labels_file'] = self.labels

        self.model = base.build_model(**self.kwargs)
        self.model.load_state_dict(fp_model["state_dict"])
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def prepare_dataset(self):
        self.kwargs["spec_features"] = self.model.spec_features(mode="test")
        self.kwargs['mol_features'] = "none"
        self.kwargs['allow_none_smiles'] = True
        paired_featurizer = featurizers.get_paired_featurizer(**self.kwargs)

        spectra_mol_pairs = datasets.get_paired_spectra(**self.kwargs)
        spectra_mol_pairs = list(zip(*spectra_mol_pairs))

        self.test_dataset = datasets.SpectraMolDataset(
            spectra_mol_list=spectra_mol_pairs, featurizer=paired_featurizer, **self.kwargs
        )

    def predict(self):
        self.prepare_dataset()
        print("Starting model prediction...")
        output_preds = (
            self.model.encode_all_spectras(self.test_dataset, no_grad=True, **self.kwargs).cpu().numpy()
        )
        output_names = self.test_dataset.get_spectra_names()
        return output_preds, output_names


if __name__ == "__main__":
    input_json_file = "data/Raw_MolPuzzle.json"
    save_dir_name = "MolPuzzle"
    base_data_path = Path(f"data/{save_dir_name}")
    mgf_input = base_data_path / f"{save_dir_name}.mgf"
    labels = base_data_path / f"{save_dir_name}_labels.tsv"
    res_dir = base_data_path / "mist_results"
    fp_ckpt = "preprocess/mist_fp_canopus_pretrain.ckpt" 

    if not os.path.exists(fp_ckpt):
        print(f"Error: model checkpoint file not found: {fp_ckpt}")
        print("Please modify the 'fp_ckpt' variable to point to the correct .ckpt file path.")
        exit(1)

    predictor = MISTPredictor(fp_ckpt, res_dir, mgf_input, labels)
    predictor.assign_subformulae()
    output_preds, output_names = predictor.predict()
    print(f"Prediction finished, output shape: {output_preds.shape}, number of samples: {len(output_names)}")

    print(f"Reading original metadata: {input_json_file}")
    with open(input_json_file, 'r') as f:
        json_data = json.load(f)
    df = pd.DataFrame(json_data)
    if 'id' in df.columns:
        df.rename(columns={'id': 'identifier'}, inplace=True)

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print(f"Processing threshold: {threshold} ...")
        indices_list = [np.where(row > threshold)[0].tolist() for row in output_preds]
        name_fps_keys = {name: fps for name, fps in zip(output_names, indices_list)}

        df_curr = df.copy()
        df_curr["fps"] = ""
        df_curr["selfies"] = ""

        to_drop = []

        for idx, row in df_curr.iterrows():
            identifier = str(row["identifier"])
            smiles = row["smiles"]

            if identifier not in name_fps_keys:
                to_drop.append(idx)
                print(f"Warning: Identifier {identifier} not found in predictions")
                continue

            fps_indices = name_fps_keys[identifier]

            vec = np.zeros(4096, dtype=np.uint8)
            vec[fps_indices] = 1
            df_curr.at[idx, "fps"] = json.dumps(vec.tolist())

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                    selfies_str = sf.encoder(canonical_smiles)
                    df_curr.at[idx, "selfies"] = selfies_str
                else:
                    print(f"Error: invalid SMILES: {smiles}")
                    to_drop.append(idx)
            except Exception as e:
                print(f"Error: SELFIES encoding failed ({identifier}): {e}")
                to_drop.append(idx)
                continue

        df_curr = df_curr.drop(index=to_drop)
        print(f"Threshold {threshold}: final number of records retained: {len(df_curr)}")

        output_file = res_dir / f"{save_dir_name}_fps_selfies_threshold_{threshold}.tsv"
        df_curr.to_csv(
            output_file,
            sep="\t",
            index=False
        )
        print(f"Saved: {output_file}")