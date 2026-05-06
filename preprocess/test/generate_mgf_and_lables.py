import json
import os
from pathlib import Path
from typing import Tuple, List, Optional
from itertools import groupby

from tqdm import tqdm
import numpy as np
import pandas as pd


def build_mgf_str(
    meta_spec_list: List[Tuple[dict, List[Tuple[str, np.ndarray]]]],
    merge_charges=True,
    parent_mass_keys=["PEPMASS", "parentmass", "PRECURSOR_MZ"],
) -> str:
    entries = []
    for meta, spec in tqdm(meta_spec_list):
        str_rows = ["BEGIN IONS"]

        for i in parent_mass_keys:
            if i in meta:
                pep_mass = float(meta.get(i, -100))
                str_rows.append(f"PEPMASS={pep_mass}")
                break

        for k, v in meta.items():
            str_rows.append(f"{k.upper().replace(' ', '_')}={v}")

        if merge_charges:
            spec_ar = np.vstack([i[1] for i in spec])
            spec_ar = np.vstack([i for i in sorted(spec_ar, key=lambda x: x[0])])
        else:
            raise NotImplementedError()

        str_rows.extend([f"{i} {j}" for i, j in spec_ar])
        str_rows.append("END IONS")

        str_out = "\n".join(str_rows)
        entries.append(str_out)

    full_out = "\n\n".join(entries)
    return full_out


def parse_spectra_mgf(
    mgf_file: str, max_num: Optional[int] = None
) -> List[Tuple[dict, List[Tuple[str, np.ndarray]]]]:
    key = lambda x: x.strip() == "BEGIN IONS"
    parsed_spectra = []
    with open(mgf_file, "r") as fp:

        for (is_header, group) in tqdm(groupby(fp, key)):
            if is_header:
                continue
            meta = dict()
            spectra = []
            cur_spectra_name = "spec"
            cur_spectra = []
            group = list(group)
            for line in group:
                line = line.strip()
                if not line:
                    pass
                elif line == "END IONS" or line == "BEGIN IONS":
                    pass
                elif "=" in line:
                    k, v = [i.strip() for i in line.split("=", 1)]
                    meta[k] = v
                else:
                    mz, intens = line.split()
                    cur_spectra.append((float(mz), float(intens)))

            if len(cur_spectra) > 0:
                cur_spectra = np.vstack(cur_spectra)
                spectra.append((cur_spectra_name, cur_spectra))
                parsed_spectra.append((meta, spectra))
            else:
                pass

            if max_num is not None and len(parsed_spectra) > max_num:
                break
        return parsed_spectra


if __name__ == "__main__":
    input_file = "data/Raw_MolPuzzle.json"
    save_dir_name = "MolPuzzle"

    save_path = Path(f"data/{save_dir_name}")
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_file} ...")
    with open(input_file, 'r') as f:
        data_list = json.load(f)

    meta_spec_list: List[Tuple[dict, List[Tuple[str, np.ndarray]]]] = []
    label_entries = []

    print("Entries before processing: ", len(data_list))

    ION_LST = [
        "[M+H]+",
        "[M+Na]+",
        "[M+K]+",
        "[M-H2O+H]+",
        "[M+H3N+H]+",
        "[M]+",
        "[M-H4O2+H]+",
    ]

    for item in data_list:
        adduct = item.get("adduct")
        if adduct not in ION_LST:
            print(f"Skipping entry {item.get('id')}: Adduct {adduct} not in allowed list")
            continue

        meta = {
            "FEATURE_ID": item["id"],
            "adduct": item["adduct"],
            "collision_energy": item.get("collision_energy", "35"),
            "parentmass": float(item["precursor_mz"]),
        }

        peaks_list = item["peaks"]

        if len(peaks_list) == 0:
            print(f"Skipping entry {item.get('id')}: Empty peak list")
            continue

        peaks = np.array(peaks_list, dtype=float)

        if peaks.ndim != 2 or peaks.shape[1] != 2:
            print(f"Error: Incorrect peaks format for entry {item.get('id')}")
            continue

        spec = [("ms2", peaks)]
        meta_spec_list.append((meta, spec))

        label_entry = {
            "spec": str(item["id"]),
            "formula": item["formula"],
            "ionization": item["adduct"],
            "dataset": save_dir_name,
            "compound": f"{item['id']}",
            "parentmass": float(item["precursor_mz"]),
            "instrument": item.get("instrument_type", "unknown"),
        }
        label_entries.append(label_entry)

    mgf_output = build_mgf_str(meta_spec_list)
    mgf_path = save_path / f"{save_dir_name}.mgf"
    with open(mgf_path, "w") as f:
        f.write(mgf_output)
    print(f"MGF file generated: {mgf_path}")

    label_df = pd.DataFrame(label_entries)
    print("Entries after processing: ", len(label_entries))

    cols = ["spec", "formula", "ionization", "dataset", "compound", "parentmass", "instrument"]
    label_df = label_df[cols]

    tsv_path = save_path / f"{save_dir_name}_labels.tsv"
    label_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"Label file generated: {tsv_path}")