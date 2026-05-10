import pandas as pd
import numpy as np
import json
import os
import gc
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from rdkit import Chem

logging.basicConfig(
    level=os.getenv("MARS_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_TSV = "data/raw_kb.tsv"
OUTPUT_DIR = "data/faiss_kb"

TARGET_FILE_1 = "data/MolPuzzle_threshold_0.2.tsv"
TARGET_FILE_2 = "data/NPLIB1_threshold_0.2.tsv"

CHUNK_SIZE = 50000

def get_2d_inchikey(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        Chem.WrapLogs()
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            inchikey = Chem.MolToInchiKey(mol)
            if inchikey:
                return inchikey.split('-')[0]
    except Exception:
        logger.debug("Failed to compute 2D InChIKey for smiles=%r", smiles, exc_info=True)
    return None

def build_blacklist_from_targets(*target_files):
    blacklist = set()
    print("=== Building exclusion blacklist ===")
    for file_path in target_files:
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: Target file not found: {file_path}")
            continue

        print(f"Processing target file: {file_path}")
        df = pd.read_csv(file_path, sep='\t', usecols=['smiles'])

        for smiles in tqdm(df['smiles'].dropna(), desc=f"Parsing {os.path.basename(file_path)}"):
            ik2d = get_2d_inchikey(smiles)
            if ik2d:
                blacklist.add(ik2d)

    print(f"✅ Blacklist built successfully! Total unique 2D InChIKeys to exclude: {len(blacklist)}\n")
    return blacklist

def process_chunk(chunk_df, fps_col):
    n = len(chunk_df)
    n_features = 4096

    temp_matrix = np.zeros((n, n_features), dtype=np.uint8)
    valid_indices = []

    fps_series = chunk_df[fps_col].tolist()

    for i, val in enumerate(fps_series):
        try:
            if isinstance(val, str):
                vec = json.loads(val)
            else:
                vec = val
            temp_matrix[i, :] = vec
            valid_indices.append(i)
        except Exception:
            logger.debug("Skipping malformed fingerprint row index=%s value=%r", i, val, exc_info=True)
            continue

    if len(valid_indices) < n:
        temp_matrix = temp_matrix[valid_indices]
        chunk_df = chunk_df.iloc[valid_indices]

    popcounts = np.sum(temp_matrix, axis=1).astype(np.int32)

    packed = np.packbits(temp_matrix, axis=1)

    meta_cols = [c for c in ['selfies', 'smiles', 'formula', 'identifier'] if c in chunk_df.columns]
    metadata = chunk_df[meta_cols].copy()

    return packed, popcounts, metadata

def generate_faiss_assets():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    blacklist = build_blacklist_from_targets(TARGET_FILE_1, TARGET_FILE_2)

    seen_inchikeys = set()

    print(f"Starting streaming process for: {INPUT_TSV}")
    print(f"Chunk size: {CHUNK_SIZE}")

    all_packed = []
    all_popcounts = []

    metadata_file = os.path.join(OUTPUT_DIR, "metadata.parquet")
    parquet_writer = None

    try:
        reader = pd.read_csv(INPUT_TSV, sep='\t', chunksize=CHUNK_SIZE, engine='pyarrow')
    except Exception:
        logger.warning("PyArrow chunking failed, falling back to default pandas engine.", exc_info=True)
        reader = pd.read_csv(INPUT_TSV, sep='\t', chunksize=CHUNK_SIZE)

    total_processed = 0
    total_skipped = 0

    for chunk in tqdm(reader, desc="Processing chunks"):
        cols_map = {c.lower(): c for c in chunk.columns}
        if 'fps' not in cols_map or 'smiles' not in cols_map:
            raise ValueError("Columns 'fps' or 'smiles' not found in the dataset.")

        fps_col = cols_map['fps']
        smiles_col = cols_map['smiles']

        initial_len = len(chunk)

        chunk['inchikey_2d'] = chunk[smiles_col].apply(get_2d_inchikey)

        chunk = chunk.dropna(subset=['inchikey_2d'])

        chunk = chunk.drop_duplicates(subset=['inchikey_2d'])

        chunk = chunk[~chunk['inchikey_2d'].isin(blacklist)]

        chunk = chunk[~chunk['inchikey_2d'].isin(seen_inchikeys)]

        seen_inchikeys.update(chunk['inchikey_2d'].tolist())

        total_skipped += (initial_len - len(chunk))

        if len(chunk) == 0:
            continue

        packed, popcounts, meta_df = process_chunk(chunk, fps_col)

        all_packed.append(packed)
        all_popcounts.append(popcounts)

        table = pa.Table.from_pandas(meta_df)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(metadata_file, table.schema, compression='snappy')

        parquet_writer.write_table(table)

        total_processed += len(meta_df)

        del chunk, packed, popcounts, meta_df, table
        gc.collect()

    if parquet_writer:
        parquet_writer.close()

    print(f"\nMetadata successfully streamed to: {metadata_file}")

    print("Concatenating binary arrays...")
    final_packed = np.concatenate(all_packed, axis=0)
    final_popcounts = np.concatenate(all_popcounts, axis=0)

    print(f"Saving binary files (Shape: {final_packed.shape})...")
    np.save(os.path.join(OUTPUT_DIR, "packed_features.npy"), final_packed)
    np.save(os.path.join(OUTPUT_DIR, "popcounts.npy"), final_popcounts)

    print(f"\n=== Process Completed ===")
    print(f"Total rows successfully saved: {total_processed}")
    print(f"Total rows skipped (Blacklisted/Duplicates/Invalid): {total_skipped}")

if __name__ == "__main__":
    generate_faiss_assets()
