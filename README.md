# MARS: LLM-Guided Retrieval and Search for Molecular Structure Elucidation from Mass Spectra
## Environment Setup

```bash
conda env create -f environment.yml
conda activate mars
```
## Dataset and Knowledge Base

To build the FAISS vector database, please follow these steps:

```
preprocess/test/generate_mgf_and_lables.py 
preprocess/test/fp_pred.py
```

You can directly use our pre-processed test sets (located in the `data/` directory, including `MolPuzzle_threshold_0.2.tsv` and `NPLIB1_threshold_0.2.tsv`).

If you wish to reproduce our data processing pipeline from scratch, please download the [MIST](https://github.com/samgoldman97/mist) model weights first, and then follow these steps:

```
preprocess/kb/fp_generate.py 
python preprocess/kb/generate_faiss.py
```

Note: The raw, manually transcribed data from the [MolPuzzle](https://github.com/KehanGuo2/MolPuzzle) dataset is stored in `data/Raw_MolPuzzle.json`.

## CFM-ID Deployment

Pull the CFM-ID Docker image before running the evaluation:

```
docker pull wishartlab/cfmid:latest
```

##  Evaluation

Run the evaluation on the test set:

```
python src/main.py
```