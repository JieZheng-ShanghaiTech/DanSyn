# DanSyn

DanSyn predicts drug synergy with `ESPF`, cell latent features, cross attention,
optional `DANN`, and `gpt_3_5_turbo` drug features.

The main entry is `main.py`.

## 1. Run The Main Experiments

Built-in datasets:

- `drugcombdb_126_67`
- `drugcomb_288_146`

Run DrugCombDB:

```bash
python main.py --tag db_supervised --dataset_split drugcombdb_126_67
python main.py --tag db_dann --dataset_split drugcombdb_126_67 --use_dann
```

Run DrugComb:

```bash
python main.py --tag dc_supervised --dataset_split drugcomb_288_146
python main.py --tag dc_dann --dataset_split drugcomb_288_146 --use_dann
```

`--use_dann` enables the domain generalization setting.

Run all four preset jobs:

```bash
python scripts/run_main_scenarios.py
```

Preview the commands only:

```bash
python scripts/run_main_scenarios.py --dry_run
```

## 2. Environment

```bash
conda create --name dansyn --file env.txt
conda activate dansyn
```

## 3. Use Your Own Data

If you already have a custom split folder, train with:

```bash
python main.py \
  --tag my_dataset_run \
  --dataset_split_dir data/datasets/my_dataset_split \
  --cell_latent_npy data/cell_line_latent_values.npy \
  --espf_npy data/ESPF/ESPF_smiles_vectors.npy \
  --llm_npy data/LLM/gpt_3_5_turbo/llm_smiles_embeddings.npy
```

If your files are stored elsewhere, add `--data_root your_data_dir`.

### 3.1 Raw CSV Format

Each row should represent one drug pair on one cell line.

Required columns:

- `cell_line_id`
- `drug_row_smiles`
- `drug_col_smiles`
- `synergy_loewe`

Recommended columns:

- `cell_line_name`
- `drug_row`
- `drug_col`

Example:

```csv
cell_line_id,cell_line_name,drug_row,drug_col,drug_row_smiles,drug_col_smiles,synergy_loewe
A375,A375,Erlotinib,Trametinib,C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1,CNc1nc(Nc2ccc(I)cc2F)c2ncc(C(=O)N3CCN(C)CC3)cc2n1,12.4
```

### 3.2 Build A Custom Split

```bash
python tools/datasets/build_dataset_splits.py \
  --input_csv data/my_dataset.csv \
  --output_dir data/datasets/my_dataset_split \
  --custom_name my_dataset
```

The split folder must contain:

- `source_train_labeled.csv`
- `source_val_labeled.csv`
- `source_test_labeled.csv`
- `target_train_unlabeled.csv`
- `target_test_labeled.csv`

## 4. Build Features For New Data

### 4.1 ESPF

```bash
python tools/espf/build_espf_features.py \
  --input_csvs data/my_dataset.csv \
  --output_npy data/ESPF/ESPF_smiles_vectors.npy
```

Multiple CSV files are also supported:

```bash
python tools/espf/build_espf_features.py \
  --input_csvs train.csv val.csv test.csv \
  --output_npy data/ESPF/ESPF_smiles_vectors.npy
```

### 4.2 LLM Features

Preview the resolved drug table without calling any API:

```bash
python tools/llm/build_llm_features.py \
  --input_csvs data/my_dataset.csv \
  --dry_run
```

Generate LLM features with your own API key:

```bash
python tools/llm/build_llm_features.py \
  --input_csvs data/my_dataset.csv \
  --api_key YOUR_API_KEY
```

If needed, you can also pass a custom `--base_url`.

### 4.3 Cell Features

This project directly uses `cell_line_latent_values.npy`.

If you want to replace it, prepare your own file with:

- key: `cell_line_id`
- value: 1D latent vector
- one consistent latent dimension for all cells

## 5. Outputs

Each run writes a timestamped folder under `results/` containing:

- `run_config.json`
- `data_stats.json`
- `epoch_metrics.csv`
- `train_summary.json`
- `best_model.pth`
- `last_model.pth`
- `source_test_predictions.csv`
- `target_test_predictions.csv`
- `test_metrics.json`
- `test_metrics.csv`

`test_metrics.csv` includes both `source_test` and `target_test`.

## 6. Data Layout

By default, `main.py` reads from `data/`.

Default feature files:

```text
data/cell_line_latent_values.npy
data/ESPF/ESPF_smiles_vectors.npy
data/LLM/gpt_3_5_turbo/llm_smiles_embeddings.npy
```

Preset split directories:

```text
data/datasets/drugcombdb_126_67/
data/datasets/drugcomb_288_146_sf03/
```
