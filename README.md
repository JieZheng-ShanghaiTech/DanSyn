# DanSyn

DanSyn is a **domain-adaptive drug synergy prediction** project that supports both:

- **Supervised training** (standard setting)
- **Domain-Adversarial training (DANN)** for better generalization under **domain shift / cold-start** (e.g., new drugs)

This repository is organized to be **easy to run** with the provided DrugComb / DrugCombDB splits, and also easy to extend to your own datasets.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Optional Feature Extraction](#optional-feature-extraction)
- [Training](#training)
- [Citation](#citation)
- [License](#license)

---

## Repository Structure

```text
DanSyn/
├── data/
│   ├── dataset/                  # train/val/test splits (source/target)
│   │   ├── source_train_labeled.csv
│   │   ├── source_val_labeled.csv
│   │   ├── source_test_labeled.csv
│   │   ├── target_train_unlabeled.csv
│   │   └── target_test_labeled.csv
│   ├── ESPF/                     # ESPF tokenization + cached vectors
│   │   ├── info/
│   │   ├── dict.ipynb
│   │   └── ESPF_smiles_vectors.npy
│   ├── LLM/                      # LLM embedding scripts + cached vectors
│   │   ├── api.ipynb
│   │   ├── drug_desc_embedding.csv
│   │   └── llm_smiles_embeddings.npy
│   ├── cell_line_latent_values.csv
│   ├── cell_line_latent_values.npy
│   ├── divide.ipynb              # dataset switch / split helper
│   ├── drugcomb_288_146.csv
│   ├── drugcombDB_126_67.csv
│   └── unique_drugs_with_smiles.csv
├── main.py
├── model.py
├── utils.py
├── env.txt
├── LICENSE
└── README.md
```

---

## Quick Start

### 1) Environment

You need to install:

- Python >= 3.9 (recommended 3.10)
- PyTorch (+ CUDA if available)
- PyTorch Geometric
- numpy / pandas / scikit-learn / tqdm / etc.

> For reference, check `env.txt`.

---

### 2) Use provided dataset splits

The repo already provides the ready-to-run split files here:

```text
data/dataset/
  source_train_labeled.csv
  source_val_labeled.csv
  source_test_labeled.csv
  target_train_unlabeled.csv
  target_test_labeled.csv
```

Then you can run training directly (see [Training](#training)).

---

## Data Format

DanSyn expects dataset CSVs with consistent column names.

Typical columns include:

- `cell_line_id`
- `drug_row_smiles`
- `drug_col_smiles`
- `synergy_loewe`

> If you use your own dataset, **make sure your column names match** the provided CSV templates in `data/dataset/`.

---

## Optional Feature Extraction

If you run with the provided DrugComb / DrugCombDB setup, you can usually **skip** this section.

If you switch to a **custom dataset**, you should regenerate:

- unique drug table
- ESPF vectors (optional)
- LLM embeddings (optional)
- cell features (if not already prepared)

---

### 1) Unique drug table

From your dataset, extract unique drug entries into:

```text
data/unique_drugs_with_smiles.csv
```

This file is used to cache and reuse drug features.

---

### 2) ESPF structural features

ESPF encodes SMILES into explainable structural token sequences.

Notebook:
```text
data/ESPF/dict.ipynb
```

Output cache:
```text
data/ESPF/ESPF_smiles_vectors.npy
```

If you use the provided dataset + cached features, you can skip this step.

---

### 3) LLM semantic embeddings

This step uses LLM-generated drug descriptions (or drug semantic embeddings) and caches them.

Notebook:
```text
data/LLM/api.ipynb
```

Outputs:
```text
data/LLM/drug_desc_embedding.csv
data/LLM/llm_smiles_embeddings.npy
```

⚠️ You must provide your own **API key** and **base URL** inside `api.ipynb`.

If you use the provided cached embeddings, you can skip this step.

---

### 4) Cell features

The repo contains cached cell features:

```text
data/cell_line_latent_values.csv
data/cell_line_latent_values.npy
```

---

### 5) Switch dataset / re-split

Notebook:
```text
data/divide.ipynb
```

Use it to:
- switch between DrugComb / DrugCombDB
- regenerate your source/target splits for a new dataset

---

## Training

DanSyn supports two common modes.

### (A) Supervised only

```bash
python main.py   --tag baseline_sup   --epochs 150   --batch_size 32   --lr 1e-4   --weight_decay 0   --early_stop_patience 15   --eval_every 1
```

### (B) DANN enabled (Domain Adaptation)

```bash
python main.py   --tag dann_run   --use_dann   --adv_start_epoch 5   --adv_warmup_epochs 5   --adv_weight_max 1.0   --lambda_int 0.3   --epochs 10   --batch_size 32   --lr 1e-4   --weight_decay 0   --early_stop_patience 5   --eval_every 1
```

**Notes**
- `target_train_unlabeled.csv` is typically used only when `--use_dann` is enabled.
- More arguments are available in `main.py`. Adjust them to your needs.

---

## Citation

If you use this project in academic work, please cite:

```bibtex
@article{zhang2026dansyn,
  title   = {DanSyn: A Domain-Adaptive Framework with Hybrid Structural-Functional Representations for Robust Drug Synergy Prediction},
  author  = {Zhang, Ruoyin and Tao, Siyu and Feng, Yimiao and Zheng, Jie},
  journal = {Bioinformatics},
  year    = {2026}
}
```

---

## License

See `LICENSE`.
