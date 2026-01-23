import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging


def prepare_device(num: int):
    """
    Gpu device selection
    """
    if torch.cuda.is_available():
        logging.info(f"CUDA is supported. CUDA version: {torch.version.cuda}")
        logging.info(f"Number of GPUs available: {torch.cuda.device_count()}")
        logging.info(f"Using device: cuda:{num} {torch.cuda.get_device_name(num)}")
        return torch.device(f"cuda:{num}")
    else:
        logging.info("CUDA is not supported in this PyTorch build.")
        return torch.device("cpu")


def pad_single_sequence(sequence, max_length: int, pad_value=0):
    """
    padding
    - list / np.ndarray / torch.Tensor
    - (padded_ids, attention_mask)
      attention_mask: 1=token, 0=pad
    """
    if isinstance(sequence, (np.ndarray, torch.Tensor)):
        sequence = sequence.tolist()

    if sequence is None:
        sequence = []

    if len(sequence) > max_length:
        return sequence[:max_length], [1] * max_length
    else:
        return (
            sequence + [pad_value] * (max_length - len(sequence)),
            [1] * len(sequence) + [0] * (max_length - len(sequence))
        )


class DrugSynergyDataset(Dataset):
    """
    SMILES as key

    essential columns
      - cell_line_id
      - drug_row_smiles
      - drug_col_smiles
      - synergy_loewe if has_label=True

    optional columns
      - cell_line_name
      - drug_row
      - drug_col

    dict
      - ESPF_dict: {smiles: token_id_list}
      - LLM_dict:  {smiles: np.ndarray(1536,)}
      - omics_latent: {cell_line_id: np.ndarray(dim,)}
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        ESPF_dict: dict,
        omics_latent: dict,
        LLM_dict: dict,
        domain_type="source",   # 'source' / 'target' -> 0/1
        has_label: bool = True,
        espf_max_len: int = 50
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.omics_latent = omics_latent
        self.ESPF_dict = ESPF_dict
        self.LLM_dict = LLM_dict
        self.has_label = bool(has_label)
        self.espf_max_len = int(espf_max_len)

        # domain_type -> 0/1
        if isinstance(domain_type, str):
            assert domain_type in ["source", "target"]
            self.domain_type = 0 if domain_type == "source" else 1
        else:
            self.domain_type = int(domain_type)

        # essential check
        need_cols = ["cell_line_id", "drug_row_smiles", "drug_col_smiles"]
        if self.has_label:
            need_cols.append("synergy_loewe")

        missing = [c for c in need_cols if c not in self.dataframe.columns]
        if missing:
            raise ValueError(f"DrugSynergyDataset missing columns: {missing}\nAll columns: {list(self.dataframe.columns)}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]

        # 1) SMILES（dict key）
        smilesA = str(row["drug_row_smiles"])
        smilesB = str(row["drug_col_smiles"])

        # 2) cell ID
        cell_id = str(row["cell_line_id"])

        # 3) name
        cell_name = str(row["cell_line_name"]) if "cell_line_name" in row.index and pd.notna(row["cell_line_name"]) else cell_id
        drugA_name = str(row["drug_row"]) if "drug_row" in row.index and pd.notna(row["drug_row"]) else smilesA
        drugB_name = str(row["drug_col"]) if "drug_col" in row.index and pd.notna(row["drug_col"]) else smilesB

        # 4) label
        if self.has_label:
            y = row["synergy_loewe"]
            y = float(y) if pd.notna(y) else 0.0
        else:
            y = 0.0

        label_tensor = torch.tensor(y, dtype=torch.float32)
        has_label_tensor = torch.tensor(1.0 if self.has_label else 0.0, dtype=torch.float32)

        # 5) domain label（DANN）
        domain_type_tensor = torch.tensor(self.domain_type, dtype=torch.long)

        # 6) cell vector
        omics_latent_vector = self.omics_latent[cell_id]
        omics_latent_tensor = torch.tensor(omics_latent_vector, dtype=torch.float32)

        # 7) ESPF token + mask
        espfA = self.ESPF_dict[smilesA]
        espfB = self.ESPF_dict[smilesB]
        espfA_pad, maskA = pad_single_sequence(espfA, self.espf_max_len, pad_value=0)
        espfB_pad, maskB = pad_single_sequence(espfB, self.espf_max_len, pad_value=0)

        ESPF_A = torch.tensor(espfA_pad, dtype=torch.long)
        ESPF_B = torch.tensor(espfB_pad, dtype=torch.long)
        mask_A = torch.tensor(maskA, dtype=torch.long)
        mask_B = torch.tensor(maskB, dtype=torch.long)

        # 8) LLM embedding
        llmA = self.LLM_dict[smilesA]  # np.ndarray (1536,)
        llmB = self.LLM_dict[smilesB]
        llm_A_tensor = torch.tensor(llmA, dtype=torch.float32)
        llm_B_tensor = torch.tensor(llmB, dtype=torch.float32)

        return {
            "ESPF_A": ESPF_A,
            "ESPF_B": ESPF_B,
            "mask_A": mask_A,
            "mask_B": mask_B,
            "omics_latent": omics_latent_tensor,

            "sample_id": cell_id,
            "label": label_tensor,
            "has_label": has_label_tensor,
            "domain_type": domain_type_tensor,

            "cell_line_name": cell_name,
            "drugA_name": drugA_name,
            "drugB_name": drugB_name,
            "smilesA": smilesA,
            "smilesB": smilesB,

            "llm_A": llm_A_tensor,
            "llm_B": llm_B_tensor,
        }


def load_data(
    source_train_csv: str,
    source_val_csv: str,
    source_test_csv: str,
    target_train_unlabeled_csv: str,
    target_test_labeled_csv: str,
    ESPF_dict: dict,
    omics_latent: dict,
    LLM_dict: dict,
    batch_size: int = 32,
    espf_max_len: int = 50,
):
    """
    DataLoader：
      - source train/val/test
      - target train_unlabeled / test_labeled
    """
    logging.info("Loading data begin")

    # 1) CSV
    src_train = pd.read_csv(source_train_csv, low_memory=False)
    src_val = pd.read_csv(source_val_csv, low_memory=False)
    src_test = pd.read_csv(source_test_csv, low_memory=False)
    tgt_unl = pd.read_csv(target_train_unlabeled_csv, low_memory=False)
    tgt_test = pd.read_csv(target_test_labeled_csv, low_memory=False)

    # 2) stats
    n_src_cells = src_train["cell_line_id"].nunique(dropna=True)
    logging.info(f"[source_train] rows={len(src_train):,} cells={n_src_cells:,}")
    logging.info(f"[source_val]   rows={len(src_val):,} cells={src_val['cell_line_id'].nunique(dropna=True):,}")
    logging.info(f"[source_test]  rows={len(src_test):,} cells={src_test['cell_line_id'].nunique(dropna=True):,}")
    logging.info(f"[target_unl]   rows={len(tgt_unl):,} cells={tgt_unl['cell_line_id'].nunique(dropna=True):,}")
    logging.info(f"[target_test]  rows={len(tgt_test):,} cells={tgt_test['cell_line_id'].nunique(dropna=True):,}")

    # val/test
    train_cells = set(src_train["cell_line_id"].astype(str).unique())
    val_cells = set(src_val["cell_line_id"].astype(str).unique())
    test_cells = set(src_test["cell_line_id"].astype(str).unique())

    unknown_in_val = list(val_cells - train_cells)
    if unknown_in_val:
        logging.warning(f"There are {len(unknown_in_val)} cell lines in the source domain validation set that do not appear in the source domain training set (example): {unknown_in_val[:10]}")

    unknown_in_test = list(test_cells - train_cells)
    if unknown_in_test:
        logging.warning(f"There are {len(unknown_in_test)} cell lines in the source domain test set that do not appear in the source domain training set (example): {unknown_in_test[:10]}")

    # 3) Dataset
    ds_src_train = DrugSynergyDataset(
        src_train, ESPF_dict, omics_latent, LLM_dict,
        domain_type="source", has_label=True, espf_max_len=espf_max_len
    )
    ds_src_val = DrugSynergyDataset(
        src_val, ESPF_dict, omics_latent, LLM_dict,
        domain_type="source", has_label=True, espf_max_len=espf_max_len
    )
    ds_src_test = DrugSynergyDataset(
        src_test, ESPF_dict, omics_latent, LLM_dict,
        domain_type="source", has_label=True, espf_max_len=espf_max_len
    )
    ds_tgt_unl = DrugSynergyDataset(
        tgt_unl, ESPF_dict, omics_latent, LLM_dict,
        domain_type="target", has_label=False, espf_max_len=espf_max_len
    )
    ds_tgt_test = DrugSynergyDataset(
        tgt_test, ESPF_dict, omics_latent, LLM_dict,
        domain_type="target", has_label=True, espf_max_len=espf_max_len
    )

    # 4) DataLoader
    src_train_loader = DataLoader(ds_src_train, batch_size=batch_size, shuffle=True)
    src_val_loader = DataLoader(ds_src_val, batch_size=batch_size, shuffle=False)
    src_test_loader = DataLoader(ds_src_test, batch_size=batch_size, shuffle=False)

    tgt_unl_loader = DataLoader(ds_tgt_unl, batch_size=batch_size, shuffle=True)
    tgt_test_loader = DataLoader(ds_tgt_test, batch_size=batch_size, shuffle=False)

    logging.info("Loading data end")

    # num_domains
    num_cells = int(n_src_cells)

    return (
        src_train_loader,
        src_val_loader,
        src_test_loader,
        tgt_unl_loader,
        tgt_test_loader,
        num_cells
    )
