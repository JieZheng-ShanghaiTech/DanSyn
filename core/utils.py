from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def prepare_device(device_index: int) -> torch.device:
    if torch.cuda.is_available():
        logging.info("CUDA is available: %s", torch.version.cuda)
        logging.info("Using device cuda:%s %s", device_index, torch.cuda.get_device_name(device_index))
        return torch.device(f"cuda:{device_index}")
    logging.info("CUDA is not available. Falling back to CPU.")
    return torch.device("cpu")


def pad_single_sequence(sequence, max_length: int, pad_value: int = 0) -> tuple[list[int], list[int]]:
    if isinstance(sequence, (np.ndarray, torch.Tensor)):
        sequence = sequence.tolist()
    if sequence is None:
        sequence = []

    if len(sequence) > max_length:
        return sequence[:max_length], [1] * max_length

    return (
        sequence + [pad_value] * (max_length - len(sequence)),
        [1] * len(sequence) + [0] * (max_length - len(sequence)),
    )


class DrugSynergyDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        espf_dict: dict,
        omics_latent: dict,
        llm_dict: dict,
        domain_type: str,
        has_label: bool,
        espf_max_len: int = 50,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.espf_dict = espf_dict
        self.omics_latent = omics_latent
        self.llm_dict = llm_dict
        self.has_label = bool(has_label)
        self.espf_max_len = int(espf_max_len)
        self.domain_type = 0 if domain_type == "source" else 1

        required_columns = ["cell_line_id", "drug_row_smiles", "drug_col_smiles"]
        if self.has_label:
            required_columns.append("synergy_loewe")
        missing = [column for column in required_columns if column not in self.dataframe.columns]
        if missing:
            raise ValueError(
                f"DrugSynergyDataset is missing required columns {missing}. "
                f"Found columns: {list(self.dataframe.columns)}"
            )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]

        smiles_a = str(row["drug_row_smiles"])
        smiles_b = str(row["drug_col_smiles"])
        cell_id = str(row["cell_line_id"])

        cell_name = (
            str(row["cell_line_name"])
            if "cell_line_name" in row.index and pd.notna(row["cell_line_name"])
            else cell_id
        )
        drug_a_name = str(row["drug_row"]) if "drug_row" in row.index and pd.notna(row["drug_row"]) else smiles_a
        drug_b_name = str(row["drug_col"]) if "drug_col" in row.index and pd.notna(row["drug_col"]) else smiles_b

        label_value = float(row["synergy_loewe"]) if self.has_label and pd.notna(row["synergy_loewe"]) else 0.0
        omics_latent_vector = self.omics_latent[cell_id]
        llm_a = self.llm_dict[smiles_a]
        llm_b = self.llm_dict[smiles_b]
        espf_a = self.espf_dict[smiles_a]
        espf_b = self.espf_dict[smiles_b]
        espf_a_pad, mask_a = pad_single_sequence(espf_a, self.espf_max_len, pad_value=0)
        espf_b_pad, mask_b = pad_single_sequence(espf_b, self.espf_max_len, pad_value=0)

        return {
            "ESPF_A": torch.tensor(espf_a_pad, dtype=torch.long),
            "ESPF_B": torch.tensor(espf_b_pad, dtype=torch.long),
            "mask_A": torch.tensor(mask_a, dtype=torch.long),
            "mask_B": torch.tensor(mask_b, dtype=torch.long),
            "omics_latent": torch.tensor(omics_latent_vector, dtype=torch.float32),
            "llm_A": torch.tensor(llm_a, dtype=torch.float32),
            "llm_B": torch.tensor(llm_b, dtype=torch.float32),
            "label": torch.tensor(label_value, dtype=torch.float32),
            "has_label": torch.tensor(1.0 if self.has_label else 0.0, dtype=torch.float32),
            "domain_type": torch.tensor(self.domain_type, dtype=torch.long),
            "sample_id": cell_id,
            "cell_line_name": cell_name,
            "drugA_name": drug_a_name,
            "drugB_name": drug_b_name,
            "smilesA": smiles_a,
            "smilesB": smiles_b,
        }


def load_data(
    *,
    source_train_csv: str,
    source_val_csv: str,
    source_test_csv: str,
    target_train_unlabeled_csv: str,
    target_test_labeled_csv: str,
    espf_dict: dict,
    omics_latent: dict,
    llm_dict: dict,
    batch_size: int,
    espf_max_len: int = 50,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    logging.info("Loading data frames.")

    source_train = pd.read_csv(source_train_csv, low_memory=False)
    source_val = pd.read_csv(source_val_csv, low_memory=False)
    source_test = pd.read_csv(source_test_csv, low_memory=False)
    target_unlabeled = pd.read_csv(target_train_unlabeled_csv, low_memory=False)
    target_test = pd.read_csv(target_test_labeled_csv, low_memory=False)

    logging.info(
        "Rows | source_train=%s source_val=%s source_test=%s target_unlabeled=%s target_test=%s",
        len(source_train),
        len(source_val),
        len(source_test),
        len(target_unlabeled),
        len(target_test),
    )

    source_train_loader = DataLoader(
        DrugSynergyDataset(
            source_train,
            espf_dict=espf_dict,
            omics_latent=omics_latent,
            llm_dict=llm_dict,
            domain_type="source",
            has_label=True,
            espf_max_len=espf_max_len,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    source_val_loader = DataLoader(
        DrugSynergyDataset(
            source_val,
            espf_dict=espf_dict,
            omics_latent=omics_latent,
            llm_dict=llm_dict,
            domain_type="source",
            has_label=True,
            espf_max_len=espf_max_len,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    source_test_loader = DataLoader(
        DrugSynergyDataset(
            source_test,
            espf_dict=espf_dict,
            omics_latent=omics_latent,
            llm_dict=llm_dict,
            domain_type="source",
            has_label=True,
            espf_max_len=espf_max_len,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    target_unlabeled_loader = DataLoader(
        DrugSynergyDataset(
            target_unlabeled,
            espf_dict=espf_dict,
            omics_latent=omics_latent,
            llm_dict=llm_dict,
            domain_type="target",
            has_label=False,
            espf_max_len=espf_max_len,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    target_test_loader = DataLoader(
        DrugSynergyDataset(
            target_test,
            espf_dict=espf_dict,
            omics_latent=omics_latent,
            llm_dict=llm_dict,
            domain_type="target",
            has_label=True,
            espf_max_len=espf_max_len,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        source_train_loader,
        source_val_loader,
        source_test_loader,
        target_unlabeled_loader,
        target_test_loader,
    )
