from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dataset_splits import (
    DATASET_SPLIT_PRESETS,
    RAW_DATA_OPTIONAL_COLUMNS,
    RAW_DATA_REQUIRED_COLUMNS,
    SPLIT_FILE_NAMES,
)


DEFAULT_RANDOM_STATE = 42
DEFAULT_SAMPLE_FRAC = 1.0
DEFAULT_TARGET_DRUG_RATIO = 0.1
DEFAULT_TARGET_DRUG_SELECTION = "uniform"
DEFAULT_SOURCE_VAL_RATIO = 0.2
DEFAULT_SOURCE_TEST_RATIO = 0.2
DEFAULT_TARGET_UNLABELED_RATIO = 0.9
DEFAULT_MIN_SAMPLES_THRESHOLD = 3


def resolve_build_arg(args, dataset_name: str, arg_name: str, fallback_value):
    value = getattr(args, arg_name)
    if value is not None:
        return value
    preset = DATASET_SPLIT_PRESETS[dataset_name]
    return preset.get("build_defaults", {}).get(arg_name, fallback_value)


def get_drug_counts(df: pd.DataFrame, drug_a_col: str, drug_b_col: str) -> pd.DataFrame:
    counts_a = df[drug_a_col].value_counts()
    counts_b = df[drug_b_col].value_counts()
    counts = counts_a.add(counts_b, fill_value=0).reset_index()
    counts.columns = ["drug", "sample_count"]
    return counts.sort_values("sample_count", ascending=False).reset_index(drop=True)


def select_target_drugs(drug_counts: pd.DataFrame, target_ratio: float, mode: str) -> tuple[set, str]:
    total_drugs = len(drug_counts)
    target_count = max(1, int(total_drugs * target_ratio))

    if mode == "high":
        selected = drug_counts.iloc[:target_count]["drug"].tolist()
        note = f"high: top {target_count}/{total_drugs} drugs by frequency"
    elif mode == "low":
        selected = drug_counts.iloc[-target_count:]["drug"].tolist()
        note = f"low: bottom {target_count}/{total_drugs} drugs by frequency"
    elif mode == "uniform":
        indices = np.linspace(0, total_drugs - 1, target_count, dtype=int)
        indices = np.unique(indices)
        selected = drug_counts.iloc[indices]["drug"].tolist()
        note = f"uniform: evenly sampled {len(selected)}/{total_drugs} drugs by rank"
    else:
        raise ValueError("target_drug_selection must be one of: high, low, uniform")

    return set(selected), note


def split_source_by_cell_3way(
    source_df: pd.DataFrame,
    cell_col: str,
    val_ratio: float,
    test_ratio: float,
    min_threshold: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("source val_ratio + test_ratio must be < 1")

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in source_df.groupby(cell_col):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        rows = len(group)

        if rows < min_threshold:
            train_parts.append(group)
            continue

        val_rows = int(round(rows * val_ratio))
        test_rows = int(round(rows * test_ratio))

        if rows >= 3:
            val_rows = max(1, val_rows)
            test_rows = max(1, test_rows)

        if val_rows + test_rows >= rows:
            if rows >= 3:
                val_rows, test_rows = 1, 1
            else:
                val_rows, test_rows = 1, 0

        train_rows = rows - val_rows - test_rows
        if train_rows <= 0:
            train_rows = 1
            if val_rows > 0:
                val_rows -= 1
            elif test_rows > 0:
                test_rows -= 1

        train_parts.append(group.iloc[:train_rows].copy())
        if val_rows > 0:
            val_parts.append(group.iloc[train_rows:train_rows + val_rows].copy())
        if test_rows > 0:
            test_parts.append(group.iloc[train_rows + val_rows:train_rows + val_rows + test_rows].copy())

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else source_df.iloc[0:0].copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else source_df.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else source_df.iloc[0:0].copy()
    return train_df, val_df, test_df


def split_target_by_cell_2way(
    target_df: pd.DataFrame,
    cell_col: str,
    unlabeled_ratio: float,
    min_threshold: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unlabeled_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in target_df.groupby(cell_col):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        rows = len(group)

        if rows < min_threshold:
            unlabeled_parts.append(group)
            continue

        unlabeled_rows = int(round(rows * unlabeled_ratio))
        unlabeled_rows = max(1, unlabeled_rows)
        if unlabeled_rows >= rows:
            unlabeled_rows = rows - 1

        unlabeled_parts.append(group.iloc[:unlabeled_rows].copy())
        test_parts.append(group.iloc[unlabeled_rows:].copy())

    unlabeled_df = pd.concat(unlabeled_parts, ignore_index=True) if unlabeled_parts else target_df.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else target_df.iloc[0:0].copy()
    return unlabeled_df, test_df


def ensure_columns(frame: pd.DataFrame, columns: list[str], csv_path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")


def standardize_input_dataframe(frame: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    ensure_columns(frame, RAW_DATA_REQUIRED_COLUMNS, csv_path)

    standardized = frame.copy()
    if "cell_line_name" not in standardized.columns:
        standardized["cell_line_name"] = standardized["cell_line_id"]
    if "drug_row" not in standardized.columns:
        standardized["drug_row"] = standardized["drug_row_smiles"]
    if "drug_col" not in standardized.columns:
        standardized["drug_col"] = standardized["drug_col_smiles"]

    for column in RAW_DATA_REQUIRED_COLUMNS + RAW_DATA_OPTIONAL_COLUMNS:
        if column in standardized.columns:
            standardized[column] = standardized[column].astype(str).where(
                standardized[column].notna(),
                standardized[column],
            )
    standardized["synergy_loewe"] = pd.to_numeric(standardized["synergy_loewe"], errors="coerce")
    return standardized


def build_dataset_split(
    dataset_name: str,
    data_root: Path,
    random_state: int,
    sample_frac: float,
    target_drug_ratio: float,
    target_drug_selection: str,
    source_val_ratio: float,
    source_test_ratio: float,
    target_unlabeled_ratio: float,
    min_samples_threshold: int,
    input_csv_override: Path | None = None,
    output_dir_override: Path | None = None,
) -> None:
    if input_csv_override is None or output_dir_override is None:
        preset = DATASET_SPLIT_PRESETS[dataset_name]
        input_csv = data_root / preset["input_csv"]
        output_dir = data_root / preset["split_dir"]
    else:
        input_csv = input_csv_override.resolve()
        output_dir = output_dir_override.resolve()

    data = pd.read_csv(input_csv, low_memory=False)
    data = standardize_input_dataframe(data, input_csv)
    original_rows = len(data)
    if sample_frac < 1.0:
        data = data.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    cell_col = "cell_line_id"
    drug_a_col = "drug_row"
    drug_b_col = "drug_col"

    drug_counts = get_drug_counts(data, drug_a_col, drug_b_col)
    target_drugs, selection_note = select_target_drugs(drug_counts, target_drug_ratio, target_drug_selection)

    is_target = data[drug_a_col].isin(target_drugs) | data[drug_b_col].isin(target_drugs)
    target_data = data[is_target].reset_index(drop=True)
    source_data = data[~is_target].reset_index(drop=True)

    leak = source_data[drug_a_col].isin(target_drugs) | source_data[drug_b_col].isin(target_drugs)
    if leak.any():
        raise RuntimeError("Detected target drugs inside source split")

    source_train, source_val, source_test = split_source_by_cell_3way(
        source_data,
        cell_col=cell_col,
        val_ratio=source_val_ratio,
        test_ratio=source_test_ratio,
        min_threshold=min_samples_threshold,
        seed=random_state,
    )
    target_train_unlabeled, target_test_labeled = split_target_by_cell_2way(
        target_data,
        cell_col=cell_col,
        unlabeled_ratio=target_unlabeled_ratio,
        min_threshold=max(2, min_samples_threshold - 1),
        seed=random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_frames = {
        "source_train_csv": source_train,
        "source_val_csv": source_val,
        "source_test_csv": source_test,
        "target_train_unlabeled_csv": target_train_unlabeled,
        "target_test_labeled_csv": target_test_labeled,
    }
    for arg_name, frame in split_frames.items():
        frame.to_csv(output_dir / SPLIT_FILE_NAMES[arg_name], index=False)

    meta = {
        "dataset_name": dataset_name,
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "random_state": random_state,
        "sample_frac": sample_frac,
        "target_drug_ratio": target_drug_ratio,
        "target_drug_selection": target_drug_selection,
        "target_drug_note": selection_note,
        "source_val_ratio": source_val_ratio,
        "source_test_ratio": source_test_ratio,
        "target_unlabeled_ratio": target_unlabeled_ratio,
        "min_samples_threshold": min_samples_threshold,
        "original_rows": original_rows,
        "used_rows": len(data),
        "n_total_cells": int(data[cell_col].nunique()),
        "n_total_drugs": int(len(drug_counts)),
        "n_target_drugs": int(len(target_drugs)),
        "split_stats": {
            "source_train": {"rows": int(len(source_train)), "cells": int(source_train[cell_col].nunique())},
            "source_val": {"rows": int(len(source_val)), "cells": int(source_val[cell_col].nunique())},
            "source_test": {"rows": int(len(source_test)), "cells": int(source_test[cell_col].nunique())},
            "target_train_unlabeled": {
                "rows": int(len(target_train_unlabeled)),
                "cells": int(target_train_unlabeled[cell_col].nunique()),
            },
            "target_test_labeled": {
                "rows": int(len(target_test_labeled)),
                "cells": int(target_test_labeled[cell_col].nunique()),
            },
        },
    }
    with (output_dir / "split_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    print(f"[done] {dataset_name} -> {output_dir}")
    for split_name, stats in meta["split_stats"].items():
        print(f"  {split_name}: rows={stats['rows']:,}, cells={stats['cells']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build preset splits or a custom split directory for DanSyn 2.0.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["all", *DATASET_SPLIT_PRESETS.keys()],
        default="all",
        help="Which preset split to build. Ignored when --input_csv and --output_dir are both provided.",
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--input_csv", type=str, default=None, help="Custom raw CSV to split.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for a custom split.")
    parser.add_argument("--custom_name", type=str, default="custom_dataset", help="Name stored in split_meta.json for a custom split.")
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--sample_frac", type=float, default=None)
    parser.add_argument("--target_drug_ratio", type=float, default=None)
    parser.add_argument("--target_drug_selection", type=str, choices=["high", "low", "uniform"], default=None)
    parser.add_argument("--source_val_ratio", type=float, default=None)
    parser.add_argument("--source_test_ratio", type=float, default=None)
    parser.add_argument("--target_unlabeled_ratio", type=float, default=None)
    parser.add_argument("--min_samples_threshold", type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    custom_mode = bool(args.input_csv or args.output_dir)
    if custom_mode and not (args.input_csv and args.output_dir):
        raise ValueError("--input_csv and --output_dir must be provided together for a custom split.")

    if custom_mode:
        build_dataset_split(
            dataset_name=args.custom_name,
            data_root=data_root,
            random_state=args.random_state,
            sample_frac=args.sample_frac if args.sample_frac is not None else DEFAULT_SAMPLE_FRAC,
            target_drug_ratio=args.target_drug_ratio if args.target_drug_ratio is not None else DEFAULT_TARGET_DRUG_RATIO,
            target_drug_selection=args.target_drug_selection if args.target_drug_selection is not None else DEFAULT_TARGET_DRUG_SELECTION,
            source_val_ratio=args.source_val_ratio if args.source_val_ratio is not None else DEFAULT_SOURCE_VAL_RATIO,
            source_test_ratio=args.source_test_ratio if args.source_test_ratio is not None else DEFAULT_SOURCE_TEST_RATIO,
            target_unlabeled_ratio=(
                args.target_unlabeled_ratio if args.target_unlabeled_ratio is not None else DEFAULT_TARGET_UNLABELED_RATIO
            ),
            min_samples_threshold=(
                args.min_samples_threshold if args.min_samples_threshold is not None else DEFAULT_MIN_SAMPLES_THRESHOLD
            ),
            input_csv_override=Path(args.input_csv).expanduser(),
            output_dir_override=Path(args.output_dir).expanduser(),
        )
        return

    dataset_names = list(DATASET_SPLIT_PRESETS.keys()) if args.dataset_name == "all" else [args.dataset_name]
    for dataset_name in dataset_names:
        build_dataset_split(
            dataset_name=dataset_name,
            data_root=data_root,
            random_state=args.random_state,
            sample_frac=resolve_build_arg(args, dataset_name, "sample_frac", DEFAULT_SAMPLE_FRAC),
            target_drug_ratio=resolve_build_arg(args, dataset_name, "target_drug_ratio", DEFAULT_TARGET_DRUG_RATIO),
            target_drug_selection=resolve_build_arg(
                args,
                dataset_name,
                "target_drug_selection",
                DEFAULT_TARGET_DRUG_SELECTION,
            ),
            source_val_ratio=resolve_build_arg(args, dataset_name, "source_val_ratio", DEFAULT_SOURCE_VAL_RATIO),
            source_test_ratio=resolve_build_arg(args, dataset_name, "source_test_ratio", DEFAULT_SOURCE_TEST_RATIO),
            target_unlabeled_ratio=resolve_build_arg(
                args,
                dataset_name,
                "target_unlabeled_ratio",
                DEFAULT_TARGET_UNLABELED_RATIO,
            ),
            min_samples_threshold=resolve_build_arg(
                args,
                dataset_name,
                "min_samples_threshold",
                DEFAULT_MIN_SAMPLES_THRESHOLD,
            ),
        )


if __name__ == "__main__":
    main()
