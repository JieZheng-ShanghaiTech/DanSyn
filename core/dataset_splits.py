from __future__ import annotations

import json
from pathlib import Path


SPLIT_FILE_NAMES = {
    "source_train_csv": "source_train_labeled.csv",
    "source_val_csv": "source_val_labeled.csv",
    "source_test_csv": "source_test_labeled.csv",
    "target_train_unlabeled_csv": "target_train_unlabeled.csv",
    "target_test_labeled_csv": "target_test_labeled.csv",
}

RAW_DATA_REQUIRED_COLUMNS = [
    "cell_line_id",
    "drug_row_smiles",
    "drug_col_smiles",
    "synergy_loewe",
]

RAW_DATA_OPTIONAL_COLUMNS = [
    "cell_line_name",
    "drug_row",
    "drug_col",
]

LABELED_SPLIT_REQUIRED_COLUMNS = [
    "cell_line_id",
    "drug_row_smiles",
    "drug_col_smiles",
    "synergy_loewe",
]

UNLABELED_SPLIT_REQUIRED_COLUMNS = [
    "cell_line_id",
    "drug_row_smiles",
    "drug_col_smiles",
]

SPLIT_OPTIONAL_COLUMNS = [
    "cell_line_name",
    "drug_row",
    "drug_col",
]


DATASET_SPLIT_PRESETS = {
    "drugcombdb_126_67": {
        "input_csv": "drugcombDB_126_67.csv",
        "split_dir": "datasets/drugcombdb_126_67",
        "description": "DrugCombDB preset built from drugcombDB_126_67.csv",
        "build_defaults": {
            "sample_frac": 1.0,
        },
    },
    "drugcomb_288_146": {
        "input_csv": "drugcomb_288_146.csv",
        "split_dir": "datasets/drugcomb_288_146_sf03",
        "description": "DrugComb preset built from drugcomb_288_146.csv with sample_frac=0.3",
        "build_defaults": {
            "sample_frac": 0.3,
        },
    },
}


def list_dataset_split_choices() -> list[str]:
    return list(DATASET_SPLIT_PRESETS.keys())


def _build_resolved_paths(split_dir: Path) -> dict[str, Path]:
    return {
        arg_name: split_dir / file_name
        for arg_name, file_name in SPLIT_FILE_NAMES.items()
    }


def _meta_matches_defaults(meta_path: Path, preset: dict) -> tuple[bool, str]:
    build_defaults = preset.get("build_defaults", {})
    if not build_defaults:
        return True, ""
    if not meta_path.exists():
        return False, f"missing metadata file: {meta_path}"

    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to read {meta_path}: {exc}"

    for key, expected_value in build_defaults.items():
        actual_value = meta.get(key)
        if isinstance(expected_value, float):
            try:
                actual_float = float(actual_value)
            except (TypeError, ValueError):
                return False, f"{key} expected {expected_value} but metadata has {actual_value}"
            if abs(actual_float - expected_value) > 1e-12:
                return False, f"{key} expected {expected_value} but metadata has {actual_value}"
        elif actual_value != expected_value:
            return False, f"{key} expected {expected_value} but metadata has {actual_value}"

    return True, ""


def get_dataset_split_paths(dataset_split: str, data_root: str | Path) -> dict:
    if dataset_split not in DATASET_SPLIT_PRESETS:
        valid = ", ".join(list_dataset_split_choices())
        raise ValueError(f"Unknown dataset_split={dataset_split}. Available: {valid}")

    data_root = Path(data_root)
    preset = DATASET_SPLIT_PRESETS[dataset_split]
    split_dir = data_root / preset["split_dir"]
    meta_path = split_dir / "split_meta.json"
    resolved_paths = _build_resolved_paths(split_dir)
    missing = [str(path) for path in resolved_paths.values() if not path.exists()]
    meta_ok, invalid_reason = _meta_matches_defaults(meta_path, preset)
    if not meta_ok and str(meta_path) not in missing:
        missing.append(str(meta_path))

    return {
        "name": dataset_split,
        "input_csv": str(data_root / preset["input_csv"]),
        "split_dir": str(split_dir),
        "description": preset.get("description", ""),
        "build_defaults": preset.get("build_defaults", {}),
        "paths": {key: str(value) for key, value in resolved_paths.items()},
        "missing": missing,
        "invalid_reason": invalid_reason,
    }


def get_dataset_split_paths_from_dir(split_dir: str | Path) -> dict:
    split_dir = Path(split_dir).expanduser().resolve()
    meta_path = split_dir / "split_meta.json"
    resolved_paths = _build_resolved_paths(split_dir)
    missing = [str(path) for path in resolved_paths.values() if not path.exists()]

    meta = {}
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            return {
                "name": split_dir.name,
                "input_csv": "",
                "split_dir": str(split_dir),
                "description": "Custom split directory",
                "build_defaults": {},
                "paths": {key: str(value) for key, value in resolved_paths.items()},
                "missing": missing,
                "invalid_reason": f"failed to read {meta_path}: {exc}",
                "is_custom": True,
            }

    return {
        "name": meta.get("dataset_name", split_dir.name),
        "input_csv": str(meta.get("input_csv", "")),
        "split_dir": str(split_dir),
        "description": "Custom split directory",
        "build_defaults": {},
        "paths": {key: str(value) for key, value in resolved_paths.items()},
        "missing": missing,
        "invalid_reason": "",
        "is_custom": True,
    }
