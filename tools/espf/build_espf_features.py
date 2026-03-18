from __future__ import annotations

import argparse
import codecs
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from subword_nmt.apply_bpe import BPE

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_default_path(path_value: str | None, default_relative_path: str) -> Path:
    if path_value:
        return Path(path_value).expanduser().resolve()
    return (ROOT / default_relative_path).resolve()


def load_vocab(subword_map_path: Path) -> dict[str, int]:
    subword_df = pd.read_csv(subword_map_path)
    if "index" not in subword_df.columns:
        raise ValueError(f"Missing required column 'index' in {subword_map_path}")
    idx_to_word = subword_df["index"].astype(str).tolist()
    return {token: index for index, token in enumerate(idx_to_word)}


def collect_smiles(input_csvs: list[Path], smiles_columns: list[str]) -> list[str]:
    smiles_set: set[str] = set()
    for csv_path in input_csvs:
        frame = pd.read_csv(csv_path, low_memory=False)
        available_columns = [column for column in smiles_columns if column in frame.columns]
        if not available_columns:
            raise ValueError(
                f"{csv_path} does not contain any of the requested SMILES columns: {smiles_columns}"
            )
        for column in available_columns:
            values = (
                frame[column]
                .dropna()
                .astype(str)
                .str.strip()
            )
            smiles_set.update(value for value in values.tolist() if value)
    return sorted(smiles_set)


def encode_smiles(smiles_list: list[str], bpe: BPE, token_to_index: dict[str, int]) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    encoded = {}
    unknown_tokens = {}
    for smiles in smiles_list:
        tokens = bpe.process_line(smiles).split()
        missing = [token for token in tokens if token not in token_to_index]
        if missing:
            unknown_tokens[smiles] = missing
            continue
        encoded[smiles] = np.asarray([token_to_index[token] for token in tokens], dtype=np.int64)
    return encoded, unknown_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build ESPF token-index features from one or more CSV files containing SMILES.",
    )
    parser.add_argument(
        "--input_csvs",
        nargs="+",
        required=True,
        help="One or more CSV files containing SMILES columns.",
    )
    parser.add_argument(
        "--smiles_columns",
        nargs="+",
        default=["smiles", "drug_row_smiles", "drug_col_smiles"],
        help="Column names to scan for SMILES in each input CSV.",
    )
    parser.add_argument(
        "--output_npy",
        type=str,
        default="data/ESPF/ESPF_smiles_vectors.npy",
        help="Output path for the SMILES -> token-index dictionary.",
    )
    parser.add_argument(
        "--output_smiles_csv",
        type=str,
        default=None,
        help="Optional CSV path for the unique SMILES list used to build the feature file.",
    )
    parser.add_argument(
        "--codes_path",
        type=str,
        default=None,
        help="Optional BPE codes file. Defaults to data/ESPF/info/codes_drug_chembl_1500.txt",
    )
    parser.add_argument(
        "--subword_map_path",
        type=str,
        default=None,
        help="Optional subword mapping CSV. Defaults to data/ESPF/info/subword_units_map_drug_chembl_1500.csv",
    )
    parser.add_argument(
        "--allow_missing_tokens",
        action="store_true",
        help="Skip SMILES with out-of-vocabulary tokens instead of failing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_csvs = [Path(path).expanduser().resolve() for path in args.input_csvs]
    codes_path = resolve_default_path(args.codes_path, "data/ESPF/info/codes_drug_chembl_1500.txt")
    subword_map_path = resolve_default_path(
        args.subword_map_path,
        "data/ESPF/info/subword_units_map_drug_chembl_1500.csv",
    )
    output_npy = Path(args.output_npy).expanduser().resolve()
    output_smiles_csv = Path(args.output_smiles_csv).expanduser().resolve() if args.output_smiles_csv else None

    with codecs.open(codes_path, encoding="utf-8") as handle:
        bpe = BPE(handle, merges=-1, separator="")
    token_to_index = load_vocab(subword_map_path)
    smiles_list = collect_smiles(input_csvs, args.smiles_columns)
    encoded, unknown_tokens = encode_smiles(smiles_list, bpe, token_to_index)

    if unknown_tokens and not args.allow_missing_tokens:
        examples = list(unknown_tokens.items())[:5]
        formatted = "; ".join(
            f"{smiles}: {', '.join(tokens[:8])}" for smiles, tokens in examples
        )
        raise KeyError(
            "Found SMILES with tokens missing from the ESPF vocabulary. "
            f"Examples: {formatted}. Re-run with --allow_missing_tokens to skip them."
        )

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, encoded, allow_pickle=True)

    if output_smiles_csv is not None:
        output_smiles_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"smiles": list(encoded.keys())}).to_csv(output_smiles_csv, index=False)

    skipped = len(unknown_tokens)
    print(f"[done] encoded {len(encoded)} unique SMILES -> {output_npy}")
    if output_smiles_csv is not None:
        print(f"[done] wrote SMILES list -> {output_smiles_csv}")
    if skipped:
        print(f"[warn] skipped {skipped} SMILES due to missing vocabulary tokens")


if __name__ == "__main__":
    main()
