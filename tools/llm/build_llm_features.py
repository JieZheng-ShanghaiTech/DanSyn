from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.llm.llm_profile_utils import build_mechanism_text, load_done_smiles, load_reference_table

DEFAULT_TEXT_MODEL = "gpt-3.5-turbo"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def parse_json_object(raw_text: str | None) -> dict | None:
    if raw_text is None:
        return None
    content = str(raw_text).strip()
    if not content:
        return None
    return json.loads(content)


def build_client(api_key: str | None, base_url: str | None, extra_headers: dict | None) -> OpenAI:
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if extra_headers:
        kwargs["default_headers"] = extra_headers
    return OpenAI(**kwargs)


def call_with_retries(fn, max_retries: int, *args, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            wait_sec = 2 ** attempt
            logging.warning("Call failed: %s; retrying in %ss", exc, wait_sec)
            time.sleep(wait_sec)
    raise RuntimeError(f"Failed after {max_retries} retries: {fn.__name__}")


def load_lookup_table(path: Path, limit: int | None = None) -> pd.DataFrame:
    lookup = pd.read_csv(path, low_memory=False)
    required_cols = {"drug_name", "smiles"}
    missing = required_cols - set(lookup.columns)
    if missing:
        raise ValueError(f"Lookup csv missing required columns: {sorted(missing)}")

    lookup = lookup.dropna(subset=["drug_name", "smiles"]).copy()
    lookup["drug_name"] = lookup["drug_name"].astype(str).str.strip()
    lookup["smiles"] = lookup["smiles"].astype(str).str.strip()
    lookup = lookup[(lookup["drug_name"] != "") & (lookup["smiles"] != "")]
    lookup = lookup.drop_duplicates(subset=["smiles"], keep="first").reset_index(drop=True)

    if limit is not None:
        lookup = lookup.head(int(limit)).copy()
    return lookup


def build_lookup_from_input_csvs(input_csvs: list[Path], limit: int | None = None) -> pd.DataFrame:
    rows_by_smiles: dict[str, dict] = {}

    for csv_path in input_csvs:
        frame = pd.read_csv(csv_path, low_memory=False)

        paired_columns = [
            ("drug_row", "drug_row_smiles"),
            ("drug_col", "drug_col_smiles"),
            ("drug_name", "smiles"),
            ("name", "smiles"),
        ]

        found_any_pair = False
        for name_col, smiles_col in paired_columns:
            if smiles_col not in frame.columns:
                continue
            found_any_pair = True
            names = frame[name_col] if name_col in frame.columns else frame[smiles_col]
            subset = pd.DataFrame(
                {
                    "drug_name": names.astype(str).str.strip(),
                    "smiles": frame[smiles_col].astype(str).str.strip(),
                }
            )
            subset = subset[(subset["drug_name"] != "") & (subset["smiles"] != "")]
            for _, row in subset.iterrows():
                smiles = row["smiles"]
                rows_by_smiles.setdefault(smiles, {"drug_name": row["drug_name"], "smiles": smiles})

        if not found_any_pair:
            raise ValueError(
                f"{csv_path} does not contain a supported SMILES schema. "
                "Expected one of: (drug_row, drug_row_smiles), (drug_col, drug_col_smiles), "
                "(drug_name, smiles), or (name, smiles)."
            )

    lookup = pd.DataFrame(rows_by_smiles.values()).sort_values("smiles").reset_index(drop=True)
    if limit is not None:
        lookup = lookup.head(int(limit)).copy()
    return lookup


def generate_description(client: OpenAI, text_model: str, drug_name: str, smiles: str, reference_df) -> str:
    return build_mechanism_text(
        client=client,
        text_model=text_model,
        drug_name=drug_name,
        smiles=smiles,
        reference_df=reference_df,
    )


def embed_text(client: OpenAI, embed_model: str, text: str) -> list[float]:
    content = (text or "").replace("\n", " ").strip()
    if not content:
        raise ValueError("Empty description text; cannot embed.")
    response = client.embeddings.create(model=embed_model, input=content)
    return response.data[0].embedding


def rebuild_npy_from_csv(csv_path: Path, npy_path: Path) -> tuple[int, int]:
    frame = pd.read_csv(csv_path, low_memory=False)
    llm_dict = {}
    embedding_dim = 0

    for _, row in frame.iterrows():
        smiles = str(row["smiles"]).strip()
        vector = np.asarray(json.loads(row["Embedding"]), dtype=np.float32)
        llm_dict[smiles] = vector
        embedding_dim = int(vector.shape[0])

    np.save(npy_path, llm_dict, allow_pickle=True)
    return len(llm_dict), embedding_dim


def write_metadata(
    metadata_path: Path,
    *,
    text_model: str,
    embed_model: str,
    lookup_source: str,
    reference_csv: Path | None,
    csv_path: Path,
    npy_path: Path,
    num_unique_drugs: int,
    num_vectors_saved: int,
    embedding_dim: int,
    request_sleep_sec: float,
    max_retries: int,
    base_url: str,
) -> None:
    metadata = {
        "text_model": text_model,
        "embedding_model": embed_model,
        "lookup_source": lookup_source,
        "reference_csv": str(reference_csv) if reference_csv is not None else "",
        "csv_path": str(csv_path),
        "npy_path": str(npy_path),
        "num_unique_drugs": int(num_unique_drugs),
        "num_vectors_saved": int(num_vectors_saved),
        "embedding_dim": int(embedding_dim),
        "request_sleep_sec": float(request_sleep_sec),
        "max_retries": int(max_retries),
        "base_url": base_url,
        "prompt_source": "tools/llm/llm_profile_utils.py",
        "generated_at": datetime.now().isoformat(),
        "complete": bool(num_vectors_saved == num_unique_drugs),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def maybe_remove_output_files(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate LLM drug mechanism embeddings with the DanSyn 2.0 prompt design.",
    )
    parser.add_argument(
        "--lookup_csv",
        type=str,
        default="data/unique_drugs_with_smiles.csv",
        help="CSV with `drug_name` and `smiles` columns. Ignored if --input_csvs is provided.",
    )
    parser.add_argument(
        "--input_csvs",
        nargs="+",
        default=None,
        help="Optional synergy CSV files to automatically build a unique drug lookup table.",
    )
    parser.add_argument(
        "--save_lookup_csv",
        type=str,
        default=None,
        help="Optional path to save the resolved unique drug lookup table.",
    )
    parser.add_argument("--reference_csv", type=str, default="")
    parser.add_argument("--output_csv", type=str, default="data/LLM/gpt_3_5_turbo/drug_desc_embedding.csv")
    parser.add_argument("--output_npy", type=str, default="data/LLM/gpt_3_5_turbo/llm_smiles_embeddings.npy")
    parser.add_argument("--metadata_json", type=str, default="data/LLM/gpt_3_5_turbo/llm_feature_metadata.json")
    parser.add_argument("--text_model", type=str, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--embed_model", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument(
        "--extra_headers_json",
        type=str,
        default="",
        help='Optional JSON object, e.g. {"Authorization":"Bearer ..."}',
    )
    parser.add_argument("--request_sleep_sec", type=float, default=1.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    lookup_csv = (ROOT / args.lookup_csv).resolve()
    reference_csv = (ROOT / args.reference_csv).resolve() if args.reference_csv else None
    output_csv = (ROOT / args.output_csv).resolve()
    output_npy = (ROOT / args.output_npy).resolve()
    metadata_json = (ROOT / args.metadata_json).resolve()
    save_lookup_csv = (ROOT / args.save_lookup_csv).resolve() if args.save_lookup_csv else None
    extra_headers = parse_json_object(args.extra_headers_json)

    if args.input_csvs:
        input_csvs = [(ROOT / path).resolve() for path in args.input_csvs]
        unique_drugs = build_lookup_from_input_csvs(input_csvs, limit=args.limit)
        lookup_source = ", ".join(str(path) for path in input_csvs)
    else:
        unique_drugs = load_lookup_table(lookup_csv, limit=args.limit)
        lookup_source = str(lookup_csv)

    if save_lookup_csv is not None:
        save_lookup_csv.parent.mkdir(parents=True, exist_ok=True)
        unique_drugs.to_csv(save_lookup_csv, index=False)

    reference_df = load_reference_table(str(reference_csv)) if reference_csv is not None else None

    logging.info("Resolved %d unique drugs.", len(unique_drugs))
    logging.info("Output csv: %s", output_csv)
    logging.info("Output npy: %s", output_npy)
    logging.info("Metadata: %s", metadata_json)
    logging.info("Prompt source: %s", ROOT / "tools" / "llm" / "llm_profile_utils.py")

    if args.dry_run:
        print(unique_drugs.head(min(5, len(unique_drugs))).to_string(index=False))
        return

    if not args.api_key:
        raise ValueError(
            "No API key provided. Pass --api_key with your own key, or leave --base_url empty to use the default provider."
        )

    client = build_client(args.api_key, args.base_url, extra_headers)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        maybe_remove_output_files(output_csv, output_npy, metadata_json)

    done_smiles = load_done_smiles(str(output_csv)) if output_csv.exists() else set()
    output_exists = output_csv.exists()

    for index, row in unique_drugs.iterrows():
        drug_name = str(row["drug_name"])
        smiles = str(row["smiles"])
        if smiles in done_smiles:
            continue

        logging.info("[%d/%d] %s", index + 1, len(unique_drugs), drug_name)
        description = call_with_retries(
            generate_description,
            args.max_retries,
            client,
            args.text_model,
            drug_name,
            smiles,
            reference_df,
        )
        vector = call_with_retries(
            embed_text,
            args.max_retries,
            client,
            args.embed_model,
            description,
        )

        row_df = pd.DataFrame(
            [
                {
                    "drug_name": drug_name,
                    "smiles": smiles,
                    "Description": description,
                    "Embedding": json.dumps(vector),
                    "EmbeddingDim": len(vector),
                    "TextModel": args.text_model,
                    "EmbeddingModel": args.embed_model,
                }
            ]
        )
        row_df.to_csv(output_csv, mode="a", header=not output_exists, index=False)
        output_exists = True
        done_smiles.add(smiles)
        time.sleep(args.request_sleep_sec)

    num_vectors_saved, embedding_dim = rebuild_npy_from_csv(output_csv, output_npy)
    write_metadata(
        metadata_json,
        text_model=args.text_model,
        embed_model=args.embed_model,
        lookup_source=lookup_source,
        reference_csv=reference_csv,
        csv_path=output_csv,
        npy_path=output_npy,
        num_unique_drugs=len(unique_drugs),
        num_vectors_saved=num_vectors_saved,
        embedding_dim=embedding_dim,
        request_sleep_sec=args.request_sleep_sec,
        max_retries=args.max_retries,
        base_url=args.base_url,
    )
    logging.info("Saved %d embeddings -> %s", num_vectors_saved, output_npy)


if __name__ == "__main__":
    main()
