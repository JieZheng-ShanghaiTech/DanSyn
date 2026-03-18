from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import RDLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

from core.dataset_splits import (
    get_dataset_split_paths,
    get_dataset_split_paths_from_dir,
    list_dataset_split_choices,
)
from model import CombinedModel, train_supervised, train_with_dann, validate
from core.utils import load_data, prepare_device

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
DEFAULT_SUPERVISED_EPOCHS = 150
DEFAULT_SUPERVISED_EARLY_STOP_PATIENCE = 15
DEFAULT_DANN_EPOCHS = 10
DEFAULT_DANN_EARLY_STOP_PATIENCE = 5


def is_usable_data_root(path: Path) -> bool:
    markers = [
        path / "cell_line_latent_values.npy",
        path / "drugcombDB_126_67.csv",
        path / "drugcomb_288_146.csv",
        path / "ESPF" / "ESPF_smiles_vectors.npy",
    ]
    return any(marker.exists() for marker in markers)


def resolve_data_root(user_data_root: str | None) -> Path:
    candidates = []
    if user_data_root:
        candidates.append(Path(user_data_root))
    candidates.append(ROOT / "data")

    checked = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        checked.append(str(resolved))
        if resolved.exists() and is_usable_data_root(resolved):
            return resolved.resolve()

    raise FileNotFoundError(
        "Could not find a usable data directory. Checked: " + ", ".join(checked)
    )


def sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "", (tag or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned


def create_output_dir(results_root: Path, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_tag = sanitize_tag(tag)
    folder_name = f"{timestamp}_{safe_tag}" if safe_tag else timestamp
    output_dir = results_root / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def infer_feature_dim(feature_dict: dict, feature_name: str) -> int:
    first_key = next(iter(feature_dict))
    first_value = np.asarray(feature_dict[first_key])
    if first_value.ndim != 1:
        raise ValueError(
            f"{feature_name} features must be 1D vectors, but {first_key} has shape {first_value.shape}"
        )
    return int(first_value.shape[0])


def resolve_dataset_csv_paths(
    dataset_split: str | None,
    dataset_split_dir: str | None,
    data_root: Path,
) -> dict:
    if bool(dataset_split) == bool(dataset_split_dir):
        raise ValueError("Provide exactly one of --dataset_split or --dataset_split_dir.")

    if dataset_split_dir:
        dataset_info = get_dataset_split_paths_from_dir(dataset_split_dir)
    else:
        dataset_info = get_dataset_split_paths(str(dataset_split), data_root)

    if dataset_info.get("invalid_reason"):
        raise FileNotFoundError(
            f"dataset split configuration is invalid: "
            f"{dataset_info['invalid_reason']}. Rebuild it with tools/datasets/build_dataset_splits.py."
        )
    if dataset_info["missing"]:
        missing_text = ", ".join(dataset_info["missing"])
        raise FileNotFoundError(
            f"dataset split is missing files: {missing_text}. "
            "Build it with tools/datasets/build_dataset_splits.py."
        )
    return dataset_info


def save_plot(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    if metrics_df.empty:
        return

    plt.figure()
    plt.plot(metrics_df["epoch"], metrics_df["train_mse"], label="Train MSE")
    plt.plot(metrics_df["epoch"], metrics_df["val_mse"], label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MSE over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mse_over_epochs.png")
    plt.close()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate_and_save_predictions(
    model: CombinedModel,
    loader,
    criterion,
    device: torch.device,
    domain_name: str,
    save_pred_path: Path,
) -> dict:
    model.eval()

    all_rows = []
    y_true: list[float] = []
    y_pred: list[float] = []
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        labels = batch["label"].to(device)
        outputs, _ = model(
            espf_a=batch["ESPF_A"].to(device),
            espf_b=batch["ESPF_B"].to(device),
            mask_a=batch["mask_A"].to(device),
            mask_b=batch["mask_B"].to(device),
            omics_latent=batch["omics_latent"].to(device),
            llm_a=batch["llm_A"].to(device),
            llm_b=batch["llm_B"].to(device),
            is_test=True,
        )
        predictions = outputs.squeeze(-1)
        loss = criterion(predictions, labels)

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        labels_np = labels.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()
        for index in range(batch_size):
            all_rows.append(
                {
                    "domain": domain_name,
                    "cell_line_id": str(batch["sample_id"][index]),
                    "cell_line_name": str(batch["cell_line_name"][index]),
                    "drugA_name": str(batch["drugA_name"][index]),
                    "drugB_name": str(batch["drugB_name"][index]),
                    "drugA_smiles": str(batch["smilesA"][index]),
                    "drugB_smiles": str(batch["smilesB"][index]),
                    "true_label": float(labels_np[index]),
                    "pred_label": float(predictions_np[index]),
                }
            )

        y_true.extend(labels_np.tolist())
        y_pred.extend(predictions_np.tolist())

    prediction_df = pd.DataFrame(all_rows)
    prediction_df.to_csv(save_pred_path, index=False, encoding="utf-8-sig")

    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((true_array - pred_array) ** 2)) if len(true_array) else float("nan")
    rmse = float(np.sqrt(mse)) if len(true_array) else float("nan")
    mae = float(np.mean(np.abs(true_array - pred_array))) if len(true_array) else float("nan")

    if len(true_array) >= 2:
        ss_res = float(np.sum((true_array - pred_array) ** 2))
        ss_tot = float(np.sum((true_array - np.mean(true_array)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        pearson = (
            float(np.corrcoef(true_array, pred_array)[0, 1])
            if np.std(true_array) > 0 and np.std(pred_array) > 0
            else float("nan")
        )
    else:
        r2 = float("nan")
        pearson = float("nan")

    return {
        "domain": domain_name,
        "n_samples": int(total_samples),
        "loss": float(total_loss / max(1, total_samples)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "pred_path": str(save_pred_path),
    }


def apply_scenario_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.use_dann:
        if args.epochs is None:
            args.epochs = DEFAULT_DANN_EPOCHS
        if args.early_stop_patience is None:
            args.early_stop_patience = DEFAULT_DANN_EARLY_STOP_PATIENCE
        return args

    if args.epochs is None:
        args.epochs = DEFAULT_SUPERVISED_EPOCHS
    if args.early_stop_patience is None:
        args.early_stop_patience = DEFAULT_SUPERVISED_EARLY_STOP_PATIENCE
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal DanSyn 2.0 training entrypoint.",
    )
    parser.add_argument("--tag", type=str, required=True, help="Run tag used in the output directory name.")
    parser.add_argument(
        "--dataset_split",
        type=str,
        choices=list_dataset_split_choices(),
        default=None,
        help="Choose one of the two fixed dataset presets.",
    )
    parser.add_argument(
        "--dataset_split_dir",
        type=str,
        default=None,
        help="Path to a custom split directory containing the five expected split CSV files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--data_root", type=str, default=None, help="Optional override for the data directory.")
    parser.add_argument("--results_root", type=str, default="results")

    parser.add_argument("--cell_latent_npy", type=str, default=None)
    parser.add_argument("--espf_npy", type=str, default=None)
    parser.add_argument("--llm_npy", type=str, default=None)
    parser.add_argument("--espf_vocab_size", type=int, default=3000)
    parser.add_argument("--espf_max_len", type=int, default=50)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs. Defaults to 150 for supervised runs and 10 for DANN runs.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="Early stopping patience. Defaults to 15 for supervised runs and 5 for DANN runs.",
    )
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--sched_patience", type=int, default=10)
    parser.add_argument("--sched_factor", type=float, default=0.5)

    parser.add_argument("--use_dann", action="store_true", help="Enable the DANN training scenario.")
    parser.add_argument("--adv_start_epoch", type=int, default=1)
    parser.add_argument("--adv_warmup_epochs", type=int, default=10)
    parser.add_argument("--adv_weight_max", type=float, default=1.0)
    parser.add_argument("--lambda_int", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = apply_scenario_defaults(parse_args())

    set_global_seed(args.seed)
    data_root = resolve_data_root(args.data_root)
    dataset_info = resolve_dataset_csv_paths(args.dataset_split, args.dataset_split_dir, data_root)

    args.cell_latent_npy = args.cell_latent_npy or str(data_root / "cell_line_latent_values.npy")
    args.espf_npy = args.espf_npy or str(data_root / "ESPF" / "ESPF_smiles_vectors.npy")
    args.llm_npy = args.llm_npy or str(data_root / "LLM" / "gpt_3_5_turbo" / "llm_smiles_embeddings.npy")

    device = prepare_device(args.device)
    results_root = (ROOT / args.results_root).resolve()
    output_dir = create_output_dir(results_root, args.tag)

    logging.info("Data root: %s", data_root)
    if dataset_info.get("is_custom"):
        logging.info("Dataset split: custom directory %s", dataset_info["split_dir"])
    else:
        logging.info("Dataset split preset: %s", dataset_info["name"])
    logging.info("Training scenario: %s", "dann" if args.use_dann else "supervised")
    logging.info("Results dir: %s", output_dir)

    run_config = vars(args).copy()
    run_config["data_root"] = str(data_root)
    run_config["resolved_dataset_input_csv"] = dataset_info["input_csv"]
    run_config["resolved_dataset_split_dir"] = dataset_info["split_dir"]
    run_config["fixed_drug_encoder"] = "espf"
    run_config["fixed_cell_feature"] = "cell_latent"
    run_config["fixed_fusion"] = "cross_attention"
    run_config["fixed_llm_feature"] = "gpt_3_5_turbo"
    run_config["start_time"] = datetime.now().isoformat()
    save_json(run_config, output_dir / "run_config.json")
    (output_dir / "run_tag.txt").write_text(str(args.tag), encoding="utf-8")

    logging.info("Loading feature dictionaries.")
    cell_latent = np.load(args.cell_latent_npy, allow_pickle=True).item()
    espf_dict = np.load(args.espf_npy, allow_pickle=True).item()
    llm_dict = np.load(args.llm_npy, allow_pickle=True).item()

    cell_dim = infer_feature_dim(cell_latent, "Cell latent")
    llm_dim = infer_feature_dim(llm_dict, "LLM")

    (
        source_train_loader,
        source_val_loader,
        source_test_loader,
        target_unlabeled_loader,
        target_test_loader,
    ) = load_data(
        source_train_csv=dataset_info["paths"]["source_train_csv"],
        source_val_csv=dataset_info["paths"]["source_val_csv"],
        source_test_csv=dataset_info["paths"]["source_test_csv"],
        target_train_unlabeled_csv=dataset_info["paths"]["target_train_unlabeled_csv"],
        target_test_labeled_csv=dataset_info["paths"]["target_test_labeled_csv"],
        espf_dict=espf_dict,
        omics_latent=cell_latent,
        llm_dict=llm_dict,
        batch_size=args.batch_size,
        espf_max_len=args.espf_max_len,
    )

    data_stats = {
        "source_train_batches": len(source_train_loader),
        "source_val_batches": len(source_val_loader),
        "source_test_batches": len(source_test_loader),
        "target_unlabeled_batches": len(target_unlabeled_loader),
        "target_test_batches": len(target_test_loader),
        "cell_feature_dim": cell_dim,
        "llm_feature_dim": llm_dim,
    }
    save_json(data_stats, output_dir / "data_stats.json")

    model = CombinedModel(
        espf_vocab_size=args.espf_vocab_size,
        espf_max_len=args.espf_max_len,
        cell_in_dim=cell_dim,
        llm_dim=llm_dim,
        lambda_adv=args.lambda_int,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.sched_patience,
        factor=args.sched_factor,
    )

    metrics_path = output_dir / "epoch_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "mode",
                "lr",
                "train_loss",
                "train_mse",
                "train_rmse",
                "train_mae",
                "train_r2",
                "val_loss",
                "val_mse",
                "val_rmse",
                "val_mae",
                "val_r2",
                "train_domain_acc",
                "train_align_loss",
                "grl_lambda",
                "adv_weight",
                "train_time_sec",
                "val_time_sec",
                "epoch_time_sec",
            ]
        )

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    best_model_path = output_dir / "best_model.pth"
    last_model_path = output_dir / "last_model.pth"

    logging.info("Start training.")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_start = time.time()

        train_domain_acc = "NA"
        train_align_loss = 0.0
        grl_lambda_logged = 0.0
        adv_weight_logged = 0.0

        if not args.use_dann or (epoch - 1 < args.adv_start_epoch):
            mode = "supervised"
            train_loss, train_mse, train_rmse, train_mae, train_r2 = train_supervised(
                model,
                source_train_loader,
                optimizer,
                criterion,
                device,
            )
        else:
            mode = "dann"
            progress = (epoch - 1 - args.adv_start_epoch) / float(max(1, args.epochs - args.adv_start_epoch))
            lambda_adv = args.lambda_int * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
            model.dann.grl.lambda_ = float(lambda_adv)
            grl_lambda_logged = float(lambda_adv)

            warmup_progress = (epoch - args.adv_start_epoch) / float(max(1, args.adv_warmup_epochs))
            warmup_progress = float(np.clip(warmup_progress, 0.0, 1.0))
            adv_weight = args.adv_weight_max * warmup_progress
            adv_weight_logged = float(adv_weight)

            (
                train_loss,
                train_mse,
                train_rmse,
                train_mae,
                train_r2,
                train_domain_acc,
                train_align_loss,
            ) = train_with_dann(
                model,
                source_train_loader,
                target_unlabeled_loader,
                optimizer,
                criterion,
                device,
                adv_weight=adv_weight,
            )

        train_time = time.time() - train_start

        val_time = 0.0
        if epoch % args.eval_every == 0:
            val_start = time.time()
            val_loss, val_mse, val_rmse, val_mae, val_r2 = validate(
                model,
                source_val_loader,
                criterion,
                device,
            )
            val_time = time.time() - val_start
        else:
            val_loss = float("nan")
            val_mse = float("nan")
            val_rmse = float("nan")
            val_mae = float("nan")
            val_r2 = float("nan")

        if epoch % args.eval_every == 0 and not np.isnan(val_loss):
            scheduler.step(val_loss)

        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.time() - epoch_start

        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch,
                    mode,
                    lr_now,
                    train_loss,
                    train_mse,
                    train_rmse,
                    train_mae,
                    train_r2,
                    val_loss,
                    val_mse,
                    val_rmse,
                    val_mae,
                    val_r2,
                    train_domain_acc,
                    train_align_loss,
                    grl_lambda_logged,
                    adv_weight_logged,
                    train_time,
                    val_time,
                    epoch_time,
                ]
            )

        logging.info(
            "Epoch %s/%s | mode=%s | lr=%.2e | train_loss=%.4f | align_loss=%.4f | val_loss=%.4f",
            epoch,
            args.epochs,
            mode,
            lr_now,
            train_loss,
            train_align_loss,
            val_loss,
        )

        if epoch % args.eval_every == 0 and not np.isnan(val_loss):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    logging.info("Early stopping at epoch %s.", epoch)
                    break

    torch.save(model.state_dict(), last_model_path)

    try:
        metrics_df = pd.read_csv(metrics_path)
        save_plot(metrics_df, output_dir)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to save training plot: %s", exc)

    train_summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_model_path": str(best_model_path),
        "last_model_path": str(last_model_path),
        "end_time": datetime.now().isoformat(),
    }
    save_json(train_summary, output_dir / "train_summary.json")

    logging.info("Loading best model for final evaluation.")
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)

    source_metrics = evaluate_and_save_predictions(
        model,
        source_test_loader,
        criterion,
        device,
        domain_name="source_test",
        save_pred_path=output_dir / "source_test_predictions.csv",
    )
    target_metrics = evaluate_and_save_predictions(
        model,
        target_test_loader,
        criterion,
        device,
        domain_name="target_test",
        save_pred_path=output_dir / "target_test_predictions.csv",
    )

    test_metrics = {
        "source_test": source_metrics,
        "target_test": target_metrics,
    }
    save_json(test_metrics, output_dir / "test_metrics.json")
    pd.DataFrame([source_metrics, target_metrics]).to_csv(
        output_dir / "test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    logging.info("Final evaluation done.")
    logging.info("Source test metrics: %s", source_metrics)
    logging.info("Target test metrics: %s", target_metrics)


if __name__ == "__main__":
    main()
