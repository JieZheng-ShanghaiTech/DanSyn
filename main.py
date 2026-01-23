import os
import re
import json
import time
import csv
import random
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rdkit import RDLogger
from utils import load_data, prepare_device
from model import CombinedModel, train_supervised, train_with_dann, validate

RDLogger.DisableLog("rdApp.*")


# =========================
# Tools
# =========================
def _sanitize_tag(tag: str) -> str:
    tag = (tag or "").strip()
    tag = re.sub(r'[<>:"/\\|?*]+', "", tag).strip()
    tag = re.sub(r"\s+", "_", tag)
    return tag


def create_output_dir(tag: str) -> str:
    """
    results/YYYY-mm-dd_HH-MM-SS_<tag>
    """
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = _sanitize_tag(tag)
    folder = f"{now}_{tag}" if tag else now
    out_dir = os.path.join("results", folder)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_plot(metrics_df: pd.DataFrame, output_dir: str):
    """
    MSE
    """
    if metrics_df.empty:
        return

    plt.figure()
    if "train_mse" in metrics_df.columns:
        plt.plot(metrics_df["epoch"], metrics_df["train_mse"], label="Train MSE")
    if "val_mse" in metrics_df.columns:
        plt.plot(metrics_df["epoch"], metrics_df["val_mse"], label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MSE over Epochs")
    plt.legend()
    plt.tight_layout()

    plt_path = os.path.join(output_dir, "mse_over_epochs.png")
    plt.savefig(plt_path)
    plt.close()


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate_and_save_predictions(
    model: nn.Module,
    loader,
    criterion,
    device,
    domain_name: str,
    save_pred_path: str,
):
    """
    - drug_name/smiles, cell_name/id, true/pred
    """
    model.eval()

    all_rows = []
    y_true, y_pred = [], []
    total_loss = 0.0
    n_total = 0

    for batch in loader:
        labels = batch["label"].to(device)
        omics_latent = batch["omics_latent"].to(device)

        ESPF_A = batch["ESPF_A"].to(device)
        ESPF_B = batch["ESPF_B"].to(device)
        mask_A = batch["mask_A"].to(device)
        mask_B = batch["mask_B"].to(device)

        llm_A = batch["llm_A"].to(device)
        llm_B = batch["llm_B"].to(device)

        outputs, _, _ = model(
            ESPF_A, ESPF_B, mask_A, mask_B,
            omics_latent,
            llm_A=llm_A, llm_B=llm_B,
            labels=None,
            is_test=True
        )
        pred = outputs.squeeze(-1)

        loss = criterion(pred, labels)
        bs = labels.shape[0]
        total_loss += loss.item() * bs
        n_total += bs

        cell_id_list = batch["sample_id"]
        cell_name_list = batch.get("cell_line_name", None)
        drugA_name_list = batch.get("drugA_name", None)
        drugB_name_list = batch.get("drugB_name", None)
        drugA_smiles_list = batch["smilesA"]
        drugB_smiles_list = batch["smilesB"]

        labels_np = labels.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        for i in range(bs):
            all_rows.append({
                "domain": domain_name,
                "cell_line_id": str(cell_id_list[i]),
                "cell_line_name": str(cell_name_list[i]) if cell_name_list is not None else str(cell_id_list[i]),
                "drugA_name": str(drugA_name_list[i]) if drugA_name_list is not None else str(drugA_smiles_list[i]),
                "drugB_name": str(drugB_name_list[i]) if drugB_name_list is not None else str(drugB_smiles_list[i]),
                "drugA_smiles": str(drugA_smiles_list[i]),
                "drugB_smiles": str(drugB_smiles_list[i]),
                "true_label": float(labels_np[i]),
                "pred_label": float(pred_np[i]),
            })

        y_true.extend(labels_np.tolist())
        y_pred.extend(pred_np.tolist())

    pred_df = pd.DataFrame(all_rows)
    pred_df.to_csv(save_pred_path, index=False, encoding="utf-8-sig")

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mse = float(np.mean((y_true - y_pred) ** 2)) if len(y_true) else float("nan")
    rmse = float(np.sqrt(mse)) if len(y_true) else float("nan")
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")

    # r2
    if len(y_true) >= 2:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        r2 = float("nan")

    avg_loss = float(total_loss / max(1, n_total))

    metrics = {
        "domain": domain_name,
        "n_samples": int(n_total),
        "loss": avg_loss,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pred_path": save_pred_path,
    }
    return metrics


# =========================
# main
# =========================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    import argparse
    parser = argparse.ArgumentParser()

    # --- run tag ---
    parser.add_argument("--tag", type=str, required=True, help="run tag")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=int, default=0, help="GPU cuda:<device>")

    # --- data path ---
    parser.add_argument("--source_train_csv", type=str, default="data/dataset/source_train_labeled.csv")
    parser.add_argument("--source_val_csv", type=str, default="data/dataset/source_val_labeled.csv")
    parser.add_argument("--source_test_csv", type=str, default="data/dataset/source_test_labeled.csv")
    parser.add_argument("--target_train_unlabeled_csv", type=str, default="data/dataset/target_train_unlabeled.csv")
    parser.add_argument("--target_test_labeled_csv", type=str, default="data/dataset/target_test_labeled.csv")

    # --- feature ---
    parser.add_argument("--cell_latent_npy", type=str, default="data/cell_line_latent_values.npy")
    parser.add_argument("--espf_npy", type=str, default="data/ESPF/ESPF_smiles_vectors.npy")
    parser.add_argument("--llm_npy", type=str, default="data/LLM/llm_smiles_embeddings.npy")

    # --- hyper parameters ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # early stopping
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=1, help="per epoch")

    # scheduler
    parser.add_argument("--sched_patience", type=int, default=10)
    parser.add_argument("--sched_factor", type=float, default=0.5)

    # --- DANN ---
    parser.add_argument("--use_dann", dest="use_dann", action="store_true",
                        help="DANN activate")
    parser.add_argument("--adv_start_epoch", type=int, default=20, help="DANN start epoch")
    parser.add_argument("--adv_warmup_epochs", type=int, default=10, help="adv_weight warmup epochs")
    parser.add_argument("--adv_weight_max", type=float, default=1.0, help="adv_weight max")
    parser.add_argument("--lambda_int", type=float, default=0.3, help="max GRL lambda")

    args = parser.parse_args()

    # --- seed / device ---
    set_global_seed(args.seed)
    device = prepare_device(args.device)

    # --- output ---
    output_dir = create_output_dir(args.tag)
    logging.info(f"Results dir: {output_dir}")

    # --- config ---
    run_cfg = vars(args).copy()
    run_cfg["output_dir"] = output_dir
    run_cfg["start_time"] = datetime.now().isoformat()
    save_json(run_cfg, os.path.join(output_dir, "run_config.json"))

    # txt
    with open(os.path.join(output_dir, "run_tag.txt"), "w", encoding="utf-8") as f:
        f.write(str(args.tag))

    # =========================
    # feature dict
    # =========================
    logging.info("Loading feature dicts...")
    cell_latent = np.load(args.cell_latent_npy, allow_pickle=True).item()
    ESPF_dict = np.load(args.espf_npy, allow_pickle=True).item()
    LLM_dict = np.load(args.llm_npy, allow_pickle=True).item()

    # =========================
    # dataloader
    # =========================
    loaders = load_data(
        args.source_train_csv,
        args.source_val_csv,
        args.source_test_csv,
        args.target_train_unlabeled_csv,
        args.target_test_labeled_csv,
        ESPF_dict, cell_latent, LLM_dict,
        batch_size=args.batch_size
    )
    (source_train_loader,
     source_val_loader,
     source_test_loader,
     target_train_unlabeled_loader,
     target_test_labeled_loader,
     num_domains) = loaders

    data_stats = {
        "source_train_batches": len(source_train_loader),
        "source_val_batches": len(source_val_loader),
        "source_test_batches": len(source_test_loader),
        "target_unlabeled_batches": len(target_train_unlabeled_loader),
        "target_test_batches": len(target_test_labeled_loader),
        "num_domains_arg": int(num_domains),
    }
    save_json(data_stats, os.path.join(output_dir, "data_stats.json"))

    # =========================
    # model optimizer scheduler
    # =========================
    model = CombinedModel(num_domains).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=args.sched_patience, factor=args.sched_factor)

    # =========================
    # epoch
    # =========================
    metrics_path = os.path.join(output_dir, "epoch_metrics.csv")
    with open(metrics_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "mode",
            "lr",
            "train_loss", "train_mse", "train_rmse", "train_mae", "train_r2",
            "val_loss", "val_mse", "val_rmse", "val_mae", "val_r2",
            "train_domain_acc",
            "grl_lambda",
            "adv_weight",
            "train_time_sec",
            "val_time_sec",
            "epoch_time_sec",
        ])

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    best_model_path = os.path.join(output_dir, "best_model.pth")
    last_model_path = os.path.join(output_dir, "last_model.pth")

    # =========================
    # train
    # =========================
    logging.info("Start training...")
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()

        train_t0 = time.time()

        train_domain_acc = "NA"
        grl_lambda_logged = 0.0
        adv_weight_logged = 0.0

        if (not args.use_dann) or (epoch - 1 < args.adv_start_epoch):
            mode = "supervised"
            train_loss, train_mse, train_rmse, train_mae, train_r2 = train_supervised(
                model, source_train_loader, optimizer, criterion, device
            )
        else:
            mode = "dann"

            # GRL lambda：sigmoid schedule
            p = (epoch - 1 - args.adv_start_epoch) / float(max(1, args.epochs - args.adv_start_epoch))
            lambda_adv = args.lambda_int * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            model.dann.grl.lambda_ = float(lambda_adv)
            grl_lambda_logged = float(lambda_adv)

            # adv_weight warmup：0 -> adv_weight_max
            warmup_progress = (epoch - args.adv_start_epoch) / float(max(1, args.adv_warmup_epochs))
            warmup_progress = float(np.clip(warmup_progress, 0.0, 1.0))
            adv_weight = args.adv_weight_max * warmup_progress
            adv_weight_logged = float(adv_weight)

            train_loss, train_mse, train_rmse, train_mae, train_r2, train_domain_acc = train_with_dann(
                model, model.dann,
                source_train_loader, target_train_unlabeled_loader,
                optimizer, criterion, device,
                adv_weight=adv_weight
            )

        train_time = time.time() - train_t0

        # ---------- val ----------
        val_time = 0.0
        if epoch % args.eval_every == 0:
            val_t0 = time.time()
            val_loss, val_mse, val_rmse, val_mae, val_r2 = validate(
                model, source_val_loader, criterion, device
            )
            val_time = time.time() - val_t0
        else:
            val_loss = float("nan")
            val_mse = float("nan")
            val_rmse = float("nan")
            val_mae = float("nan")
            val_r2 = float("nan")

        # ---------- scheduler ----------
        if epoch % args.eval_every == 0 and not np.isnan(val_loss):
            scheduler.step(val_loss)

        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.time() - epoch_t0

        # ---------- epoch ----------
        with open(metrics_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                mode,
                lr_now,
                train_loss, train_mse, train_rmse, train_mae, train_r2,
                val_loss, val_mse, val_rmse, val_mae, val_r2,
                train_domain_acc,
                grl_lambda_logged,
                adv_weight_logged,
                train_time,
                val_time,
                epoch_time,
            ])

        logging.info(
            f"Epoch {epoch}/{args.epochs} | mode={mode} | lr={lr_now:.2e} "
            f"| train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| time(train/val/epoch)={train_time:.1f}/{val_time:.1f}/{epoch_time:.1f}s"
        )

        # ---------- early stopping + save best ----------
        if epoch % args.eval_every == 0 and not np.isnan(val_loss):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"Saved BEST model -> {best_model_path} (best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch} (patience={args.early_stop_patience}).")
                    break

    # save last
    torch.save(model.state_dict(), last_model_path)
    logging.info(f"Saved LAST model -> {last_model_path}")

    # save metrics
    try:
        df_metrics = pd.read_csv(metrics_path)
        save_plot(df_metrics, output_dir)
    except Exception as e:
        logging.warning(f"Failed to plot metrics: {e}")

    # save train summary
    train_summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_model_path": best_model_path,
        "last_model_path": last_model_path,
        "end_time": datetime.now().isoformat(),
    }
    save_json(train_summary, os.path.join(output_dir, "train_summary.json"))

    # =========================
    # BEST test（source_test / target_test）
    # =========================
    logging.info("Loading BEST model for final evaluation...")
    try:
        state = torch.load(best_model_path, map_location=device)
    except TypeError:
        state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)

    # source
    src_pred_path = os.path.join(output_dir, "source_test_predictions.csv")
    src_metrics = evaluate_and_save_predictions(
        model, source_test_loader, criterion, device,
        domain_name="source_test",
        save_pred_path=src_pred_path
    )

    # target
    tgt_pred_path = os.path.join(output_dir, "target_test_predictions.csv")
    tgt_metrics = evaluate_and_save_predictions(
        model, target_test_labeled_loader, criterion, device,
        domain_name="target_test",
        save_pred_path=tgt_pred_path
    )

    # save
    test_metrics = {
        "source_test": src_metrics,
        "target_test": tgt_metrics,
    }
    save_json(test_metrics, os.path.join(output_dir, "test_metrics.json"))

    test_metrics_csv = os.path.join(output_dir, "test_metrics.csv")
    pd.DataFrame([src_metrics, tgt_metrics]).to_csv(test_metrics_csv, index=False, encoding="utf-8-sig")

    logging.info("Final evaluation done.")
    logging.info(f"Source test metrics: {src_metrics}")
    logging.info(f"Target test metrics: {tgt_metrics}")


if __name__ == "__main__":
    main()

"""
1) Supervised only
python main.py --tag baseline_sup --epochs 150 --batch_size 32 --lr 1e-4 --weight_decay 0 --early_stop_patience 15 --eval_every 1

2) DANN activate
python main.py --tag dann_run --use_dann --adv_start_epoch 5 --adv_warmup_epochs 5 --adv_weight_max 1.0 --lambda_int 0.3 --epochs 10 --batch_size 32 --lr 1e-4 --weight_decay 0 --early_stop_patience 5 --eval_every 1
"""