from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_SUPERVISED_EPOCHS = 150
DEFAULT_SUPERVISED_EARLY_STOP_PATIENCE = 15
DEFAULT_DANN_EPOCHS = 10
DEFAULT_DANN_EARLY_STOP_PATIENCE = 5
DATASET_SHORT_NAMES = {
    "drugcombdb_126_67": "db12667",
    "drugcomb_288_146": "dc288146",
}


def resolve_scenario_epochs(scenario: str, user_epochs: int | None) -> int:
    if user_epochs is not None:
        return int(user_epochs)
    return DEFAULT_DANN_EPOCHS if scenario == "dann" else DEFAULT_SUPERVISED_EPOCHS


def resolve_scenario_patience(scenario: str, user_patience: int | None) -> int:
    if user_patience is not None:
        return int(user_patience)
    return DEFAULT_DANN_EARLY_STOP_PATIENCE if scenario == "dann" else DEFAULT_SUPERVISED_EARLY_STOP_PATIENCE


def build_command(dataset_split: str, scenario: str, args) -> list[str]:
    tag = args.tag_prefix
    if tag is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"{timestamp}_{DATASET_SHORT_NAMES[dataset_split]}_{scenario}"
    else:
        tag = f"{args.tag_prefix}_{DATASET_SHORT_NAMES[dataset_split]}_{scenario}"

    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--tag",
        tag,
        "--dataset_split",
        dataset_split,
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--epochs",
        str(resolve_scenario_epochs(scenario, args.epochs)),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--early_stop_patience",
        str(resolve_scenario_patience(scenario, args.early_stop_patience)),
        "--eval_every",
        str(args.eval_every),
    ]
    if args.data_root:
        command.extend(["--data_root", args.data_root])
    if scenario == "dann":
        command.extend(
            [
                "--use_dann",
                "--adv_start_epoch",
                str(args.adv_start_epoch),
                "--adv_warmup_epochs",
                str(args.adv_warmup_epochs),
                "--adv_weight_max",
                str(args.adv_weight_max),
                "--lambda_int",
                str(args.lambda_int),
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the two requested DanSyn scenarios: supervised and DANN.",
    )
    parser.add_argument(
        "--dataset_splits",
        nargs="+",
        default=["drugcombdb_126_67", "drugcomb_288_146"],
        choices=["drugcombdb_126_67", "drugcomb_288_146"],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["supervised", "dann"],
        choices=["supervised", "dann"],
    )
    parser.add_argument("--tag_prefix", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--adv_start_epoch", type=int, default=1)
    parser.add_argument("--adv_warmup_epochs", type=int, default=10)
    parser.add_argument("--adv_weight_max", type=float, default=1.0)
    parser.add_argument("--lambda_int", type=float, default=0.3)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    commands = [
        build_command(dataset_split, scenario, args)
        for dataset_split in args.dataset_splits
        for scenario in args.scenarios
    ]

    for command in commands:
        print(subprocess.list2cmdline(command))

    if args.dry_run:
        return

    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
