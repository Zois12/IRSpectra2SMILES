import argparse
import csv
from datetime import datetime
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # pure
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import optuna
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "Optuna is required for this script. Install it first with: pip install optuna\n"
        f"Import error: {exc}"
    )

from TransformerModel import train_transformer as train_mod
from util.tokenizer import build_vocab


def ensure_vocab_exists():
    if Path(train_mod.VOCAB_PATH).exists():
        return
    print(f"Vocab not found at {train_mod.VOCAB_PATH}, building once before search...")
    build_vocab(train_mod.DATA_PATH, canonicalize_smiles=train_mod.CANONICALIZE_SMILES)


def make_trial_overrides(args, trial: "optuna.trial.Trial", out_dir: Path):
    trial_dir = out_dir / f"trial_{trial.number:04d}"
    model_path = trial_dir / "best_transformer_model.pth"
    best_loss_model_path = trial_dir / "best_transformer_loss_model.pth"
    config_path = trial_dir / "transformer_config.json"

    return {
        "TRAIN_EPOCH": args.search_epochs,
        "BATCH_SIZE": args.batch_size,
        "LEARNING_RATE": trial.suggest_float("learning_rate", args.lr_min, args.lr_max, log=True),
        "WEIGHT_DECAY": trial.suggest_float("weight_decay", args.wd_min, args.wd_max, log=True),
        "DROPOUT": trial.suggest_float("dropout", args.dropout_min, args.dropout_max),
        "LABEL_SMOOTHING": trial.suggest_float("label_smoothing", args.label_smoothing_min, args.label_smoothing_max),
        "ENCODER_PATCH_SIZE": trial.suggest_categorical("encoder_patch_size", args.patch_sizes),
        "TANIMOTO_LOSS_WEIGHT": trial.suggest_float("tanimoto_loss_weight", args.tanimoto_min, args.tanimoto_max),
        "MASKED_PATCH_LOSS_WEIGHT": trial.suggest_float("masked_patch_loss_weight", args.masked_patch_loss_min, args.masked_patch_loss_max),
        "TEACHER_FORCING_END": trial.suggest_float("teacher_forcing_end", args.tf_end_min, args.tf_end_max),
        "TEACHER_FORCING_WARMUP_EPOCHS": trial.suggest_int("teacher_forcing_warmup_epochs", args.tf_warmup_min, args.tf_warmup_max),
        "TEACHER_FORCING_DECAY_EPOCHS": trial.suggest_int("teacher_forcing_decay_epochs", args.tf_decay_min, args.tf_decay_max),
        "VAL_SEQ_EM_MAX_SAMPLES": args.val_seq_em_samples,
        "VAL_SEQ_EM_EVERY": args.val_seq_em_every,
        "EARLY_STOP_PATIENCE": args.early_stop_patience,
        "USE_WANDB": False,
        "REBUILD_VOCAB": False,
        "SAVE_TRAINING_ARTIFACTS": args.save_trial_artifacts,
        "MODEL_PATH": str(model_path),
        "BEST_LOSS_MODEL_PATH": str(best_loss_model_path),
        "CONFIG_PATH": str(config_path),
    }


def log_trial_start(out_dir: Path, trial: "optuna.trial.Trial", overrides: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trial_number": trial.number,
        "params": dict(trial.params),
        "overrides": {
            "TRAIN_EPOCH": overrides.get("TRAIN_EPOCH"),
            "BATCH_SIZE": overrides.get("BATCH_SIZE"),
            "VAL_SEQ_EM_MAX_SAMPLES": overrides.get("VAL_SEQ_EM_MAX_SAMPLES"),
            "VAL_SEQ_EM_EVERY": overrides.get("VAL_SEQ_EM_EVERY"),
            "EARLY_STOP_PATIENCE": overrides.get("EARLY_STOP_PATIENCE"),
            "SAVE_TRAINING_ARTIFACTS": overrides.get("SAVE_TRAINING_ARTIFACTS"),
            "MODEL_PATH": overrides.get("MODEL_PATH"),
            "BEST_LOSS_MODEL_PATH": overrides.get("BEST_LOSS_MODEL_PATH"),
            "CONFIG_PATH": overrides.get("CONFIG_PATH"),
        },
    }

    print("-" * 70)
    print(f"Starting trial {trial.number}")
    for key, value in payload["params"].items():
        print(f"  {key}: {value}")
    print(f"  batch_size: {payload['overrides']['BATCH_SIZE']}")
    print(f"  search_epochs: {payload['overrides']['TRAIN_EPOCH']}")
    print(f"  val_seq_em_samples: {payload['overrides']['VAL_SEQ_EM_MAX_SAMPLES']}")
    print(f"  trial_model_path: {payload['overrides']['MODEL_PATH']}")

    with open(out_dir / "trial_start_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_study_outputs(study: "optuna.study.Study", out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    best_payload = {
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
    }
    with open(out_dir / "best_trial.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "trial_number",
        "state",
        "value",
        "learning_rate",
        "weight_decay",
        "dropout",
        "label_smoothing",
        "encoder_patch_size",
        "tanimoto_loss_weight",
        "masked_patch_loss_weight",
        "teacher_forcing_end",
        "teacher_forcing_warmup_epochs",
        "teacher_forcing_decay_epochs",
        "best_seq_em",
        "best_loss",
        "epochs_ran",
        "val_loss",
        "val_seq_em",
    ]
    with open(out_dir / "trial_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            row = {
                "trial_number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "best_seq_em": trial.user_attrs.get("best_seq_em"),
                "best_loss": trial.user_attrs.get("best_loss"),
                "epochs_ran": trial.user_attrs.get("epochs_ran"),
                "val_loss": trial.user_attrs.get("val_loss"),
                "val_seq_em": trial.user_attrs.get("val_seq_em"),
            }
            for key in (
                "learning_rate",
                "weight_decay",
                "dropout",
                "label_smoothing",
                "encoder_patch_size",
                "tanimoto_loss_weight",
                "masked_patch_loss_weight",
                "teacher_forcing_end",
                "teacher_forcing_warmup_epochs",
                "teacher_forcing_decay_epochs",
            ):
                row[key] = trial.params.get(key)
            writer.writerow(row)


def objective_factory(args, out_dir: Path):
    def objective(trial: "optuna.trial.Trial"):
        overrides = make_trial_overrides(args, trial, out_dir)
        log_trial_start(out_dir, trial, overrides)

        def epoch_callback(epoch: int, metrics: dict):
            report_value = metrics["val_seq_em"] if args.objective == "best_seq_em" else -metrics["val_loss"]
            trial.report(report_value, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        summary = train_mod.train_model(overrides=overrides, epoch_callback=epoch_callback)
        trial.set_user_attr("best_seq_em", summary.get("best_seq_em"))
        trial.set_user_attr("best_loss", summary.get("best_loss"))
        trial.set_user_attr("epochs_ran", summary.get("epochs_ran"))
        trial.set_user_attr("val_loss", summary.get("val_loss"))
        trial.set_user_attr("val_seq_em", summary.get("val_seq_em"))

        if args.objective == "best_seq_em":
            return float(summary["best_seq_em"])
        return float(-summary["best_loss"])

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for Transformer IR->SMILES.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--search-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-seq-em-samples", type=int, default=1000)
    parser.add_argument("--val-seq-em-every", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--objective", type=str, choices=["best_seq_em", "best_loss"], default="best_seq_em")
    parser.add_argument("--study-name", type=str, default="transformer_hparam_search")
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///checkpoints/TransformerModel/optuna/transformer_hparam_search.db",
    )
    parser.add_argument("--save-trial-artifacts", action="store_true")
    parser.add_argument("--output-dir", type=str, default="checkpoints/TransformerModel/optuna")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr-min", type=float, default=5e-5)
    parser.add_argument("--lr-max", type=float, default=3e-4)
    parser.add_argument("--wd-min", type=float, default=1e-4)
    parser.add_argument("--wd-max", type=float, default=2e-2)
    parser.add_argument("--dropout-min", type=float, default=0.05)
    parser.add_argument("--dropout-max", type=float, default=0.25)
    parser.add_argument("--label-smoothing-min", type=float, default=0.0)
    parser.add_argument("--label-smoothing-max", type=float, default=0.15)
    parser.add_argument("--tanimoto-min", type=float, default=0.0)
    parser.add_argument("--tanimoto-max", type=float, default=0.1)
    parser.add_argument("--masked-patch-loss-min", type=float, default=0.0)
    parser.add_argument("--masked-patch-loss-max", type=float, default=0.15)
    parser.add_argument("--tf-end-min", type=float, default=0.5)
    parser.add_argument("--tf-end-max", type=float, default=0.9)
    parser.add_argument("--tf-warmup-min", type=int, default=2)
    parser.add_argument("--tf-warmup-max", type=int, default=6)
    parser.add_argument("--tf-decay-min", type=int, default=10)
    parser.add_argument("--tf-decay-max", type=int, default=25)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[2, 4, 5, 8])
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_vocab_exists()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2)
    direction = "maximize" if args.objective == "best_seq_em" else "maximize"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
    )

    print("=" * 70)
    print("Starting Optuna search")
    print(f"Study name            : {args.study_name}")
    print(f"Storage               : {args.storage}")
    print(f"Trials                : {args.n_trials}")
    print(f"Search epochs         : {args.search_epochs}")
    print(f"Batch size            : {args.batch_size}")
    print(f"Objective             : {args.objective}")
    print(f"Val seq-em samples    : {args.val_seq_em_samples}")
    print(f"Save trial artifacts  : {args.save_trial_artifacts}")
    print("=" * 70)

    study.optimize(objective_factory(args, out_dir), n_trials=args.n_trials)
    save_study_outputs(study, out_dir)

    print("=" * 70)
    print("Optuna search finished")
    print(f"Best trial number     : {study.best_trial.number}")
    print(f"Best objective value  : {study.best_value:.6f}")
    print("Best params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"Saved summary         : {out_dir / 'best_trial.json'}")
    print(f"Saved trial table     : {out_dir / 'trial_results.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
