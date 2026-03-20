import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from FunctionalGroupModel.IRFunctionalGroupModel import (
    DEFAULT_FUNCTIONAL_GROUPS,
    LightweightIRFunctionalGroupClassifier,
    functional_group_loss,
)
from FunctionalGroupModel.functional_group_utils import (
    IRFunctionalGroupDataset,
    compute_multilabel_metrics,
    compute_pos_weight,
    load_split_indices,
    save_json,
    tune_thresholds,
)


DEFAULT_DATA_PATH = "data/raw_processed_data.pt"
DEFAULT_SPLIT_PATH = "checkpoints/data_split.pt"
DEFAULT_LABEL_CACHE_PATH = "checkpoints/FunctionalGroupModel/functional_group_labels.pt"
DEFAULT_MODEL_PATH = "checkpoints/FunctionalGroupModel/best_functional_group_model.pth"
DEFAULT_CONFIG_PATH = "checkpoints/FunctionalGroupModel/functional_group_config.json"
DEFAULT_TEST_METRICS_PATH = "checkpoints/FunctionalGroupModel/test_metrics.json"


def build_loader(dataset, indices, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
    )


@torch.no_grad()
def run_eval(
    model,
    loader,
    device,
    pos_weight: torch.Tensor,
    label_smoothing: float,
    label_names,
    thresholds: torch.Tensor = None,
    tune_on_split: bool = False,
) -> Tuple[float, Dict[str, object], torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_probs = []
    all_targets = []

    for spectra, labels in loader:
        spectra = spectra.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).float()
        logits = model(spectra)
        loss = functional_group_loss(
            logits,
            labels,
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
        )
        probs = torch.sigmoid(logits)

        total_loss += float(loss.item())
        total_batches += 1
        all_probs.append(probs.detach().cpu())
        all_targets.append(labels.detach().cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if tune_on_split:
        thresholds, _best_per_label = tune_thresholds(
            targets=all_targets,
            probs=all_probs,
            label_names=label_names,
        )

    metrics = compute_multilabel_metrics(
        targets=all_targets,
        probs=all_probs,
        label_names=label_names,
        thresholds=thresholds,
    )
    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss, metrics, thresholds if thresholds is not None else torch.full((len(label_names),), 0.5)


def main():
    parser = argparse.ArgumentParser(description="Train lightweight IR functional-group classifier.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split-path", type=str, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--label-cache-path", type=str, default=DEFAULT_LABEL_CACHE_PATH)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--test-metrics-path", type=str, default=DEFAULT_TEST_METRICS_PATH)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--stem-channels", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--use-coordconv", action="store_true")
    parser.add_argument("--no-coordconv", dest="use_coordconv", action="store_false")
    parser.set_defaults(use_coordconv=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    dataset = IRFunctionalGroupDataset(
        data_path=args.data_path,
        cache_path=args.label_cache_path,
        label_names=DEFAULT_FUNCTIONAL_GROUPS,
        canonicalize=True,
    )
    train_idx, val_idx, test_idx = load_split_indices(args.split_path)

    train_loader = build_loader(dataset, train_idx, args.batch_size, True, args.num_workers, pin_memory)
    val_loader = build_loader(dataset, val_idx, args.batch_size, False, args.num_workers, pin_memory)
    test_loader = build_loader(dataset, test_idx, args.batch_size, False, args.num_workers, pin_memory)

    model = LightweightIRFunctionalGroupClassifier(
        input_points=dataset.input_points,
        num_labels=dataset.num_labels,
        label_names=dataset.label_names,
        stem_channels=args.stem_channels,
        hidden_dim=args.hidden_dim,
        head_dim=args.head_dim,
        dropout=args.dropout,
        use_coordconv=args.use_coordconv,
    ).to(device)

    train_labels = dataset.labels[train_idx].float()
    pos_weight = compute_pos_weight(train_labels).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    model_dir = os.path.dirname(args.model_path)
    config_dir = os.path.dirname(args.config_path)
    metrics_dir = os.path.dirname(args.test_metrics_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    tqdm.write(f"Dataset size: {len(dataset)}")
    tqdm.write(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    tqdm.write(f"Invalid SMILES in label build: {int(dataset.invalid_smiles.sum().item())}")
    tqdm.write(f"Functional groups: {', '.join(dataset.label_names)}")

    best_macro_f1 = -1.0
    best_epoch = 0
    best_thresholds = torch.full((dataset.num_labels,), 0.5)
    early_stop_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for spectra, labels in train_bar:
            spectra = spectra.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(spectra)
                loss = functional_group_loss(
                    logits,
                    labels,
                    pos_weight=pos_weight,
                    label_smoothing=args.label_smoothing,
                )

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item())
            train_batches += 1
            train_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss, val_metrics, tuned_thresholds = run_eval(
            model=model,
            loader=val_loader,
            device=device,
            pos_weight=pos_weight,
            label_smoothing=args.label_smoothing,
            label_names=dataset.label_names,
            thresholds=best_thresholds,
            tune_on_split=True,
        )
        scheduler.step(val_metrics["macro_f1"])

        epoch_summary = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_micro_f1": float(val_metrics["micro_f1"]),
            "val_subset_accuracy": float(val_metrics["subset_accuracy"]),
            "val_macro_auroc": val_metrics.get("macro_auroc"),
            "val_macro_average_precision": val_metrics.get("macro_average_precision"),
            "val_macro_ece": float(val_metrics["macro_ece"]),
        }
        history.append(epoch_summary)

        tqdm.write(
            "Epoch {}/{} | lr={:.2e} train_loss={:.4f} | val_loss={:.4f} val_macro_f1={:.4f} val_micro_f1={:.4f} val_subset_acc={:.4f} val_macro_ece={:.4f}".format(
                epoch,
                args.epochs,
                optimizer.param_groups[0]["lr"],
                train_loss,
                val_loss,
                val_metrics["macro_f1"],
                val_metrics["micro_f1"],
                val_metrics["subset_accuracy"],
                val_metrics["macro_ece"],
            )
        )

        if val_metrics["macro_f1"] > best_macro_f1 + args.early_stop_min_delta:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_thresholds = tuned_thresholds.clone().cpu()
            early_stop_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "thresholds": best_thresholds,
                    "label_names": list(dataset.label_names),
                    "config": {
                        "input_points": dataset.input_points,
                        "num_labels": dataset.num_labels,
                        "label_names": list(dataset.label_names),
                        "stem_channels": args.stem_channels,
                        "hidden_dim": args.hidden_dim,
                        "head_dim": args.head_dim,
                        "dropout": args.dropout,
                        "use_coordconv": args.use_coordconv,
                    },
                    "best_epoch": best_epoch,
                    "best_val_metrics": val_metrics,
                },
                args.model_path,
            )
            save_json(
                args.config_path,
                {
                    "data_path": args.data_path,
                    "split_path": args.split_path,
                    "label_cache_path": args.label_cache_path,
                    "model_path": args.model_path,
                    "input_points": dataset.input_points,
                    "num_labels": dataset.num_labels,
                    "label_names": list(dataset.label_names),
                    "thresholds": best_thresholds.tolist(),
                    "stem_channels": args.stem_channels,
                    "hidden_dim": args.hidden_dim,
                    "head_dim": args.head_dim,
                    "dropout": args.dropout,
                    "use_coordconv": args.use_coordconv,
                    "best_epoch": best_epoch,
                    "best_val_metrics": val_metrics,
                    "history": history,
                },
            )
            tqdm.write(f"Saved best model to {args.model_path}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.early_stop_patience:
            tqdm.write(
                f"Early stopping at epoch {epoch} (best macro_f1={best_macro_f1:.4f}, patience={args.early_stop_patience})"
            )
            break

    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    best_thresholds = checkpoint.get("thresholds", best_thresholds).float()

    test_loss, test_metrics, _ = run_eval(
        model=model,
        loader=test_loader,
        device=device,
        pos_weight=pos_weight,
        label_smoothing=args.label_smoothing,
        label_names=dataset.label_names,
        thresholds=best_thresholds,
        tune_on_split=False,
    )
    test_payload = {
        "best_epoch": checkpoint.get("best_epoch", best_epoch),
        "test_loss": test_loss,
        "test_metrics": test_metrics,
    }
    save_json(args.test_metrics_path, test_payload)

    tqdm.write(
        "Test | loss={:.4f} macro_f1={:.4f} micro_f1={:.4f} subset_acc={:.4f} macro_auroc={} macro_ap={} macro_ece={:.4f}".format(
            test_loss,
            test_metrics["macro_f1"],
            test_metrics["micro_f1"],
            test_metrics["subset_accuracy"],
            "{:.4f}".format(test_metrics["macro_auroc"]) if test_metrics["macro_auroc"] is not None else "None",
            "{:.4f}".format(test_metrics["macro_average_precision"]) if test_metrics["macro_average_precision"] is not None else "None",
            test_metrics["macro_ece"],
        )
    )


if __name__ == "__main__":
    main()
