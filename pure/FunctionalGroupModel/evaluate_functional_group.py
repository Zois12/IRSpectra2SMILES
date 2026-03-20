import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from FunctionalGroupModel.IRFunctionalGroupModel import LightweightIRFunctionalGroupClassifier
from FunctionalGroupModel.functional_group_utils import (
    DEFAULT_FUNCTIONAL_GROUPS,
    IRFunctionalGroupDataset,
    compute_multilabel_metrics,
    load_split_indices,
    save_json,
)


DEFAULT_DATA_PATH = "data/raw_processed_data.pt"
DEFAULT_SPLIT_PATH = "checkpoints/data_split.pt"
DEFAULT_LABEL_CACHE_PATH = "checkpoints/FunctionalGroupModel/functional_group_labels.pt"
DEFAULT_MODEL_PATH = "checkpoints/FunctionalGroupModel/best_functional_group_model.pth"
DEFAULT_CONFIG_PATH = "checkpoints/FunctionalGroupModel/functional_group_config.json"
DEFAULT_OUTPUT_PATH = "checkpoints/FunctionalGroupModel/eval_metrics.json"


class IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        spectrum, labels = self.base_dataset[base_idx]
        return spectrum, labels, base_idx


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def evaluate(model, loader, device, label_names, thresholds):
    model.eval()
    all_probs = []
    all_targets = []
    all_indices = []

    for spectra, labels, base_indices in tqdm(loader, desc="Evaluate", leave=False):
        spectra = spectra.to(device, non_blocking=True).float()
        logits = model(spectra)
        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_targets.append(labels.float())
        all_indices.extend(int(v) for v in base_indices)

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_multilabel_metrics(
        targets=all_targets,
        probs=all_probs,
        label_names=label_names,
        thresholds=thresholds,
    )
    return metrics, all_probs, all_targets, all_indices


def format_label_set(label_names, binary_vec, probs=None):
    out = []
    for i, name in enumerate(label_names):
        if int(binary_vec[i]) == 1:
            if probs is None:
                out.append(name)
            else:
                out.append(f"{name}({float(probs[i]):.2f})")
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate IR functional-group classifier reliability.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split-path", type=str, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--label-cache-path", type=str, default=DEFAULT_LABEL_CACHE_PATH)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    config = load_json(args.config_path)
    checkpoint = torch.load(args.model_path, map_location="cpu")

    label_names = tuple(config.get("label_names", checkpoint.get("label_names", DEFAULT_FUNCTIONAL_GROUPS)))
    thresholds = checkpoint.get("thresholds")
    if thresholds is None:
        thresholds = torch.tensor(config.get("thresholds", [0.5] * len(label_names)), dtype=torch.float32)
    else:
        thresholds = thresholds.float().cpu()
    thresholds = torch.clamp(thresholds * float(args.threshold_scale), min=0.05, max=0.95)

    dataset = IRFunctionalGroupDataset(
        data_path=args.data_path,
        cache_path=args.label_cache_path,
        label_names=label_names,
        canonicalize=True,
    )
    train_idx, val_idx, test_idx = load_split_indices(args.split_path)
    split_map = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    indices = split_map[args.split]
    indexed_dataset = IndexedDataset(dataset, indices)
    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(args.num_workers > 0),
    )

    model = LightweightIRFunctionalGroupClassifier(
        input_points=int(config["input_points"]),
        num_labels=int(config["num_labels"]),
        label_names=label_names,
        stem_channels=int(config.get("stem_channels", 48)),
        hidden_dim=int(config.get("hidden_dim", 192)),
        head_dim=int(config.get("head_dim", 256)),
        dropout=float(config.get("dropout", 0.15)),
        use_coordconv=bool(config.get("use_coordconv", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics, probs, targets, used_indices = evaluate(model, loader, device, label_names, thresholds)
    payload = {
        "split": args.split,
        "thresholds": thresholds.tolist(),
        "metrics": metrics,
    }
    save_json(args.output_path, payload)

    print(f"Split                  : {args.split}")
    print(f"Samples                : {metrics['num_samples']}")
    print(f"Subset accuracy        : {metrics['subset_accuracy']:.4f}")
    print(f"Hamming accuracy       : {metrics['hamming_accuracy']:.4f}")
    print(f"Macro F1               : {metrics['macro_f1']:.4f}")
    print(f"Micro F1               : {metrics['micro_f1']:.4f}")
    print(f"Macro Precision        : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall           : {metrics['macro_recall']:.4f}")
    print(f"Macro AUROC            : {metrics['macro_auroc'] if metrics['macro_auroc'] is not None else 'None'}")
    print(f"Macro Avg Precision    : {metrics['macro_average_precision'] if metrics['macro_average_precision'] is not None else 'None'}")
    print(f"Macro Brier            : {metrics['macro_brier']:.4f}")
    print(f"Macro ECE              : {metrics['macro_ece']:.4f}")
    print(f"Positive Recall        : {metrics['positive_recall']:.4f}")
    print(f"Predicted Positive Rate: {metrics['predicted_positive_rate']:.4f}")
    print("-" * 94)
    print("Per-label metrics")
    for item in sorted(metrics["per_label"], key=lambda x: (x["f1"], x["support"])):
        print(
            f"{item['name']:<18} thr={item['threshold']:.2f} support={item['support']:<5d} "
            f"P={item['precision']:.4f} R={item['recall']:.4f} F1={item['f1']:.4f} "
            f"AUROC={item['auroc'] if item['auroc'] is not None else 'None'} "
            f"AP={item['average_precision'] if item['average_precision'] is not None else 'None'} "
            f"ECE={item['ece']:.4f}"
        )

    pred_binary = (probs >= thresholds.unsqueeze(0)).long()
    print("-" * 94)
    print("Sample predictions")
    mismatch_rows = [i for i in range(len(used_indices)) if not torch.equal(targets[i].long(), pred_binary[i].long())]
    ordered_rows = mismatch_rows + [i for i in range(len(used_indices)) if i not in mismatch_rows]
    shown = 0
    for row_idx in ordered_rows:
        if shown >= args.sample_count:
            break
        base_idx = used_indices[row_idx]
        gt = targets[row_idx].long()
        pred = pred_binary[row_idx].long()
        gt_labels = format_label_set(label_names, gt)
        pred_labels = format_label_set(label_names, pred, probs[row_idx])
        missing = [label_names[i] for i in range(len(label_names)) if int(gt[i]) == 1 and int(pred[i]) == 0]
        extra = [label_names[i] for i in range(len(label_names)) if int(gt[i]) == 0 and int(pred[i]) == 1]
        smiles = dataset.smiles_list[base_idx]
        print(f"[{shown + 1}] SMILES : {smiles}")
        print(f"    GT    : {', '.join(gt_labels) if gt_labels else '(none)'}")
        print(f"    Pred  : {', '.join(pred_labels) if pred_labels else '(none)'}")
        print(f"    Miss  : {', '.join(missing) if missing else '(none)'}")
        print(f"    Extra : {', '.join(extra) if extra else '(none)'}")
        shown += 1


if __name__ == "__main__":
    main()

