import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from FunctionalGroupModel.IRFunctionalGroupModel import DEFAULT_FUNCTIONAL_GROUPS

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
except Exception as exc:  # pragma: no cover - runtime dependency
    Chem = None
    RDLogger = None
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:
    average_precision_score = None
    roc_auc_score = None


FUNCTIONAL_GROUP_VERSION = 1


def require_rdkit():
    if Chem is None:
        raise ImportError(
            "RDKit is required for functional-group label construction. "
            f"Import error: {_RDKIT_IMPORT_ERROR}"
        )


def _compile_smarts(pattern: str):
    require_rdkit()
    mol = Chem.MolFromSmarts(pattern)
    if mol is None:
        raise ValueError(f"Invalid SMARTS pattern: {pattern}")
    return mol


SMARTS_PATTERNS: Dict[str, str] = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "phenol": "c[OX2H]",
    "alcohol_raw": "[OX2H][CX4;!$(C=O)]",
    "ether_raw": "[OD2]([#6])[#6]",
    "amine_raw": "[NX3;H2,H1,H0;!$([N+](=O)[O-])]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "nitrile": "[CX2]#N",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "halide": "[F,Cl,Br,I]",
}


COMPILED_SMARTS = {name: _compile_smarts(pattern) for name, pattern in SMARTS_PATTERNS.items()} if Chem else {}


def _match_atom_sets(mol, pattern_name: str) -> List[set]:
    patt = COMPILED_SMARTS[pattern_name]
    return [set(match) for match in mol.GetSubstructMatches(patt)]


def _atom_indices_from_matches(match_sets: Iterable[set], atomic_nums: Optional[Sequence[int]] = None, mol=None) -> set:
    out = set()
    if atomic_nums is None:
        for match in match_sets:
            out.update(match)
        return out

    atomic_nums = set(int(v) for v in atomic_nums)
    for match in match_sets:
        for idx in match:
            if mol.GetAtomWithIdx(int(idx)).GetAtomicNum() in atomic_nums:
                out.add(int(idx))
    return out


def canonicalize_smiles(smiles: str) -> str:
    require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def extract_functional_group_labels(
    smiles: str,
    label_names: Sequence[str] = DEFAULT_FUNCTIONAL_GROUPS,
) -> List[int]:
    require_rdkit()
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return [0 for _ in label_names]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0 for _ in label_names]

    carbox_matches = _match_atom_sets(mol, "carboxylic_acid")
    ester_matches = _match_atom_sets(mol, "ester")
    amide_matches = _match_atom_sets(mol, "amide")
    nitro_matches = _match_atom_sets(mol, "nitro")
    phenol_matches = _match_atom_sets(mol, "phenol")
    alcohol_matches = _match_atom_sets(mol, "alcohol_raw")
    ether_matches = _match_atom_sets(mol, "ether_raw")
    amine_matches = _match_atom_sets(mol, "amine_raw")
    aldehyde_matches = _match_atom_sets(mol, "aldehyde")
    ketone_matches = _match_atom_sets(mol, "ketone")
    nitrile_matches = _match_atom_sets(mol, "nitrile")
    alkene_matches = _match_atom_sets(mol, "alkene")
    alkyne_matches = _match_atom_sets(mol, "alkyne")
    halide_matches = _match_atom_sets(mol, "halide")

    acid_oxygen = _atom_indices_from_matches(carbox_matches, atomic_nums=[8], mol=mol)
    ester_oxygen = _atom_indices_from_matches(ester_matches, atomic_nums=[8], mol=mol)
    amide_nitrogen = _atom_indices_from_matches(amide_matches, atomic_nums=[7], mol=mol)
    nitro_nitrogen = _atom_indices_from_matches(nitro_matches, atomic_nums=[7], mol=mol)
    phenol_oxygen = _atom_indices_from_matches(phenol_matches, atomic_nums=[8], mol=mol)
    alcohol_oxygen = _atom_indices_from_matches(alcohol_matches, atomic_nums=[8], mol=mol)
    ether_oxygen = _atom_indices_from_matches(ether_matches, atomic_nums=[8], mol=mol)
    amine_nitrogen = _atom_indices_from_matches(amine_matches, atomic_nums=[7], mol=mol)

    alcohol_oxygen = alcohol_oxygen - acid_oxygen - phenol_oxygen
    ether_oxygen = ether_oxygen - acid_oxygen - ester_oxygen
    amine_nitrogen = amine_nitrogen - amide_nitrogen - nitro_nitrogen

    aromatic_ring = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
    halide_present = any(
        mol.GetAtomWithIdx(int(idx)).GetAtomicNum() in {9, 17, 35, 53}
        for idx in _atom_indices_from_matches(halide_matches)
    )

    label_map = {
        "alcohol": int(len(alcohol_oxygen) > 0),
        "phenol": int(len(phenol_oxygen) > 0),
        "carboxylic_acid": int(len(carbox_matches) > 0),
        "ester": int(len(ester_matches) > 0),
        "ether": int(len(ether_oxygen) > 0),
        "aldehyde": int(len(aldehyde_matches) > 0),
        "ketone": int(len(ketone_matches) > 0),
        "amine": int(len(amine_nitrogen) > 0),
        "amide": int(len(amide_matches) > 0),
        "nitrile": int(len(nitrile_matches) > 0),
        "nitro": int(len(nitro_matches) > 0),
        "alkene": int(len(alkene_matches) > 0),
        "alkyne": int(len(alkyne_matches) > 0),
        "aromatic_ring": int(aromatic_ring),
        "halide": int(halide_present),
    }
    return [int(label_map.get(name, 0)) for name in label_names]


class IRFunctionalGroupDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        cache_path: Optional[str] = None,
        label_names: Sequence[str] = DEFAULT_FUNCTIONAL_GROUPS,
        canonicalize: bool = True,
    ):
        require_rdkit()
        self.data_path = data_path
        self.label_names = tuple(label_names)
        self.canonicalize = bool(canonicalize)
        self.data = torch.load(data_path, map_location="cpu")

        if not isinstance(self.data, dict):
            raise ValueError(f"Unexpected dataset format: {type(self.data)}")
        if "ir_spectra" not in self.data or "smiles" not in self.data:
            raise KeyError("Dataset must contain 'ir_spectra' and 'smiles'")

        self.ir_spectra = self.data["ir_spectra"].float()
        self.smiles_list = list(self.data["smiles"])
        if self.canonicalize:
            self.smiles_list = [canonicalize_smiles(s) for s in self.smiles_list]

        self.labels, self.invalid_smiles = self._load_or_build_labels(cache_path)
        self.num_labels = len(self.label_names)
        self.input_points = int(self.ir_spectra.size(-1))

    def _load_or_build_labels(self, cache_path: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache_path and os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            if (
                isinstance(cached, dict)
                and cached.get("version") == FUNCTIONAL_GROUP_VERSION
                and tuple(cached.get("label_names", ())) == self.label_names
                and int(cached.get("num_samples", -1)) == len(self.smiles_list)
            ):
                labels = cached["labels"].float()
                invalid = cached.get("invalid_smiles")
                if invalid is None:
                    invalid = torch.zeros(len(self.smiles_list), dtype=torch.bool)
                else:
                    invalid = invalid.bool()
                return labels, invalid

        labels = []
        invalid = []
        for smiles in self.smiles_list:
            mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            invalid.append(mol is None)
            labels.append(extract_functional_group_labels(smiles, self.label_names))

        label_tensor = torch.tensor(labels, dtype=torch.float32)
        invalid_tensor = torch.tensor(invalid, dtype=torch.bool)

        if cache_path:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            torch.save(
                {
                    "version": FUNCTIONAL_GROUP_VERSION,
                    "label_names": list(self.label_names),
                    "num_samples": len(self.smiles_list),
                    "labels": label_tensor,
                    "invalid_smiles": invalid_tensor,
                },
                cache_path,
            )
        return label_tensor, invalid_tensor

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, index: int):
        return self.ir_spectra[index], self.labels[index]


def load_split_indices(split_path: str) -> Tuple[List[int], List[int], List[int]]:
    split = torch.load(split_path, map_location="cpu")
    return list(split["train_indices"]), list(split["val_indices"]), list(split["test_indices"])


def compute_pos_weight(labels: torch.Tensor, min_value: float = 1.0, max_value: float = 20.0) -> torch.Tensor:
    pos = labels.sum(dim=0)
    neg = labels.size(0) - pos
    pos_weight = neg / pos.clamp_min(1.0)
    return pos_weight.clamp(min=min_value, max=max_value)


def _binary_counts(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float, float]:
    target = target.float()
    pred = pred.float()
    tp = float(((pred == 1) & (target == 1)).sum().item())
    tn = float(((pred == 0) & (target == 0)).sum().item())
    fp = float(((pred == 1) & (target == 0)).sum().item())
    fn = float(((pred == 0) & (target == 1)).sum().item())
    return tp, tn, fp, fn


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _prf_from_counts(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _ece_single_label(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    bin_edges = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=probs.device)
    total = probs.numel()
    ece = 0.0
    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        count = int(mask.sum().item())
        if count == 0:
            continue
        conf = float(probs[mask].mean().item())
        acc = float(targets[mask].float().mean().item())
        ece += (count / max(total, 1)) * abs(conf - acc)
    return float(ece)


def tune_thresholds(
    targets: torch.Tensor,
    probs: torch.Tensor,
    label_names: Sequence[str],
    min_threshold: float = 0.1,
    max_threshold: float = 0.9,
    num_steps: int = 17,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    targets = targets.float().cpu()
    probs = probs.float().cpu()
    grid = torch.linspace(min_threshold, max_threshold, steps=num_steps)
    thresholds = []
    best_f1 = {}

    for label_idx, label_name in enumerate(label_names):
        tgt = targets[:, label_idx]
        prb = probs[:, label_idx]
        positives = int(tgt.sum().item())
        negatives = int(tgt.numel() - positives)
        if positives == 0:
            thresholds.append(0.95)
            best_f1[label_name] = 0.0
            continue
        if negatives == 0:
            thresholds.append(0.05)
            best_f1[label_name] = 1.0
            continue

        label_best_thr = 0.5
        label_best_f1 = -1.0
        for thr in grid:
            pred = (prb >= float(thr)).float()
            tp, _tn, fp, fn = _binary_counts(pred, tgt)
            _precision, _recall, f1 = _prf_from_counts(tp, fp, fn)
            if f1 > label_best_f1 + 1e-8 or (abs(f1 - label_best_f1) <= 1e-8 and float(thr) > label_best_thr):
                label_best_f1 = f1
                label_best_thr = float(thr)
        thresholds.append(label_best_thr)
        best_f1[label_name] = float(max(label_best_f1, 0.0))

    return torch.tensor(thresholds, dtype=torch.float32), best_f1


def compute_multilabel_metrics(
    targets: torch.Tensor,
    probs: torch.Tensor,
    label_names: Sequence[str],
    thresholds: Optional[torch.Tensor] = None,
    calibration_bins: int = 10,
) -> Dict[str, object]:
    targets = targets.float().cpu()
    probs = probs.float().cpu()
    if targets.ndim != 2 or probs.ndim != 2:
        raise ValueError(f"Expected [N, C] tensors, got {tuple(targets.shape)} and {tuple(probs.shape)}")
    if targets.shape != probs.shape:
        raise ValueError(f"targets shape {tuple(targets.shape)} does not match probs shape {tuple(probs.shape)}")

    num_labels = targets.size(1)
    if thresholds is None:
        thresholds = torch.full((num_labels,), 0.5, dtype=torch.float32)
    else:
        thresholds = thresholds.float().cpu()

    preds = (probs >= thresholds.unsqueeze(0)).float()
    per_label = []
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    macro_brier = 0.0
    macro_ece = 0.0

    micro_tp = 0.0
    micro_fp = 0.0
    micro_fn = 0.0

    for label_idx, label_name in enumerate(label_names):
        tgt = targets[:, label_idx]
        prb = probs[:, label_idx]
        pred = preds[:, label_idx]
        tp, tn, fp, fn = _binary_counts(pred, tgt)
        precision, recall, f1 = _prf_from_counts(tp, fp, fn)
        support = int(tgt.sum().item())
        prevalence = float(tgt.mean().item())
        brier = float(((prb - tgt) ** 2).mean().item())
        ece = _ece_single_label(prb, tgt, n_bins=calibration_bins)

        auc = None
        ap = None
        if roc_auc_score is not None and support > 0 and support < tgt.numel():
            auc = float(roc_auc_score(tgt.numpy(), prb.numpy()))
        if average_precision_score is not None and support > 0:
            ap = float(average_precision_score(tgt.numpy(), prb.numpy()))

        per_label.append(
            {
                "name": label_name,
                "threshold": float(thresholds[label_idx].item()),
                "support": support,
                "prevalence": prevalence,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "brier": brier,
                "ece": ece,
                "auroc": auc,
                "average_precision": ap,
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }
        )

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        macro_brier += brier
        macro_ece += ece
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    macro_precision /= max(num_labels, 1)
    macro_recall /= max(num_labels, 1)
    macro_f1 /= max(num_labels, 1)
    macro_brier /= max(num_labels, 1)
    macro_ece /= max(num_labels, 1)

    micro_precision, micro_recall, micro_f1 = _prf_from_counts(micro_tp, micro_fp, micro_fn)
    subset_accuracy = float((preds == targets).all(dim=1).float().mean().item())
    hamming_accuracy = float((preds == targets).float().mean().item())
    positive_recall = float(((preds * targets).sum() / targets.sum().clamp_min(1.0)).item())
    predicted_positive_rate = float(preds.mean().item())
    true_positive_rate = float(targets.mean().item())

    metrics = {
        "subset_accuracy": subset_accuracy,
        "hamming_accuracy": hamming_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_brier": macro_brier,
        "macro_ece": macro_ece,
        "positive_recall": positive_recall,
        "predicted_positive_rate": predicted_positive_rate,
        "true_positive_rate": true_positive_rate,
        "per_label": per_label,
        "num_samples": int(targets.size(0)),
        "num_labels": int(num_labels),
    }

    valid_aurocs = [item["auroc"] for item in per_label if item["auroc"] is not None]
    valid_aps = [item["average_precision"] for item in per_label if item["average_precision"] is not None]
    if valid_aurocs:
        metrics["macro_auroc"] = float(sum(valid_aurocs) / len(valid_aurocs))
    else:
        metrics["macro_auroc"] = None
    if valid_aps:
        metrics["macro_average_precision"] = float(sum(valid_aps) / len(valid_aps))
    else:
        metrics["macro_average_precision"] = None
    return metrics


def metrics_to_json_ready(metrics: Dict[str, object]) -> Dict[str, object]:
    def convert(value):
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        if isinstance(value, tuple):
            return [convert(v) for v in value]
        return value

    return convert(metrics)


def save_json(path: str, payload: Dict[str, object]):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_json_ready(payload), f, indent=2, ensure_ascii=False)


