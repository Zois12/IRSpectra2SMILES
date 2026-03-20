import argparse
import json
import os
from typing import Any, Dict, List

import torch

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
except Exception:
    Chem = None
    RDLogger = None


DEFAULT_INPUT_PATH = "data/raw_processed_data.pt"
DEFAULT_OUTPUT_PATH = "data/raw_processed_data_filtered.pt"
DEFAULT_REPORT_PATH = "data/raw_processed_data_filtered_report.json"


def validate_smiles(smiles: Any):
    if Chem is None:
        raise ImportError("RDKit is required to filter invalid SMILES.")
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return False, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, None
    return True, mol


def filter_aligned_field(value: Any, keep_indices: List[int], total_size: int):
    if torch.is_tensor(value):
        if value.dim() > 0 and value.size(0) == total_size:
            index_tensor = torch.tensor(keep_indices, dtype=torch.long)
            return value.index_select(0, index_tensor)
        return value

    if isinstance(value, list) and len(value) == total_size:
        return [value[i] for i in keep_indices]

    if isinstance(value, tuple) and len(value) == total_size:
        return tuple(value[i] for i in keep_indices)

    return value


def main():
    parser = argparse.ArgumentParser(description="Filter invalid SMILES from a processed .pt dataset.")
    parser.add_argument("--input-path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report-path", type=str, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--canonicalize", action="store_true", help="Replace kept SMILES with canonical RDKit SMILES.")
    parser.add_argument("--show-invalid", type=int, default=20, help="How many invalid examples to print/save.")
    args = parser.parse_args()

    if Chem is None:
        raise ImportError("RDKit is not available. Please install RDKit before running this script.")

    data = torch.load(args.input_path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict dataset, got {type(data)}")
    if "smiles" not in data:
        raise KeyError("Dataset must contain key 'smiles'")

    smiles_list = data["smiles"]
    total_size = len(smiles_list)

    keep_indices: List[int] = []
    invalid_indices: List[int] = []
    invalid_examples: List[Dict[str, Any]] = []
    canonical_smiles: List[str] = []

    for idx, smiles in enumerate(smiles_list):
        is_valid, mol = validate_smiles(smiles)
        if is_valid:
            keep_indices.append(idx)
            if args.canonicalize:
                canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            continue

        invalid_indices.append(idx)
        if len(invalid_examples) < args.show_invalid:
            invalid_examples.append({
                "index": idx,
                "smiles": smiles,
            })

    filtered_data = {}
    for key, value in data.items():
        filtered_data[key] = filter_aligned_field(value, keep_indices=keep_indices, total_size=total_size)

    if args.canonicalize:
        filtered_data["smiles"] = canonical_smiles

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(filtered_data, args.output_path)

    report = {
        "input_path": args.input_path,
        "output_path": args.output_path,
        "total_samples": total_size,
        "kept_samples": len(keep_indices),
        "removed_samples": len(invalid_indices),
        "kept_ratio": len(keep_indices) / max(total_size, 1),
        "removed_ratio": len(invalid_indices) / max(total_size, 1),
        "invalid_indices": invalid_indices,
        "invalid_examples": invalid_examples,
        "canonicalized": bool(args.canonicalize),
        "filtered_keys": list(filtered_data.keys()),
    }

    report_dir = os.path.dirname(args.report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Input dataset   : {args.input_path}")
    print(f"Output dataset  : {args.output_path}")
    print(f"Report path     : {args.report_path}")
    print(f"Total samples   : {total_size}")
    print(f"Kept samples    : {len(keep_indices)}")
    print(f"Removed samples : {len(invalid_indices)}")
    print(f"Kept ratio      : {report['kept_ratio']:.4%}")
    print(f"Removed ratio   : {report['removed_ratio']:.4%}")

    if invalid_examples:
        print("Invalid examples:")
        for item in invalid_examples:
            print(f"  - idx={item['index']} smiles={item['smiles']}")


if __name__ == "__main__":
    main()
