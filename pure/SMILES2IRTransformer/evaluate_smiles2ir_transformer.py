import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]  # pure
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SMILES2IRTransformer.SMILES2IRTransformer import SMILES2IRTransformer
from util.dataloader import IRDataset


DEFAULT_DATA_PATH = "data/raw_processed_data.pt"
DEFAULT_VOCAB_PATH = "pure/vocab.json"
DEFAULT_SPLIT_PATH = "checkpoints/data_split.pt"
DEFAULT_MODEL_PATH = "checkpoints/SMILES2IRTransformer/best_smiles2ir_transformer.pth"
DEFAULT_CONFIG_PATH = "checkpoints/SMILES2IRTransformer/smiles2ir_transformer_config.json"


def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    return {k: int(v) for k, v in token_to_id.items()}


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split_indices(split_path: str, split_name: str):
    split_data = torch.load(split_path, map_location="cpu")
    key = f"{split_name}_indices"
    if key not in split_data:
        raise KeyError(f"{key} not found in {split_path}")
    return split_data[key]


def build_model(model_path: str, config: Dict, dataset_formula_dim: int, device: torch.device) -> SMILES2IRTransformer:
    model = SMILES2IRTransformer(
        vocab_size=config["vocab_size"],
        output_points=config["output_points"],
        formula_dim=config.get("formula_dim", dataset_formula_dim),
        d_model=config["d_model"],
        nhead=config["nhead"],
        encoder_layers=config.get("encoder_layers", 6),
        decoder_layers=config.get("decoder_layers", 6),
        decoder_query_len=config.get("decoder_query_len"),
        dim_feedforward=config.get("dim_feedforward", config.get("ffn_dim", 2048)),
        dropout=config.get("dropout", 0.1),
        pad_id=config["pad_id"],
        max_len=config["max_len"],
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def build_smiles2ir_collate_fn(max_len: int, pad_id: int = 0):
    def collate(batch):
        smiles_list = []
        formula_list = []
        ir_list = []
        for ir_spectrum, formula_vec, smiles_ids in batch:
            smiles_list.append(smiles_ids[:max_len])
            formula_list.append(formula_vec.float())
            ir_list.append(ir_spectrum.float())

        max_batch_len = max(seq.size(0) for seq in smiles_list)
        padded = []
        for seq in smiles_list:
            if seq.size(0) < max_batch_len:
                pad = torch.full((max_batch_len - seq.size(0),), pad_id, dtype=seq.dtype)
                seq = torch.cat([seq, pad], dim=0)
            padded.append(seq)

        smiles_batch = torch.stack(padded, dim=0).long()
        formula_batch = torch.stack(formula_list, dim=0).float()
        ir_batch = torch.stack(ir_list, dim=0).float()
        return smiles_batch, formula_batch, ir_batch

    return collate


def evaluate_model(model, loader, device) -> Dict[str, float]:
    total_mse = 0.0
    total_mae = 0.0
    total_cos = 0.0
    total_batches = 0

    sse = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    numel = 0

    with torch.no_grad():
        for smiles_ids, formula_vec, ir_target in loader:
            smiles_ids = smiles_ids.to(device)
            formula_vec = formula_vec.to(device)
            ir_target = ir_target.to(device)
            pred = model(smiles_ids, formula_vec=formula_vec)

            mse = F.mse_loss(pred, ir_target)
            mae = F.l1_loss(pred, ir_target)
            cos = F.cosine_similarity(pred, ir_target, dim=-1).mean()

            total_mse += float(mse.item())
            total_mae += float(mae.item())
            total_cos += float(cos.item())
            total_batches += 1

            diff = (pred - ir_target).float()
            y = ir_target.float()
            sse += float((diff * diff).sum().item())
            sum_y += float(y.sum().item())
            sum_y2 += float((y * y).sum().item())
            numel += int(y.numel())

    denom = max(total_batches, 1)
    tss = sum_y2 - (sum_y * sum_y) / max(numel, 1)
    r2 = 1.0 - (sse / max(tss, 1e-12))
    return {
        "mse": total_mse / denom,
        "mae": total_mae / denom,
        "cosine": total_cos / denom,
        "r2": float(r2),
    }


def evaluate_mean_baseline(mean_spectrum: torch.Tensor, loader, device) -> Dict[str, float]:
    total_mse = 0.0
    total_mae = 0.0
    total_cos = 0.0
    total_batches = 0

    sse = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    numel = 0

    mean_spectrum = mean_spectrum.to(device)
    with torch.no_grad():
        for _smiles_ids, _formula_vec, ir_target in loader:
            ir_target = ir_target.to(device)
            pred = mean_spectrum.unsqueeze(0).expand(ir_target.size(0), -1)

            mse = F.mse_loss(pred, ir_target)
            mae = F.l1_loss(pred, ir_target)
            cos = F.cosine_similarity(pred, ir_target, dim=-1).mean()

            total_mse += float(mse.item())
            total_mae += float(mae.item())
            total_cos += float(cos.item())
            total_batches += 1

            diff = (pred - ir_target).float()
            y = ir_target.float()
            sse += float((diff * diff).sum().item())
            sum_y += float(y.sum().item())
            sum_y2 += float((y * y).sum().item())
            numel += int(y.numel())

    denom = max(total_batches, 1)
    tss = sum_y2 - (sum_y * sum_y) / max(numel, 1)
    r2 = 1.0 - (sse / max(tss, 1e-12))
    return {
        "mse": total_mse / denom,
        "mae": total_mae / denom,
        "cosine": total_cos / denom,
        "r2": float(r2),
    }


def build_train_mean_spectrum(dataset: IRDataset, train_indices) -> torch.Tensor:
    total = torch.zeros_like(dataset[0][0], dtype=torch.float32)
    for idx in train_indices:
        total += dataset[idx][0].float()
    return total / max(len(train_indices), 1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SMILES -> IR transformer regressor.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--vocab-path", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--split-path", type=str, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--split", type=str, choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--with-mean-baseline", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_to_id = load_vocab(args.vocab_path)
    config = load_config(args.config_path)
    dataset = IRDataset(args.data_path, token_to_id)

    if args.split == "all":
        eval_dataset = dataset
    else:
        split_indices = load_split_indices(args.split_path, args.split)
        eval_dataset = Subset(dataset, split_indices)

    collate_fn = build_smiles2ir_collate_fn(max_len=int(config["max_len"]), pad_id=int(config["pad_id"]))
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = build_model(args.model_path, config, dataset_formula_dim=dataset.formula_dim, device=device)
    metrics = evaluate_model(model, loader, device)

    print("=" * 70)
    print("SMILES->IR Transformer evaluation finished")
    print(f"Device                 : {device}")
    print(f"Model checkpoint       : {args.model_path}")
    print(f"Evaluate split         : {args.split}")
    print(f"Dataset size           : {len(eval_dataset)}")
    print("-" * 70)
    print(f"MSE                    : {metrics['mse']:.8f}")
    print(f"MAE                    : {metrics['mae']:.8f}")
    print(f"Mean Cosine Similarity : {metrics['cosine']:.8f}")
    print(f"R2                     : {metrics['r2']:.8f}")

    if args.with_mean_baseline:
        train_indices = load_split_indices(args.split_path, "train")
        mean_spectrum = build_train_mean_spectrum(dataset, train_indices)
        base = evaluate_mean_baseline(mean_spectrum=mean_spectrum, loader=loader, device=device)
        print("-" * 70)
        print("Mean-spectrum baseline:")
        print(f"MSE                    : {base['mse']:.8f}")
        print(f"MAE                    : {base['mae']:.8f}")
        print(f"Mean Cosine Similarity : {base['cosine']:.8f}")
        print(f"R2                     : {base['r2']:.8f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
