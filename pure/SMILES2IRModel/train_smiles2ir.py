import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]  # pure
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SMILES2IRModel.SMILES2IRModel import SMILES2IRRegressor
from util.dataloader import IRDataset
from util.tokenizer import build_vocab


DEFAULT_DATA_PATH = "data/raw_processed_data.pt"
DEFAULT_VOCAB_PATH = "pure/vocab.json"
DEFAULT_SPLIT_PATH = "checkpoints/data_split.pt"
DEFAULT_MODEL_PATH = "checkpoints/Smiles2IRModel/best_smiles2ir_model.pth"
DEFAULT_CONFIG_PATH = "checkpoints/Smiles2IRModel/smiles2ir_config.json"


def load_vocab(vocab_path: str, data_path: str) -> Dict[str, int]:
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
    except Exception:
        tqdm.write("vocab.json not found, building vocab...")
        build_vocab(data_path)
        with open(vocab_path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
    return {k: int(v) for k, v in token_to_id.items()}


def load_split_indices(split_path: str) -> Tuple[list, list, list]:
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"{split_path} not found. Please run the IR->SMILES training once to create consistent train/val/test split."
        )
    split_data = torch.load(split_path, map_location="cpu")
    return split_data["train_indices"], split_data["val_indices"], split_data["test_indices"]


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


def apply_token_mask(
    smiles_ids: torch.Tensor,
    mask_prob: float,
    pad_id: int,
    sos_id: int,
    eos_id: int,
    unk_id: int,
) -> torch.Tensor:
    if mask_prob <= 0:
        return smiles_ids

    device = smiles_ids.device
    rand_mask = torch.rand_like(smiles_ids.float(), device=device) < mask_prob
    special = smiles_ids.eq(pad_id) | smiles_ids.eq(sos_id) | smiles_ids.eq(eos_id)
    mask = rand_mask & (~special)
    if not mask.any():
        return smiles_ids
    masked = smiles_ids.clone()
    masked[mask] = unk_id
    return masked


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float,
    deriv_weight: float,
    huber_beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    huber = F.smooth_l1_loss(pred, target, beta=huber_beta)
    cosine = F.cosine_similarity(pred, target, dim=-1).mean()
    pred_diff = pred[:, 1:] - pred[:, :-1]
    tgt_diff = target[:, 1:] - target[:, :-1]
    deriv = F.smooth_l1_loss(pred_diff, tgt_diff, beta=huber_beta)
    loss = huber + cosine_weight * (1.0 - cosine) + deriv_weight * deriv
    return loss, {
        "huber": float(huber.item()),
        "cos": float(cosine.item()),
        "deriv": float(deriv.item()),
    }


def evaluate(model, loader, device, cosine_weight: float, deriv_weight: float, huber_beta: float):
    model.eval()
    total_loss = 0.0
    total_huber = 0.0
    total_deriv = 0.0
    total_cos = 0.0
    total_batches = 0

    with torch.no_grad():
        for smiles_ids, formula_vec, ir_target in loader:
            smiles_ids = smiles_ids.to(device)
            formula_vec = formula_vec.to(device)
            ir_target = ir_target.to(device)

            pred = model(smiles_ids, formula_vec=formula_vec)
            loss, info = compute_loss(
                pred,
                ir_target,
                cosine_weight=cosine_weight,
                deriv_weight=deriv_weight,
                huber_beta=huber_beta,
            )

            total_loss += float(loss.item())
            total_huber += info["huber"]
            total_deriv += info["deriv"]
            total_cos += info["cos"]
            total_batches += 1

    denom = max(total_batches, 1)
    return (
        total_loss / denom,
        total_huber / denom,
        total_deriv / denom,
        total_cos / denom,
    )


def build_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train stronger SMILES -> IR regressor for candidate reranking.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--vocab-path", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--split-path", type=str, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument("--cosine-loss-weight", type=float, default=0.1)
    parser.add_argument("--deriv-loss-weight", type=float, default=0.15)
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--token-mask-prob", type=float, default=0.05)

    parser.add_argument("--max-smiles-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--spectral-decoder-layers", type=int, default=3)
    parser.add_argument("--decoder-query-len", type=int, default=325)
    parser.add_argument("--refine-channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_to_id = load_vocab(args.vocab_path, args.data_path)
    pad_id = token_to_id.get("<PAD>", 0)
    sos_id = token_to_id.get("<SOS>", 1)
    eos_id = token_to_id.get("<EOS>", 2)
    unk_id = token_to_id.get("<UNK>", 3)

    dataset = IRDataset(args.data_path, token_to_id)
    train_idx, val_idx, test_idx = load_split_indices(args.split_path)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    collate_fn = build_smiles2ir_collate_fn(max_len=args.max_smiles_len, pad_id=pad_id)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    output_points = int(dataset[0][0].numel())
    model = SMILES2IRRegressor(
        vocab_size=len(token_to_id),
        output_points=output_points,
        formula_dim=dataset.formula_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        pad_id=pad_id,
        max_len=args.max_smiles_len,
        spectral_decoder_layers=args.spectral_decoder_layers,
        decoder_query_len=args.decoder_query_len,
        refine_channels=args.refine_channels,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * max(len(train_loader), 1))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    model_dir = os.path.dirname(args.model_path)
    config_dir = os.path.dirname(args.config_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    with open(args.config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab_size": len(token_to_id),
                "output_points": output_points,
                "formula_dim": dataset.formula_dim,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.ffn_dim,
                "dropout": args.dropout,
                "pad_id": pad_id,
                "max_len": args.max_smiles_len,
                "spectral_decoder_layers": args.spectral_decoder_layers,
                "decoder_query_len": args.decoder_query_len,
                "refine_channels": args.refine_channels,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )
    tqdm.write(f"Saved config to {args.config_path}")

    best_val_loss = float("inf")
    no_improve_epochs = 0
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_huber = 0.0
        train_deriv = 0.0
        train_cos = 0.0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]", leave=False)
        for smiles_ids, formula_vec, ir_target in pbar:
            smiles_ids = smiles_ids.to(device)
            formula_vec = formula_vec.to(device)
            ir_target = ir_target.to(device)

            smiles_in = apply_token_mask(
                smiles_ids,
                mask_prob=args.token_mask_prob,
                pad_id=pad_id,
                sos_id=sos_id,
                eos_id=eos_id,
                unk_id=unk_id,
            )

            optimizer.zero_grad()
            pred = model(smiles_in, formula_vec=formula_vec)
            loss, info = compute_loss(
                pred,
                ir_target,
                cosine_weight=args.cosine_loss_weight,
                deriv_weight=args.deriv_loss_weight,
                huber_beta=args.huber_beta,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_loss += float(loss.item())
            train_huber += info["huber"]
            train_deriv += info["deriv"]
            train_cos += info["cos"]
            batch_count += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                huber=f"{info['huber']:.4f}",
                deriv=f"{info['deriv']:.4f}",
                cos=f"{info['cos']:.4f}",
            )

        denom = max(batch_count, 1)
        train_loss /= denom
        train_huber /= denom
        train_deriv /= denom
        train_cos /= denom

        val_loss, val_huber, val_deriv, val_cos = evaluate(
            model,
            val_loader,
            device=device,
            cosine_weight=args.cosine_loss_weight,
            deriv_weight=args.deriv_loss_weight,
            huber_beta=args.huber_beta,
        )
        tqdm.write(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"train_loss={train_loss:.6f} train_huber={train_huber:.6f} train_deriv={train_deriv:.6f} train_cos={train_cos:.6f} | "
            f"val_loss={val_loss:.6f} val_huber={val_huber:.6f} val_deriv={val_deriv:.6f} val_cos={val_cos:.6f}"
        )

        if val_loss < best_val_loss - args.early_stop_min_delta:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), args.model_path)
            tqdm.write(f"Saved best model to {args.model_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.early_stop_patience:
                tqdm.write(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best_val_loss={best_val_loss:.6f}, patience={args.early_stop_patience})"
                )
                break

    if os.path.exists(args.model_path):
        best_state = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(best_state)
    test_loss, test_huber, test_deriv, test_cos = evaluate(
        model,
        test_loader,
        device=device,
        cosine_weight=args.cosine_loss_weight,
        deriv_weight=args.deriv_loss_weight,
        huber_beta=args.huber_beta,
    )

    tqdm.write("Training finished.")
    tqdm.write(f"Dataset size: {len(dataset)} | Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    tqdm.write(
        "Best-checkpoint test metrics | "
        f"test_loss={test_loss:.6f} test_huber={test_huber:.6f} test_deriv={test_deriv:.6f} test_cos={test_cos:.6f}"
    )


if __name__ == "__main__":
    main()
