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

from DiffusionAlign.diffusion_model import DiffusionBundle, build_schedule, q_sample
from util.dataloader import IRDataset
from util.tokenizer import build_vocab


DEFAULT_DATA_PATH = "data/raw_processed_data.pt"
DEFAULT_VOCAB_PATH = "pure/vocab.json"
DEFAULT_SPLIT_PATH = "checkpoints/data_split.pt"
DEFAULT_CKPT_PATH = "checkpoints/DiffusionAlign/diffusion_align.pth"
DEFAULT_CONFIG_PATH = "checkpoints/DiffusionAlign/diffusion_config.json"


def load_vocab(vocab_path: str, data_path: str) -> Dict[str, int]:
    if not os.path.exists(vocab_path):
        build_vocab(data_path, canonicalize_smiles=True)
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


def compute_recon_loss(pred: torch.Tensor, target: torch.Tensor, huber_beta: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    huber = F.smooth_l1_loss(pred, target, beta=huber_beta)
    pred_diff = pred[:, 1:] - pred[:, :-1]
    tgt_diff = target[:, 1:] - target[:, :-1]
    deriv = F.smooth_l1_loss(pred_diff, tgt_diff, beta=huber_beta)
    loss = huber + 0.1 * deriv
    return loss, {"huber": float(huber.item()), "deriv": float(deriv.item())}


def main():
    parser = argparse.ArgumentParser(description="Train conditional diffusion to align SMILES and IR latent.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--vocab-path", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--split-path", type=str, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--ckpt-path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)

    parser.add_argument("--stage", type=str, choices=["ae", "diffusion", "all"], default="all")
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--diff-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--cond-drop-prob", type=float, default=0.1)
    parser.add_argument("--huber-beta", type=float, default=0.5)

    parser.add_argument("--max-smiles-len", type=int, default=256)
    parser.add_argument("--latent-len", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--cond-dim", type=int, default=512)
    parser.add_argument("--cond-layers", type=int, default=6)
    parser.add_argument("--cond-nhead", type=int, default=8)
    parser.add_argument("--cond-ffn", type=int, default=2048)
    parser.add_argument("--cond-dropout", type=float, default=0.1)
    parser.add_argument("--diff-layers", type=int, default=8)
    parser.add_argument("--diff-nhead", type=int, default=8)
    parser.add_argument("--diff-ffn", type=int, default=2048)
    parser.add_argument("--diff-dropout", type=float, default=0.1)
    parser.add_argument("--multiscale-target", type=str, default="mid")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_to_id = load_vocab(args.vocab_path, args.data_path)
    pad_id = token_to_id.get("<PAD>", 0)

    dataset = IRDataset(
        args.data_path,
        token_to_id,
        canonicalize_smiles=True,
        randomize_smiles=True,
        randomize_prob=0.5,
        seed=args.seed,
    )
    train_idx, val_idx, test_idx = load_split_indices(args.split_path)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    collate_fn = build_smiles2ir_collate_fn(max_len=args.max_smiles_len, pad_id=pad_id)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    input_points = int(dataset[0][0].numel())
    config = {
        "vocab_size": len(token_to_id),
        "formula_dim": dataset.formula_dim,
        "input_points": input_points,
        "max_len": args.max_smiles_len,
        "pad_id": pad_id,
        "latent_len": args.latent_len,
        "latent_dim": args.latent_dim,
        "cond_dim": args.cond_dim,
        "cond_layers": args.cond_layers,
        "cond_nhead": args.cond_nhead,
        "cond_ffn": args.cond_ffn,
        "cond_dropout": args.cond_dropout,
        "diff_layers": args.diff_layers,
        "diff_nhead": args.diff_nhead,
        "diff_ffn": args.diff_ffn,
        "diff_dropout": args.diff_dropout,
        "timesteps": args.timesteps,
        "multiscale_target": args.multiscale_target,
    }

    bundle = DiffusionBundle.build_from_config(config, device=device)
    schedule = build_schedule(args.timesteps, device=device)

    optim_ae = optim.AdamW(bundle.autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optim_diff = optim.AdamW(
        list(bundle.condition_encoder.parameters()) + list(bundle.diffusion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.config_path), exist_ok=True)
    with open(args.config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=True, indent=2)

    if args.stage in {"ae", "all"}:
        best_val = float("inf")
        for epoch in range(args.ae_epochs):
            bundle.autoencoder.train()
            train_loss = 0.0
            for _smiles_ids, _formula_vec, ir_target in tqdm(
                train_loader, desc=f"AE {epoch + 1}/{args.ae_epochs}", leave=False
            ):
                ir_target = ir_target.to(device)
                optim_ae.zero_grad()
                pred = bundle.autoencoder(ir_target)
                loss, _ = compute_recon_loss(pred, ir_target, huber_beta=args.huber_beta)
                loss.backward()
                nn.utils.clip_grad_norm_(bundle.autoencoder.parameters(), args.grad_clip)
                optim_ae.step()
                train_loss += float(loss.item())

            bundle.autoencoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _smiles_ids, _formula_vec, ir_target in val_loader:
                    ir_target = ir_target.to(device)
                    pred = bundle.autoencoder(ir_target)
                    loss, _ = compute_recon_loss(pred, ir_target, huber_beta=args.huber_beta)
                    val_loss += float(loss.item())

            train_loss /= max(len(train_loader), 1)
            val_loss /= max(len(val_loader), 1)
            tqdm.write(f"AE epoch {epoch + 1} | train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(bundle.autoencoder.state_dict(), args.ckpt_path + ".ae")

    if args.stage in {"diffusion", "all"}:
        if os.path.exists(args.ckpt_path + ".ae"):
            bundle.autoencoder.load_state_dict(torch.load(args.ckpt_path + ".ae", map_location="cpu"))
        bundle.autoencoder.eval()
        for p in bundle.autoencoder.parameters():
            p.requires_grad = False

        best_val = float("inf")
        for epoch in range(args.diff_epochs):
            bundle.condition_encoder.train()
            bundle.diffusion.train()
            train_loss = 0.0

            for smiles_ids, formula_vec, ir_target in tqdm(
                train_loader, desc=f"Diff {epoch + 1}/{args.diff_epochs}", leave=False
            ):
                smiles_ids = smiles_ids.to(device)
                formula_vec = formula_vec.to(device)
                ir_target = ir_target.to(device)

                with torch.no_grad():
                    z0 = bundle.autoencoder.encode(ir_target)

                t = torch.randint(0, args.timesteps, (z0.size(0),), device=device)
                noise = torch.randn_like(z0)
                z_t = q_sample(z0, t, noise, schedule)

                cond_mem, cond_mask = bundle.condition_encoder.encode_ids(smiles_ids, formula_vec=formula_vec)
                optim_diff.zero_grad()
                pred_noise = bundle.diffusion(
                    z_t,
                    t,
                    cond_memory=cond_mem,
                    cond_pad_mask=cond_mask,
                    cond_drop_prob=args.cond_drop_prob,
                )
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(bundle.condition_encoder.parameters()) + list(bundle.diffusion.parameters()),
                    args.grad_clip,
                )
                optim_diff.step()
                train_loss += float(loss.item())

            bundle.condition_encoder.eval()
            bundle.diffusion.eval()
            val_loss = 0.0
            with torch.no_grad():
                for smiles_ids, formula_vec, ir_target in val_loader:
                    smiles_ids = smiles_ids.to(device)
                    formula_vec = formula_vec.to(device)
                    ir_target = ir_target.to(device)

                    z0 = bundle.autoencoder.encode(ir_target)
                    t = torch.randint(0, args.timesteps, (z0.size(0),), device=device)
                    noise = torch.randn_like(z0)
                    z_t = q_sample(z0, t, noise, schedule)

                    cond_mem, cond_mask = bundle.condition_encoder.encode_ids(smiles_ids, formula_vec=formula_vec)
                    pred_noise = bundle.diffusion(z_t, t, cond_memory=cond_mem, cond_pad_mask=cond_mask)
                    loss = F.mse_loss(pred_noise, noise)
                    val_loss += float(loss.item())

            train_loss /= max(len(train_loader), 1)
            val_loss /= max(len(val_loader), 1)
            tqdm.write(f"Diff epoch {epoch + 1} | train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "autoencoder": bundle.autoencoder.state_dict(),
                        "condition_encoder": bundle.condition_encoder.state_dict(),
                        "diffusion": bundle.diffusion.state_dict(),
                        "config": config,
                    },
                    args.ckpt_path,
                )


if __name__ == "__main__":
    main()
