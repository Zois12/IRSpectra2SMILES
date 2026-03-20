import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]  # pure
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TransformerModel.TransformerModel import IRFormulaTransformer
from DiffusionAlign.diffusion_model import DiffusionBundle, build_schedule, q_sample
from util.dataloader import IRDataset, collate_fn
from util.tokenizer import build_vocab

TRAIN_EPOCH = 35
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "data/raw_processed_data.pt"
VOCAB_PATH = "pure/vocab.json"
SPLIT_PATH = "checkpoints/data_split.pt"
MODEL_PATH = "checkpoints/TransformerModel/best_transformer_model.pth"
BEST_LOSS_MODEL_PATH = "checkpoints/TransformerModel/best_transformer_loss_model.pth"
CONFIG_PATH = "checkpoints/TransformerModel/transformer_config.json"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42

# Transformer config
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
MAX_TGT_LEN = 256
MAX_MEMORY_LEN = 1024
ENCODER_BUFFER_LAYERS = 2
ENCODER_BUFFER_DIM_FEEDFORWARD = 2048
ENCODER_MULTISCALE_TARGET = "mid"
ENCODER_USE_COORDCONV = True
VAL_SEQ_EM_MAX_SAMPLES = 300
VAL_SEQ_EM_MAX_LEN = 120
VAL_SEQ_EM_EVERY = 2
EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_DELTA = 1e-4
SPECTRUM_NOISE_STD = 0.0
LABEL_SMOOTHING = 0.1
USE_TANIMOTO_AUX_LOSS = True
TANIMOTO_LOSS_WEIGHT = 0.05
USE_MASKED_PATCH_AUX = True
MASKED_PATCH_RATIO = 0.15
MASKED_PATCH_SIZE = 16
MASKED_PATCH_LOSS_WEIGHT = 0.10
MASKED_PATCH_DERIV_WEIGHT = 0.25
MASKED_PATCH_HIDDEN_DIM = 256
MASKED_PATCH_MASK_VALUE = 0.0
MASKED_PATCH_APPLY_TO_SEQ2SEQ = True
CANONICALIZE_SMILES = True
RANDOMIZE_SMILES = False
RANDOMIZE_PROB = 0.0
REBUILD_VOCAB = True

# DataLoader perf
NUM_WORKERS = 4
PIN_MEMORY = True

# Weights & Biases
USE_WANDB = True
WANDB_PROJECT = "IRSpectra2SMILES"
WANDB_ENTITY = None  # set to your W&B entity/team or keep None
WANDB_RUN_NAME = "transformer_ir2smiles"

# LR scheduler (ReduceLROnPlateau)
USE_REDUCE_LR = True
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 2
PLATEAU_MIN_LR = 1e-6
PLATEAU_THRESHOLD = 1e-4

# Diffusion alignment (soft-token path)
USE_DIFFUSION_ALIGN = True
DIFFUSION_CKPT_PATH = "checkpoints/DiffusionAlign/diffusion_align.pth"
DIFFUSION_LOSS_WEIGHT = 0.1
DIFFUSION_TAU = 1.0


def get_tokenizer():
    if REBUILD_VOCAB or not os.path.exists(VOCAB_PATH):
        tqdm.write("Building vocab (canonicalized SMILES)...")
        build_vocab(DATA_PATH, canonicalize_smiles=CANONICALIZE_SMILES)
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
    else:
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
    return {k: int(v) for k, v in token_to_id.items()}


def get_or_create_splits(total_size: int, split_path: str):
    if os.path.exists(split_path):
        split_data = torch.load(split_path, map_location="cpu")
        train_idx = split_data["train_indices"]
        val_idx = split_data["val_indices"]
        test_idx = split_data["test_indices"]
        tqdm.write(f"Loaded existing split from {split_path}")
        return train_idx, val_idx, test_idx

    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(total_size, generator=g).tolist()

    train_idx = perm[:train_size]
    val_idx = perm[train_size : train_size + val_size]
    test_idx = perm[train_size + val_size :]

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    torch.save(
        {
            "seed": SEED,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
        },
        split_path,
    )
    tqdm.write(f"Saved split to {split_path}")
    tqdm.write(f"Split size: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def load_diffusion_bundle(ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config")
    if config is None:
        raise KeyError("Diffusion checkpoint missing config")

    bundle = DiffusionBundle.build_from_config(config, device=device)
    bundle.autoencoder.load_state_dict(ckpt["autoencoder"])
    bundle.condition_encoder.load_state_dict(ckpt["condition_encoder"])
    bundle.diffusion.load_state_dict(ckpt["diffusion"])

    for module in (bundle.autoencoder, bundle.condition_encoder, bundle.diffusion):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    schedule = build_schedule(int(config.get("timesteps", 1000)), device=device)
    return bundle, schedule, config


def token_accuracy(logits: torch.Tensor, target: torch.Tensor, pad_id: int = 0):
    pred = torch.argmax(logits, dim=-1)
    valid_mask = target != pad_id
    correct = ((pred == target) & valid_mask).sum().item()
    total = valid_mask.sum().item()
    return correct, total


def soft_token_tanimoto_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_id: int,
    ignore_token_ids: Optional[list] = None,
    eps: float = 1e-8,
):
    vocab_size = logits.size(-1)
    probs = F.softmax(logits, dim=-1)
    valid_mask = target.ne(pad_id).float()

    pred_counts = (probs * valid_mask.unsqueeze(-1)).sum(dim=1)
    target_counts = torch.zeros_like(pred_counts)
    safe_target = target.masked_fill(target.eq(pad_id), 0)
    target_counts.scatter_add_(1, safe_target, valid_mask)

    pred_presence = 1.0 - torch.exp(-pred_counts)
    target_presence = (target_counts > 0).float()

    if ignore_token_ids:
        ignore_idx = torch.tensor(sorted(set(ignore_token_ids)), device=logits.device, dtype=torch.long)
        ignore_idx = ignore_idx[(ignore_idx >= 0) & (ignore_idx < vocab_size)]
        if ignore_idx.numel() > 0:
            pred_presence.index_fill_(1, ignore_idx, 0.0)
            target_presence.index_fill_(1, ignore_idx, 0.0)

    intersection = (pred_presence * target_presence).sum(dim=1)
    denominator = (
        pred_presence.square().sum(dim=1)
        + target_presence.square().sum(dim=1)
        - intersection
    ).clamp_min(eps)
    tanimoto = intersection / denominator
    return 1.0 - tanimoto.mean(), tanimoto.mean()


class MaskedPatchReconstructionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        output_points: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = max(int(hidden_dim), 64)
        mid_dim = max(hidden_dim // 2, 32)
        self.output_points = int(output_points)
        self.proj = nn.Sequential(
            nn.Conv1d(d_model, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, mid_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Conv1d(mid_dim, 1, kernel_size=1),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        spectral_memory = memory[:, 1:, :].transpose(1, 2)
        x = self.proj(spectral_memory)
        x = F.interpolate(x, size=self.output_points, mode="linear", align_corners=False)
        x = self.refine(x)
        return x.squeeze(1)


def apply_masked_patch_augmentation(
    spectra: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
    mask_value: float = 0.0,
):
    if spectra.dim() != 2:
        raise ValueError(f"Expected spectra shape [B, L], got {tuple(spectra.shape)}")
    if mask_ratio <= 0 or patch_size <= 0:
        return spectra, torch.zeros_like(spectra, dtype=torch.bool)

    bsz, seq_len = spectra.shape
    num_patches = max((seq_len + patch_size - 1) // patch_size, 1)
    num_masked = min(num_patches, max(1, int(round(num_patches * mask_ratio))))

    masked = spectra.clone()
    mask = torch.zeros_like(spectra, dtype=torch.bool)
    for batch_idx in range(bsz):
        patch_indices = torch.randperm(num_patches, device=spectra.device)[:num_masked]
        for patch_idx in patch_indices.tolist():
            start = patch_idx * patch_size
            end = min(start + patch_size, seq_len)
            mask[batch_idx, start:end] = True

    masked[mask] = float(mask_value)
    return masked, mask


def masked_patch_reconstruction_loss(
    pred_spectra: torch.Tensor,
    target_spectra: torch.Tensor,
    patch_mask: torch.Tensor,
    deriv_weight: float = 0.25,
):
    if not patch_mask.any():
        zero = pred_spectra.new_zeros(())
        return zero, zero, zero

    huber = F.smooth_l1_loss(pred_spectra[patch_mask], target_spectra[patch_mask])

    pair_mask = patch_mask[:, 1:] & patch_mask[:, :-1]
    if pair_mask.any():
        pred_diff = pred_spectra[:, 1:] - pred_spectra[:, :-1]
        target_diff = target_spectra[:, 1:] - target_spectra[:, :-1]
        deriv = F.smooth_l1_loss(pred_diff[pair_mask], target_diff[pair_mask])
    else:
        deriv = pred_spectra.new_zeros(())

    loss = huber + float(deriv_weight) * deriv
    return loss, huber, deriv


def _normalize_token_seq(ids: torch.Tensor, pad_id: int, sos_id: int, eos_id: int):
    seq = []
    for tid in ids.tolist():
        if tid == eos_id:
            break
        if tid in {pad_id, sos_id}:
            continue
        seq.append(int(tid))
    return tuple(seq)


def validate(model, loader, criterion, device, pad_id: int, tanimoto_ignore_ids: Optional[list] = None):
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_tanimoto_aux_loss = 0.0
    total_soft_tanimoto = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        val_pbar = tqdm(loader, desc="Validate", leave=False)
        for spectra, formula_vec, smiles_ids in val_pbar:
            spectra = spectra.to(device)
            formula_vec = formula_vec.to(device)
            smiles_ids = smiles_ids.to(device)

            decoder_input = smiles_ids[:, :-1]
            target = smiles_ids[:, 1:]
            outputs = model(spectra, formula_vec, decoder_input)
            logits = outputs
            ce_loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            if USE_TANIMOTO_AUX_LOSS and TANIMOTO_LOSS_WEIGHT > 0:
                tanimoto_aux_loss, soft_tanimoto = soft_token_tanimoto_loss(
                    logits,
                    target,
                    pad_id=pad_id,
                    ignore_token_ids=tanimoto_ignore_ids,
                )
            else:
                tanimoto_aux_loss = torch.zeros((), device=logits.device)
                soft_tanimoto = torch.zeros((), device=logits.device)
            loss = ce_loss + TANIMOTO_LOSS_WEIGHT * tanimoto_aux_loss

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_tanimoto_aux_loss += tanimoto_aux_loss.item()
            total_soft_tanimoto += soft_tanimoto.item()
            c, t = token_accuracy(logits, target, pad_id=pad_id)
            total_correct += c
            total_tokens += t
            val_pbar.set_postfix(loss=f"{loss.item():.4f}", tan=f"{soft_tanimoto.item():.4f}")

    avg_loss = total_loss / max(len(loader), 1)
    acc = total_correct / max(total_tokens, 1)
    avg_ce_loss = total_ce_loss / max(len(loader), 1)
    avg_tanimoto_aux_loss = total_tanimoto_aux_loss / max(len(loader), 1)
    avg_soft_tanimoto = total_soft_tanimoto / max(len(loader), 1)
    return avg_loss, acc, avg_ce_loss, avg_tanimoto_aux_loss, avg_soft_tanimoto


def evaluate_val_sequence_exact_match(
    model,
    loader,
    device,
    pad_id: int,
    sos_id: int,
    eos_id: int,
    max_len: int,
    max_samples: int = 0,
):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for spectra, formula_vec, smiles_ids in loader:
            spectra = spectra.to(device)
            formula_vec = formula_vec.to(device)
            generated_ids = model.generate(
                spectra,
                formula_vec,
                max_len=max_len,
                sos_id=sos_id,
                eos_id=eos_id,
            ).cpu()

            target_ids = smiles_ids.cpu()
            for i in range(target_ids.size(0)):
                pred_seq = _normalize_token_seq(generated_ids[i], pad_id=pad_id, sos_id=sos_id, eos_id=eos_id)
                gt_seq = _normalize_token_seq(target_ids[i], pad_id=pad_id, sos_id=sos_id, eos_id=eos_id)
                if pred_seq == gt_seq:
                    correct += 1
                total += 1
                if max_samples > 0 and total >= max_samples:
                    return correct / max(total, 1)

    return correct / max(total, 1)


def train_model():
    wandb_run = None
    if USE_WANDB:
        try:
            import wandb

            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=WANDB_RUN_NAME,
                config={
                    "train_epoch": TRAIN_EPOCH,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "grad_clip": GRAD_CLIP,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers": NUM_LAYERS,
                    "dim_feedforward": DIM_FEEDFORWARD,
                    "dropout": DROPOUT,
                    "max_tgt_len": MAX_TGT_LEN,
                    "max_memory_len": MAX_MEMORY_LEN,
                    "encoder_buffer_layers": ENCODER_BUFFER_LAYERS,
                    "encoder_buffer_dim_feedforward": ENCODER_BUFFER_DIM_FEEDFORWARD,
                    "encoder_multiscale_target": ENCODER_MULTISCALE_TARGET,
                    "encoder_use_coordconv": ENCODER_USE_COORDCONV,
                    "label_smoothing": LABEL_SMOOTHING,
                    "use_tanimoto_aux_loss": USE_TANIMOTO_AUX_LOSS,
                    "tanimoto_loss_weight": TANIMOTO_LOSS_WEIGHT,
                    "use_masked_patch_aux": USE_MASKED_PATCH_AUX,
                    "masked_patch_ratio": MASKED_PATCH_RATIO,
                    "masked_patch_size": MASKED_PATCH_SIZE,
                    "masked_patch_loss_weight": MASKED_PATCH_LOSS_WEIGHT,
                    "masked_patch_deriv_weight": MASKED_PATCH_DERIV_WEIGHT,
                    "masked_patch_hidden_dim": MASKED_PATCH_HIDDEN_DIM,
                    "masked_patch_apply_to_seq2seq": MASKED_PATCH_APPLY_TO_SEQ2SEQ,
                    "spectrum_noise_std": SPECTRUM_NOISE_STD,
                    "use_reduce_lr": USE_REDUCE_LR,
                    "plateau_factor": PLATEAU_FACTOR,
                    "plateau_patience": PLATEAU_PATIENCE,
                    "plateau_min_lr": PLATEAU_MIN_LR,
                    "plateau_threshold": PLATEAU_THRESHOLD,
                    "randomize_smiles": RANDOMIZE_SMILES,
                    "randomize_prob": RANDOMIZE_PROB,
                },
            )
        except Exception as exc:
            tqdm.write(f"W&B disabled (import/init failed): {exc}")
            wandb_run = None

    tokenizer = get_tokenizer()
    pad_id = tokenizer.get("<PAD>", 0)
    sos_id = tokenizer.get("<SOS>", 1)
    eos_id = tokenizer.get("<EOS>", 2)
    tanimoto_ignore_ids = [tid for tok, tid in tokenizer.items() if tok in {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}]

    dataset = IRDataset(
        DATA_PATH,
        tokenizer,
        canonicalize_smiles=CANONICALIZE_SMILES,
        randomize_smiles=False,
        randomize_prob=0.0,
        seed=SEED,
    )
    total_size = len(dataset)
    train_idx, val_idx, test_idx = get_or_create_splits(total_size, SPLIT_PATH)

    tqdm.write(f"Dataset size: {total_size}")
    tqdm.write(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    tqdm.write(f"Formula feature dim: {dataset.formula_dim}")

    if RANDOMIZE_SMILES and RANDOMIZE_PROB > 0:
        train_base = IRDataset(
            DATA_PATH,
            tokenizer,
            formula_vocab=dataset.formula_vocab,
            data=dataset.data,
            canonicalize_smiles=CANONICALIZE_SMILES,
            randomize_smiles=True,
            randomize_prob=RANDOMIZE_PROB,
            seed=SEED,
        )
        train_dataset = Subset(train_base, train_idx)
    else:
        train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and DEVICE.type == "cuda",
    )

    model = IRFormulaTransformer(
        vocab_size=len(tokenizer),
        formula_dim=dataset.formula_dim,
        input_points=dataset[0][0].numel(),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_tgt_len=MAX_TGT_LEN,
        max_memory_len=MAX_MEMORY_LEN,
        encoder_buffer_layers=ENCODER_BUFFER_LAYERS,
        encoder_buffer_dim_feedforward=ENCODER_BUFFER_DIM_FEEDFORWARD,
        encoder_multiscale_target=ENCODER_MULTISCALE_TARGET,
        encoder_use_coordconv=ENCODER_USE_COORDCONV,
        pad_id=pad_id,
        sos_id=sos_id,
        eos_id=eos_id,
    ).to(DEVICE)

    diffusion_bundle = None
    diffusion_schedule = None
    diffusion_config = None
    if USE_DIFFUSION_ALIGN:
        diffusion_bundle, diffusion_schedule, diffusion_config = load_diffusion_bundle(DIFFUSION_CKPT_PATH, DEVICE)
        tqdm.write(
            f"Loaded diffusion bundle from {DIFFUSION_CKPT_PATH} "
            f"(latent_len={diffusion_config.get('latent_len')}, latent_dim={diffusion_config.get('latent_dim')})"
        )

    masked_patch_head = None
    if USE_MASKED_PATCH_AUX:
        masked_patch_head = MaskedPatchReconstructionHead(
            d_model=D_MODEL,
            output_points=dataset[0][0].numel(),
            hidden_dim=MASKED_PATCH_HIDDEN_DIM,
            dropout=DROPOUT,
        ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=LABEL_SMOOTHING)
    optim_params = list(model.parameters())
    if masked_patch_head is not None:
        optim_params.extend(masked_patch_head.parameters())
    optimizer = optim.AdamW(optim_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if USE_REDUCE_LR:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=PLATEAU_FACTOR,
            patience=PLATEAU_PATIENCE,
            threshold=PLATEAU_THRESHOLD,
            min_lr=PLATEAU_MIN_LR,
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCH)

    model_dir = os.path.dirname(MODEL_PATH)
    config_dir = os.path.dirname(CONFIG_PATH)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab_size": len(tokenizer),
                "formula_dim": dataset.formula_dim,
                "input_points": dataset[0][0].numel(),
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "num_layers": NUM_LAYERS,
                "dim_feedforward": DIM_FEEDFORWARD,
                "dropout": DROPOUT,
                "max_tgt_len": MAX_TGT_LEN,
                "max_memory_len": MAX_MEMORY_LEN,
                "encoder_buffer_layers": ENCODER_BUFFER_LAYERS,
                "encoder_buffer_dim_feedforward": ENCODER_BUFFER_DIM_FEEDFORWARD,
                "encoder_multiscale_target": ENCODER_MULTISCALE_TARGET,
                "encoder_use_coordconv": ENCODER_USE_COORDCONV,
                "pad_id": pad_id,
                "sos_id": sos_id,
                "eos_id": eos_id,
                "label_smoothing": LABEL_SMOOTHING,
                "use_tanimoto_aux_loss": USE_TANIMOTO_AUX_LOSS,
                "tanimoto_loss_weight": TANIMOTO_LOSS_WEIGHT,
                "use_masked_patch_aux": USE_MASKED_PATCH_AUX,
                "masked_patch_ratio": MASKED_PATCH_RATIO,
                "masked_patch_size": MASKED_PATCH_SIZE,
                "masked_patch_loss_weight": MASKED_PATCH_LOSS_WEIGHT,
                "masked_patch_deriv_weight": MASKED_PATCH_DERIV_WEIGHT,
                "masked_patch_hidden_dim": MASKED_PATCH_HIDDEN_DIM,
                "masked_patch_mask_value": MASKED_PATCH_MASK_VALUE,
                "masked_patch_apply_to_seq2seq": MASKED_PATCH_APPLY_TO_SEQ2SEQ,
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
                "spectrum_noise_std": SPECTRUM_NOISE_STD,
                "canonicalize_smiles": CANONICALIZE_SMILES,
                "randomize_smiles": RANDOMIZE_SMILES,
                "randomize_prob": RANDOMIZE_PROB,
                "rebuild_vocab": REBUILD_VOCAB,
                "val_seq_em_every": VAL_SEQ_EM_EVERY,
                "val_seq_em_max_samples": VAL_SEQ_EM_MAX_SAMPLES,
                "num_workers": NUM_WORKERS,
                "pin_memory": PIN_MEMORY,
                "use_wandb": USE_WANDB,
                "wandb_project": WANDB_PROJECT,
                "wandb_entity": WANDB_ENTITY,
                "wandb_run_name": WANDB_RUN_NAME,
                "use_reduce_lr": USE_REDUCE_LR,
                "plateau_factor": PLATEAU_FACTOR,
                "plateau_patience": PLATEAU_PATIENCE,
                "plateau_min_lr": PLATEAU_MIN_LR,
                "plateau_threshold": PLATEAU_THRESHOLD,
                "use_diffusion_align": USE_DIFFUSION_ALIGN,
                "diffusion_ckpt_path": DIFFUSION_CKPT_PATH if USE_DIFFUSION_ALIGN else None,
                "diffusion_loss_weight": DIFFUSION_LOSS_WEIGHT,
                "diffusion_tau": DIFFUSION_TAU,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )
    tqdm.write(f"Saved config to {CONFIG_PATH}")

    best_loss = float("inf")
    best_seq_em = -1.0
    no_improve_seq_em_epochs = 0
    last_seq_em = 0.0

    for epoch in range(TRAIN_EPOCH):
        model.train()
        if masked_patch_head is not None:
            masked_patch_head.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_tanimoto_aux_loss = 0.0
        total_soft_tanimoto = 0.0
        total_masked_patch_loss = 0.0
        total_masked_patch_huber = 0.0
        total_masked_patch_deriv = 0.0
        total_correct = 0
        total_tokens = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{TRAIN_EPOCH} [train]", leave=False)
        for spectra, formula_vec, smiles_ids in train_pbar:
            spectra = spectra.to(DEVICE)
            clean_spectra = spectra
            encoder_input_spectra = clean_spectra
            recon_input_spectra = None
            patch_mask = None
            masked_patch_loss = clean_spectra.new_zeros(())
            masked_patch_huber = clean_spectra.new_zeros(())
            masked_patch_deriv = clean_spectra.new_zeros(())

            if USE_MASKED_PATCH_AUX and masked_patch_head is not None:
                masked_spectra, patch_mask = apply_masked_patch_augmentation(
                    clean_spectra,
                    mask_ratio=MASKED_PATCH_RATIO,
                    patch_size=MASKED_PATCH_SIZE,
                    mask_value=MASKED_PATCH_MASK_VALUE,
                )
                recon_input_spectra = masked_spectra
                if MASKED_PATCH_APPLY_TO_SEQ2SEQ:
                    encoder_input_spectra = masked_spectra
            if SPECTRUM_NOISE_STD > 0:
                encoder_input_spectra = encoder_input_spectra + torch.randn_like(encoder_input_spectra) * SPECTRUM_NOISE_STD
                if recon_input_spectra is not None and not MASKED_PATCH_APPLY_TO_SEQ2SEQ:
                    recon_input_spectra = recon_input_spectra + torch.randn_like(recon_input_spectra) * SPECTRUM_NOISE_STD
            formula_vec = formula_vec.to(DEVICE)
            smiles_ids = smiles_ids.to(DEVICE)

            optimizer.zero_grad()
            decoder_input = smiles_ids[:, :-1]
            target = smiles_ids[:, 1:]
            if USE_MASKED_PATCH_AUX and masked_patch_head is not None:
                memory = model.encoder(encoder_input_spectra, formula_vec)
                logits = model.decoder(memory, decoder_input)
                recon_memory = memory
                if not MASKED_PATCH_APPLY_TO_SEQ2SEQ:
                    recon_memory = model.encoder(recon_input_spectra, formula_vec)
                recon_spectra = masked_patch_head(recon_memory)
                masked_patch_loss, masked_patch_huber, masked_patch_deriv = masked_patch_reconstruction_loss(
                    recon_spectra,
                    clean_spectra,
                    patch_mask,
                    deriv_weight=MASKED_PATCH_DERIV_WEIGHT,
                )
            else:
                outputs = model(encoder_input_spectra, formula_vec, decoder_input)
                logits = outputs
            ce_loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            if USE_TANIMOTO_AUX_LOSS and TANIMOTO_LOSS_WEIGHT > 0:
                tanimoto_aux_loss, soft_tanimoto = soft_token_tanimoto_loss(
                    logits,
                    target,
                    pad_id=pad_id,
                    ignore_token_ids=tanimoto_ignore_ids,
                )
            else:
                tanimoto_aux_loss = torch.zeros((), device=logits.device)
                soft_tanimoto = torch.zeros((), device=logits.device)

            diff_loss = None
            if USE_DIFFUSION_ALIGN and diffusion_bundle is not None:
                with torch.no_grad():
                    z0 = diffusion_bundle.autoencoder.encode(clean_spectra)
                t = torch.randint(0, diffusion_bundle.schedule_steps, (z0.size(0),), device=DEVICE)
                noise = torch.randn_like(z0)
                z_t = q_sample(z0, t, noise, diffusion_schedule)

                tau = max(float(DIFFUSION_TAU), 1e-4)
                probs = F.softmax(logits / tau, dim=-1)
                soft_emb = torch.matmul(probs, diffusion_bundle.condition_encoder.token_emb.weight)
                pad_mask = target.eq(pad_id)

                cond_mem, cond_mask = diffusion_bundle.condition_encoder.encode_soft(
                    soft_emb,
                    pad_mask=pad_mask,
                    formula_vec=formula_vec,
                )
                pred_noise = diffusion_bundle.diffusion(
                    z_t,
                    t,
                    cond_memory=cond_mem,
                    cond_pad_mask=cond_mask,
                    cond_drop_prob=0.0,
                )
                diff_loss = F.mse_loss(pred_noise, noise)

            loss = ce_loss + TANIMOTO_LOSS_WEIGHT * tanimoto_aux_loss
            if USE_MASKED_PATCH_AUX and masked_patch_head is not None:
                loss = loss + MASKED_PATCH_LOSS_WEIGHT * masked_patch_loss
            if diff_loss is not None:
                loss = loss + DIFFUSION_LOSS_WEIGHT * diff_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if masked_patch_head is not None:
                nn.utils.clip_grad_norm_(masked_patch_head.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_tanimoto_aux_loss += tanimoto_aux_loss.item()
            total_soft_tanimoto += soft_tanimoto.item()
            total_masked_patch_loss += masked_patch_loss.item()
            total_masked_patch_huber += masked_patch_huber.item()
            total_masked_patch_deriv += masked_patch_deriv.item()
            c, t = token_accuracy(logits, target, pad_id=pad_id)
            total_correct += c
            total_tokens += t
            if diff_loss is None:
                train_pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    ce=f"{ce_loss.item():.4f}",
                    tan=f"{soft_tanimoto.item():.4f}",
                    mpatch=f"{masked_patch_loss.item():.4f}",
                )
            else:
                train_pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    ce=f"{ce_loss.item():.4f}",
                    tan=f"{soft_tanimoto.item():.4f}",
                    mpatch=f"{masked_patch_loss.item():.4f}",
                    diff=f"{diff_loss.item():.4f}",
                )

        train_loss = total_loss / max(len(train_loader), 1)
        train_ce_loss = total_ce_loss / max(len(train_loader), 1)
        train_tanimoto_aux_loss = total_tanimoto_aux_loss / max(len(train_loader), 1)
        train_soft_tanimoto = total_soft_tanimoto / max(len(train_loader), 1)
        train_masked_patch_loss = total_masked_patch_loss / max(len(train_loader), 1)
        train_masked_patch_huber = total_masked_patch_huber / max(len(train_loader), 1)
        train_masked_patch_deriv = total_masked_patch_deriv / max(len(train_loader), 1)
        train_acc = total_correct / max(total_tokens, 1)
        val_loss, val_acc, val_ce_loss, val_tanimoto_aux_loss, val_soft_tanimoto = validate(
            model,
            val_loader,
            criterion,
            DEVICE,
            pad_id,
            tanimoto_ignore_ids=tanimoto_ignore_ids,
        )
        compute_seq_em = (epoch == 0) or ((epoch + 1) % VAL_SEQ_EM_EVERY == 0)
        if compute_seq_em:
            val_seq_em = evaluate_val_sequence_exact_match(
                model=model,
                loader=val_loader,
                device=DEVICE,
                pad_id=pad_id,
                sos_id=sos_id,
                eos_id=eos_id,
                max_len=VAL_SEQ_EM_MAX_LEN,
                max_samples=VAL_SEQ_EM_MAX_SAMPLES,
            )
            last_seq_em = val_seq_em
        else:
            val_seq_em = last_seq_em
        if USE_REDUCE_LR:
            scheduler.step(val_loss)
        else:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        tqdm.write(
            f"Epoch {epoch + 1}/{TRAIN_EPOCH} | "
            f"lr={lr_now:.2e} "
            f"train_loss={train_loss:.4f} train_ce={train_ce_loss:.4f} "
            f"train_tan_loss={train_tanimoto_aux_loss:.4f} train_tan={train_soft_tanimoto:.4f} "
            f"train_mpatch={train_masked_patch_loss:.4f} "
            f"train_acc={train_acc:.4%} | "
            f"val_loss={val_loss:.4f} val_ce={val_ce_loss:.4f} "
            f"val_tan_loss={val_tanimoto_aux_loss:.4f} val_tan={val_soft_tanimoto:.4f} "
            f"val_acc={val_acc:.4%} val_seq_em={val_seq_em:.4%}"
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "lr": lr_now,
                    "train_loss": train_loss,
                    "train_ce_loss": train_ce_loss,
                    "train_tanimoto_aux_loss": train_tanimoto_aux_loss,
                    "train_soft_tanimoto": train_soft_tanimoto,
                    "train_masked_patch_loss": train_masked_patch_loss,
                    "train_masked_patch_huber": train_masked_patch_huber,
                    "train_masked_patch_deriv": train_masked_patch_deriv,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_ce_loss": val_ce_loss,
                    "val_tanimoto_aux_loss": val_tanimoto_aux_loss,
                    "val_soft_tanimoto": val_soft_tanimoto,
                    "val_acc": val_acc,
                    "val_seq_em": val_seq_em,
                }
            )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_LOSS_MODEL_PATH)
            tqdm.write(f"Saved best-loss model to {BEST_LOSS_MODEL_PATH}")

        if compute_seq_em:
            if val_seq_em > best_seq_em + EARLY_STOP_MIN_DELTA:
                best_seq_em = val_seq_em
                no_improve_seq_em_epochs = 0
                torch.save(model.state_dict(), MODEL_PATH)
                tqdm.write(f"Saved best-seq-em model to {MODEL_PATH}")
            else:
                no_improve_seq_em_epochs += 1
                if no_improve_seq_em_epochs >= EARLY_STOP_PATIENCE:
                    tqdm.write(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(best val_seq_em={best_seq_em:.4%}, patience={EARLY_STOP_PATIENCE})"
                    )
                    break

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    train_model()
