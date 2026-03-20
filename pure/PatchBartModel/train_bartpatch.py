import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]  # pure
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BARTPatchModel import BARTPatchModel
from util.dataloader import IRDataset, collate_fn
from util.tokenizer import build_vocab

TRAIN_EPOCH = 40
BATCH_SIZE = 64
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 2e-2
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "data/raw_processed_data.pt"
VOCAB_PATH = "pure/vocab.json"
SPLIT_PATH = "checkpoints/data_split.pt"
MODEL_PATH = "checkpoints/BartModel/best_bartpatch_model.pth"
BEST_LOSS_MODEL_PATH = "checkpoints/BartModel/best_bartpatch_loss_model.pth"
CONFIG_PATH = "checkpoints/BartModel/bartpatch_config.json"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42

# BARTPatch config
D_MODEL = 384
NHEAD = 8
ENC_LAYERS = 4
DEC_LAYERS = 4
DIM_FEEDFORWARD = 1536
DROPOUT = 0.2
PATCH_SIZE = 25
USE_NONUNIFORM_PATCH = True
# For 650~3900 cm^-1 with 1625 points:
# step ~= (3900-650)/(1625-1) ~= 2.0012 cm^-1 per point
# fingerprint 650~1300 cm^-1 -> index range about [0, 325], end-exclusive -> 326
FINGERPRINT_START_IDX = 0
FINGERPRINT_END_IDX = 326
FINGERPRINT_PATCH_SIZE = 10
NON_FINGERPRINT_PATCH_SIZE = 32
NONUNIFORM_PATCH_TOKEN_LEN = 16
MAX_TGT_LEN = 256
VAL_SEQ_EM_MAX_SAMPLES = 0
LABEL_SMOOTHING = 0.0
EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_DELTA = 1e-4
SPECTRUM_NOISE_STD = 0.00



def get_tokenizer():
    try:
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
    except Exception:
        tqdm.write("vocab.json not found, building vocab...")
        build_vocab(DATA_PATH)
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
    return {k: int(v) for k, v in token_to_id.items()}


def get_or_create_splits(total_size: int, split_path: str):
    if os.path.exists(split_path):
        split_data = torch.load(split_path, map_location="cpu")
        return split_data["train_indices"], split_data["val_indices"], split_data["test_indices"]

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
    return train_idx, val_idx, test_idx


def token_accuracy(logits: torch.Tensor, target: torch.Tensor, pad_id: int = 0):
    pred = torch.argmax(logits, dim=-1)
    valid_mask = target != pad_id
    correct = ((pred == target) & valid_mask).sum().item()
    total = valid_mask.sum().item()
    return correct, total


def _normalize_token_seq(ids: torch.Tensor, pad_id: int, sos_id: int, eos_id: int):
    seq = []
    for tid in ids.tolist():
        if tid == eos_id:
            break
        if tid in {pad_id, sos_id}:
            continue
        seq.append(int(tid))
    return tuple(seq)


def validate(model, loader, criterion, device, pad_id: int):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for spectra, formula_vec, smiles_ids in loader:
            spectra = spectra.to(device)
            formula_vec = formula_vec.to(device)
            smiles_ids = smiles_ids.to(device)

            decoder_input = smiles_ids[:, :-1]
            target = smiles_ids[:, 1:]
            logits = model(spectra, formula_vec, decoder_input)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

            total_loss += loss.item()
            c, t = token_accuracy(logits, target, pad_id=pad_id)
            total_correct += c
            total_tokens += t

    return total_loss / max(len(loader), 1), total_correct / max(total_tokens, 1)


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
            generated_ids = model.generate(spectra, formula_vec, max_len=max_len, sos_id=sos_id, eos_id=eos_id).cpu()
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
    tokenizer = get_tokenizer()
    pad_id = tokenizer.get("<PAD>", 0)
    sos_id = tokenizer.get("<SOS>", 1)
    eos_id = tokenizer.get("<EOS>", 2)

    dataset = IRDataset(DATA_PATH, tokenizer)
    total_size = len(dataset)
    train_idx, val_idx, test_idx = get_or_create_splits(total_size, SPLIT_PATH)

    tqdm.write(f"Dataset size: {total_size}")
    tqdm.write(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    tqdm.write(f"Formula feature dim: {dataset.formula_dim}")

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = BARTPatchModel(
        vocab_size=len(tokenizer),
        formula_dim=dataset.formula_dim,
        input_points=dataset[0][0].numel(),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=ENC_LAYERS,
        num_decoder_layers=DEC_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        patch_size=PATCH_SIZE,
        use_nonuniform_patch=USE_NONUNIFORM_PATCH,
        fingerprint_start_idx=FINGERPRINT_START_IDX,
        fingerprint_end_idx=FINGERPRINT_END_IDX,
        fingerprint_patch_size=FINGERPRINT_PATCH_SIZE,
        non_fingerprint_patch_size=NON_FINGERPRINT_PATCH_SIZE,
        nonuniform_patch_token_len=NONUNIFORM_PATCH_TOKEN_LEN,
        max_tgt_len=MAX_TGT_LEN,
        pad_id=pad_id,
        sos_id=sos_id,
        eos_id=eos_id,
    ).to(DEVICE)
    if model.use_nonuniform_patch:
        tqdm.write(
            "Non-uniform patch enabled | "
            f"fingerprint[{model.fingerprint_start_idx}:{model.fingerprint_end_idx}) "
            f"fp_patch={model.fingerprint_patch_size} non_fp_patch={model.non_fingerprint_patch_size} "
            f"patch_tokens={len(model.patch_spans)}"
        )
    else:
        tqdm.write(f"Uniform patch enabled | patch_size={PATCH_SIZE}")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
                "num_encoder_layers": ENC_LAYERS,
                "num_decoder_layers": DEC_LAYERS,
                "dim_feedforward": DIM_FEEDFORWARD,
                "dropout": DROPOUT,
                "patch_size": PATCH_SIZE,
                "use_nonuniform_patch": USE_NONUNIFORM_PATCH,
                "fingerprint_start_idx": FINGERPRINT_START_IDX,
                "fingerprint_end_idx": FINGERPRINT_END_IDX,
                "fingerprint_patch_size": FINGERPRINT_PATCH_SIZE,
                "non_fingerprint_patch_size": NON_FINGERPRINT_PATCH_SIZE,
                "nonuniform_patch_token_len": NONUNIFORM_PATCH_TOKEN_LEN,
                "max_tgt_len": MAX_TGT_LEN,
                "pad_id": pad_id,
                "sos_id": sos_id,
                "eos_id": eos_id,
                "label_smoothing": LABEL_SMOOTHING,
                "spectrum_noise_std": SPECTRUM_NOISE_STD,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )
    tqdm.write(f"Saved config to {CONFIG_PATH}")

    best_loss = float("inf")
    best_seq_em = -1.0
    no_improve_epochs = 0

    for epoch in range(TRAIN_EPOCH):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{TRAIN_EPOCH} [train]", leave=False)
        for spectra, formula_vec, smiles_ids in pbar:
            spectra = spectra.to(DEVICE)
            if SPECTRUM_NOISE_STD > 0:
                spectra = spectra + torch.randn_like(spectra) * SPECTRUM_NOISE_STD
            formula_vec = formula_vec.to(DEVICE)
            smiles_ids = smiles_ids.to(DEVICE)

            optimizer.zero_grad()
            decoder_input = smiles_ids[:, :-1]
            target = smiles_ids[:, 1:]
            logits = model(spectra, formula_vec, decoder_input)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            c, t = token_accuracy(logits, target, pad_id=pad_id)
            total_correct += c
            total_tokens += t
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_loss = total_loss / max(len(train_loader), 1)
        train_acc = total_correct / max(total_tokens, 1)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, pad_id)
        val_seq_em = evaluate_val_sequence_exact_match(
            model=model,
            loader=val_loader,
            device=DEVICE,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            max_len=MAX_TGT_LEN,
            max_samples=VAL_SEQ_EM_MAX_SAMPLES,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        tqdm.write(
            f"Epoch {epoch + 1}/{TRAIN_EPOCH} | "
            f"lr={lr_now:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4%} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4%} val_seq_em={val_seq_em:.4%}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_LOSS_MODEL_PATH)
            tqdm.write(f"Saved best-loss model to {BEST_LOSS_MODEL_PATH}")

        if val_seq_em > best_seq_em + EARLY_STOP_MIN_DELTA:
            best_seq_em = val_seq_em
            no_improve_epochs = 0
            torch.save(model.state_dict(), MODEL_PATH)
            tqdm.write(f"Saved best-seq-em model to {MODEL_PATH}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                tqdm.write(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best val_seq_em={best_seq_em:.4%}, patience={EARLY_STOP_PATIENCE})"
                )
                break


if __name__ == "__main__":
    train_model()
