import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from GRUmodel import FusionEncoder, IREncoder, IR2SMILES, SMILESDecoder
from util.dataloader import IRDataset, collate_fn
from util.tokenizer import build_vocab

MODEL = "GRU"
TRAIN_EPOCH = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_FUNC = nn.CrossEntropyLoss(ignore_index=0)
INPUT_POINTS = 1652
LATENT_DIM = 512
HIDDEN_DIM = 512
EMBED_DIM = 256
DATA_PATH = "data/raw_processed_data.pt"
VOCAB_PATH = "pure/vocab.json"
SPLIT_PATH = "checkpoints/data_split.pt"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42
DEFAULT_MODEL_PATH = "pure/GRUModel/best_model.pth"

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


def build_model(formula_dim: int, vocab_size: int):
    if MODEL != "GRU":
        raise ValueError(f"Unsupported model type: {MODEL}")
    tqdm.write("Using GRU model (IR + molecular_formula fusion)")

    ir_encoder = IREncoder(input_points=INPUT_POINTS, latent_dim=LATENT_DIM)
    encoder = FusionEncoder(ir_encoder=ir_encoder, formula_dim=formula_dim, latent_dim=LATENT_DIM)
    decoder = SMILESDecoder(
        vocab_size=vocab_size,
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
    )
    return IR2SMILES(encoder, decoder).to(DEVICE)


def token_accuracy(logits: torch.Tensor, target: torch.Tensor, pad_id: int = 0):
    pred = torch.argmax(logits, dim=-1)
    valid_mask = target != pad_id
    correct = ((pred == target) & valid_mask).sum().item()
    total = valid_mask.sum().item()
    return correct, total


def trian_model():
    tokenizer = get_tokenizer()
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

    model = build_model(formula_dim=dataset.formula_dim, vocab_size=len(tokenizer))
    criterion = LOSS_FUNC
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float("inf")

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(TRAIN_EPOCH):
        model.train()
        total_loss = 0.0
        train_correct_tokens = 0
        train_total_tokens = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{TRAIN_EPOCH} [train]", leave=False)
        for spectra, formula_vec, smiles_ids in train_pbar:
            spectra = spectra.to(DEVICE)
            formula_vec = formula_vec.to(DEVICE)
            smiles_ids = smiles_ids.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(spectra, formula_vec, smiles_ids)
            logits = outputs[:, 1:, :]
            target = smiles_ids[:, 1:]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            c, t = token_accuracy(logits, target, pad_id=0)
            train_correct_tokens += c
            train_total_tokens += t

            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        avg_train_loss = total_loss / max(len(train_loader), 1)
        train_acc = train_correct_tokens / max(train_total_tokens, 1)

        tqdm.write(
            f"Epoch {epoch + 1}/{TRAIN_EPOCH} | "
            f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.4%} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4%}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), DEFAULT_MODEL_PATH)
            tqdm.write("Saved best model to checkpoints/best_model.pth")


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct_tokens = 0
    val_total_tokens = 0

    with torch.no_grad():
        val_pbar = tqdm(loader, desc="Validate", leave=False)
        for spectra, formula_vec, smiles_ids in val_pbar:
            spectra = spectra.to(device)
            formula_vec = formula_vec.to(device)
            smiles_ids = smiles_ids.to(device)

            outputs = model(spectra, formula_vec, smiles_ids)
            logits = outputs[:, 1:, :]
            target = smiles_ids[:, 1:]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

            val_loss += loss.item()
            c, t = token_accuracy(logits, target, pad_id=0)
            val_correct_tokens += c
            val_total_tokens += t

            val_pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_val_loss = val_loss / max(len(loader), 1)
    val_acc = val_correct_tokens / max(val_total_tokens, 1)
    return avg_val_loss, val_acc


if __name__ == "__main__":
    trian_model()
