import argparse
import json
import random
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from GRUmodel import FusionEncoder, IREncoder, IR2SMILES, SMILESDecoder
from util.dataloader import IRDataset, collate_fn


DEFAULT_MODEL_PATH = "checkpoints/GRUModel/best_model.pth"
DEFAULT_DATA_PATH = "data/raw_processed_data.pt"
DEFAULT_VOCAB_PATH = "pure/vocab.json"
DEFAULT_SPLIT_PATH = "checkpoints/data_split.pt"


def load_vocab(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    token_to_id = {k: int(v) for k, v in token_to_id.items()}
    id_to_token = {v: k for k, v in token_to_id.items()}
    return token_to_id, id_to_token


def load_split_indices(split_path: str, split_name: str):
    split_data = torch.load(split_path, map_location="cpu")
    key = f"{split_name}_indices"
    if key not in split_data:
        raise KeyError(f"{key} not found in {split_path}")
    return split_data[key]


def build_model(model_path: str, data_path: str, vocab_size: int, formula_dim: int, device: torch.device) -> IR2SMILES:
    raw_data = torch.load(data_path, map_location="cpu")
    input_points = raw_data["ir_spectra"].shape[1]

    ir_encoder = IREncoder(input_points=input_points, latent_dim=512)
    encoder = FusionEncoder(ir_encoder=ir_encoder, formula_dim=formula_dim, latent_dim=512)
    decoder = SMILESDecoder(vocab_size=vocab_size, embedding_dim=256, hidden_dim=512, latent_dim=512)
    model = IR2SMILES(encoder, decoder)

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def decode_ids(ids: List[int], id_to_token: Dict[int, str]) -> str:
    out = []
    for tid in ids:
        token = id_to_token.get(int(tid), "<UNK>")
        if token == "<EOS>":
            break
        if token in {"<SOS>", "<PAD>", "<UNK>"}:
            continue
        out.append(token)
    return "".join(out)


def parse_smiles_element_counts(smiles: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    i = 0
    n = len(smiles)

    def add_elem(elem: str):
        if elem in {"c", "n", "o", "p", "s", "b"}:
            elem = elem.upper()
        counts[elem] = counts.get(elem, 0) + 1

    while i < n:
        ch = smiles[i]

        if ch == "[":
            j = smiles.find("]", i + 1)
            if j == -1:
                break
            bracket_content = smiles[i + 1 : j]
            # Example: [13C@@H], [nH+], [Si], [O-]
            m = re.search(r"([A-Z][a-z]?|[cnospb])", bracket_content)
            if m:
                add_elem(m.group(1))
            i = j + 1
            continue

        if i + 1 < n and smiles[i : i + 2] in {"Br", "Cl"}:
            add_elem(smiles[i : i + 2])
            i += 2
            continue

        if ch.isupper():
            if i + 1 < n and smiles[i + 1].islower():
                add_elem(smiles[i : i + 2])
                i += 2
            else:
                add_elem(ch)
                i += 1
            continue

        if ch in {"c", "n", "o", "p", "s", "b"}:
            add_elem(ch)
            i += 1
            continue

        i += 1

    return counts


def formula_vec_to_counts(formula_vec: torch.Tensor, idx_to_elem: Dict[int, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    vec = formula_vec.detach().cpu().tolist()
    for idx, value in enumerate(vec):
        if value > 0:
            counts[idx_to_elem[idx]] = int(round(float(value)))
    return counts


def greedy_decode_batch(model: IR2SMILES, spectra: torch.Tensor, formula_vec: torch.Tensor, max_len: int, sos_id: int) -> torch.Tensor:
    latent = model.encoder(spectra, formula_vec)
    h = model.decoder.latent_to_h0(latent).unsqueeze(0)

    batch_size = spectra.size(0)
    input_token = torch.full((batch_size,), sos_id, dtype=torch.long, device=spectra.device)
    generated = []

    for _ in range(max_len):
        embedded = model.decoder.embedding(input_token).unsqueeze(1)
        output, h = model.decoder.gru(embedded, h)
        logits = model.decoder.fc_out(output.squeeze(1))
        input_token = torch.argmax(logits, dim=1)
        generated.append(input_token)

    if not generated:
        return torch.empty((batch_size, 0), dtype=torch.long, device=spectra.device)
    return torch.stack(generated, dim=1)


def _length_penalized_score(score: float, length: int, alpha: float = 0.7) -> float:
    # Google NMT style length penalty to avoid over-preferring short sequences
    lp = ((5.0 + max(length, 1)) / 6.0) ** alpha
    return score / lp


def beam_search_single(
    model: IR2SMILES,
    spectrum: torch.Tensor,
    formula_vec: torch.Tensor,
    beam_size: int,
    max_len: int,
    sos_id: int,
    eos_id: int,
    alpha: float = 0.7,
) -> List[List[int]]:
    """
    Returns top beam token sequences (including <SOS>, maybe <EOS>).
    """
    device = spectrum.device
    with torch.no_grad():
        latent = model.encoder(spectrum.unsqueeze(0), formula_vec.unsqueeze(0))
        h0 = model.decoder.latent_to_h0(latent).unsqueeze(0)  # [1, 1, H]

        beams: List[Tuple[List[int], float, torch.Tensor, bool]] = [([sos_id], 0.0, h0, False)]

        for _ in range(max_len):
            new_beams: List[Tuple[List[int], float, torch.Tensor, bool]] = []
            all_finished = True

            for seq, score, h, finished in beams:
                if finished:
                    new_beams.append((seq, score, h, True))
                    continue

                all_finished = False
                last_token = torch.tensor([seq[-1]], dtype=torch.long, device=device)
                embedded = model.decoder.embedding(last_token).unsqueeze(1)
                output, h_new = model.decoder.gru(embedded, h)
                logits = model.decoder.fc_out(output.squeeze(1))
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                k = min(beam_size, log_probs.numel())
                topv, topi = torch.topk(log_probs, k=k, dim=-1)
                for logp, token_id in zip(topv.tolist(), topi.tolist()):
                    new_seq = seq + [int(token_id)]
                    is_finished = int(token_id) == eos_id
                    new_beams.append((new_seq, score + float(logp), h_new.clone(), is_finished))

            new_beams.sort(
                key=lambda x: _length_penalized_score(x[1], len(x[0]) - 1, alpha=alpha),
                reverse=True,
            )
            beams = new_beams[:beam_size]

            if all_finished:
                break

        beams.sort(
            key=lambda x: _length_penalized_score(x[1], len(x[0]) - 1, alpha=alpha),
            reverse=True,
        )
        return [seq for seq, _, _, _ in beams]


def evaluate_teacher_forcing(
    model: IR2SMILES,
    loader: DataLoader,
    device: torch.device,
    pad_id: int = 0,
    topk: int = 5,
) -> Tuple[float, float, float, float]:
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_loss = 0.0
    total_batches = 0
    total_tokens = 0
    correct_tokens = 0
    topk_correct_tokens = 0

    with torch.no_grad():
        for spectra, formula_vec, smiles_ids in loader:
            spectra = spectra.to(device)
            formula_vec = formula_vec.to(device)
            smiles_ids = smiles_ids.to(device)

            outputs = model(spectra, formula_vec, smiles_ids)
            logits = outputs[:, 1:, :]
            target = smiles_ids[:, 1:]

            loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            total_loss += loss.item()
            total_batches += 1

            pred = torch.argmax(logits, dim=-1)
            valid_mask = target != pad_id
            correct_tokens += ((pred == target) & valid_mask).sum().item()
            total_tokens += valid_mask.sum().item()

            k = min(topk, logits.shape[-1])
            topk_idx = torch.topk(logits, k=k, dim=-1).indices
            in_topk = topk_idx.eq(target.unsqueeze(-1)).any(dim=-1)
            topk_correct_tokens += (in_topk & valid_mask).sum().item()

    avg_loss = total_loss / max(total_batches, 1)
    perplexity = float(torch.exp(torch.tensor(avg_loss)).item())
    token_acc = correct_tokens / max(total_tokens, 1)
    topk_acc = topk_correct_tokens / max(total_tokens, 1)
    return avg_loss, perplexity, token_acc, topk_acc


def evaluate_generation(
    model: IR2SMILES,
    dataset: Dataset,
    id_to_token: Dict[int, str],
    idx_to_elem: Dict[int, str],
    device: torch.device,
    sample_count: int,
    batch_size: int,
    max_len: int,
    sos_id: int,
    eos_id: int,
    beam_size: int,
    seed: int,
) -> Tuple[float, List[Tuple[str, str]], Dict[str, float]]:
    rng = random.Random(seed)
    total = len(dataset)
    sample_count = min(sample_count, total)
    indices = list(range(total))
    rng.shuffle(indices)
    indices = indices[:sample_count]

    pairs: List[Tuple[str, str]] = []
    exact_match = 0
    elem_set_match = 0
    elem_count_match = 0
    elem_abs_err = 0
    elem_total = 0
    beam_topk_exact_match = 0

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            spectra_batch = []
            formula_batch = []
            target_ids_batch = []
            for idx in batch_indices:
                spectra, formula_vec, target_ids = dataset[idx]
                spectra_batch.append(spectra)
                formula_batch.append(formula_vec)
                target_ids_batch.append(target_ids)

            spectra_tensor = torch.stack(spectra_batch, dim=0).to(device)
            formula_tensor = torch.stack(formula_batch, dim=0).to(device)
            pred_ids = greedy_decode_batch(model, spectra_tensor, formula_tensor, max_len=max_len, sos_id=sos_id)

            for i in range(len(batch_indices)):
                gt = decode_ids(target_ids_batch[i].tolist(), id_to_token)
                pred = decode_ids(pred_ids[i].tolist(), id_to_token)
                pairs.append((gt, pred))
                if gt == pred:
                    exact_match += 1

                beam_token_seqs = beam_search_single(
                    model=model,
                    spectrum=spectra_tensor[i],
                    formula_vec=formula_tensor[i],
                    beam_size=beam_size,
                    max_len=max_len,
                    sos_id=sos_id,
                    eos_id=eos_id,
                )
                beam_smiles = [decode_ids(seq, id_to_token) for seq in beam_token_seqs]
                if gt in beam_smiles:
                    beam_topk_exact_match += 1

                gt_elem_counts = formula_vec_to_counts(formula_batch[i], idx_to_elem)
                pred_elem_counts = parse_smiles_element_counts(pred)

                gt_set = {k for k, v in gt_elem_counts.items() if v > 0}
                pred_set = {k for k, v in pred_elem_counts.items() if v > 0}
                if gt_set == pred_set:
                    elem_set_match += 1

                union_elems = gt_set | pred_set
                all_equal = True
                for elem in union_elems:
                    gv = gt_elem_counts.get(elem, 0)
                    pv = pred_elem_counts.get(elem, 0)
                    if gv != pv:
                        all_equal = False
                    elem_abs_err += abs(gv - pv)
                    elem_total += 1
                if all_equal:
                    elem_count_match += 1

    em_rate = exact_match / max(len(pairs), 1)
    elem_metrics = {
        "element_set_match_rate": elem_set_match / max(len(pairs), 1),
        "element_count_exact_match_rate": elem_count_match / max(len(pairs), 1),
        "element_count_mae": elem_abs_err / max(elem_total, 1),
        "beam_topk_exact_match_rate": beam_topk_exact_match / max(len(pairs), 1),
    }
    return em_rate, pairs, elem_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained IR+Formula -> SMILES model.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--vocab-path", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--split-path", type=str, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--split", type=str, choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gen-samples", type=int, default=200)
    parser.add_argument("--print-samples", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=120)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_to_id, id_to_token = load_vocab(args.vocab_path)

    full_dataset = IRDataset(args.data_path, token_to_id)
    idx_to_elem = {v: k for k, v in full_dataset.formula_vocab.items()}

    if args.split == "all":
        eval_dataset = full_dataset
    else:
        split_indices = load_split_indices(args.split_path, args.split)
        eval_dataset = Subset(full_dataset, split_indices)

    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = build_model(
        model_path=args.model_path,
        data_path=args.data_path,
        vocab_size=len(token_to_id),
        formula_dim=full_dataset.formula_dim,
        device=device,
    )

    avg_loss, ppl, token_acc, topk_acc = evaluate_teacher_forcing(
        model,
        loader,
        device,
        pad_id=token_to_id.get("<PAD>", 0),
        topk=args.topk,
    )

    em_rate, pairs, elem_metrics = evaluate_generation(
        model=model,
        dataset=eval_dataset,
        id_to_token=id_to_token,
        idx_to_elem=idx_to_elem,
        device=device,
        sample_count=args.gen_samples,
        batch_size=args.batch_size,
        max_len=args.max_len,
        sos_id=token_to_id.get("<SOS>", 1),
        eos_id=token_to_id.get("<EOS>", 2),
        beam_size=args.beam_size,
        seed=args.seed,
    )

    print("=" * 70)
    print("Model evaluation finished")
    print(f"Device                 : {device}")
    print(f"Model checkpoint       : {args.model_path}")
    print(f"Evaluate split         : {args.split}")
    print(f"Dataset size           : {len(eval_dataset)}")
    print(f"Formula feature dim    : {full_dataset.formula_dim}")
    print("-" * 70)
    print(f"Teacher forcing loss   : {avg_loss:.6f}")
    print(f"Perplexity             : {ppl:.6f}")
    print(f"Token accuracy         : {token_acc:.4%}")
    print(f"Top-{args.topk} token acc     : {topk_acc:.4%}")
    print("-" * 70)
    print(f"Greedy exact-match@{len(pairs)} : {em_rate:.4%}")
    print(f"Beam top-{args.beam_size} exact  : {elem_metrics['beam_topk_exact_match_rate']:.4%}")
    print(f"Element set match      : {elem_metrics['element_set_match_rate']:.4%}")
    print(f"Element count exact    : {elem_metrics['element_count_exact_match_rate']:.4%}")
    print(f"Element count MAE      : {elem_metrics['element_count_mae']:.6f}")
    print("=" * 70)

    show_n = min(args.print_samples, len(pairs))
    if show_n > 0:
        print("\nSample predictions:")
        for i in range(show_n):
            gt, pred = pairs[i]
            print(f"[{i + 1}] GT   : {gt}")
            print(f"    Pred : {pred}")


if __name__ == "__main__":
    main()
