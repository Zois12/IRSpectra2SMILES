import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]  # pure
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BARTPatchModel import BARTPatchModel
from SMILES2IRModel.SMILES2IRModel import SMILES2IRRegressor
from util.dataloader import IRDataset, collate_fn


DEFAULT_MODEL_PATH = "checkpoints/BartModel/best_bartpatch_model.pth"
DEFAULT_CONFIG_PATH = "checkpoints/BartModel/bartpatch_config.json"
DEFAULT_DATA_PATH = "data/raw_processed_data.pt"
DEFAULT_VOCAB_PATH = "pure/vocab.json"
DEFAULT_SPLIT_PATH = "checkpoints/data_split.pt"
DEFAULT_SMILES2IR_MODEL_PATH = "checkpoints/BartModel/best_smiles2ir_model.pth"
DEFAULT_SMILES2IR_CONFIG_PATH = "checkpoints/Smiles2IRModel/smiles2ir_config.json"

SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|C|N|O|P|S|F|I|b|n|o|s|p|c|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def load_vocab(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    token_to_id = {k: int(v) for k, v in token_to_id.items()}
    id_to_token = {v: k for k, v in token_to_id.items()}
    return token_to_id, id_to_token


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split_indices(split_path: str, split_name: str):
    split_data = torch.load(split_path, map_location="cpu")
    key = f"{split_name}_indices"
    if key not in split_data:
        raise KeyError(f"{key} not found in {split_path}")
    return split_data[key]


def build_model(model_path: str, config: Dict, device: torch.device) -> BARTPatchModel:
    model = BARTPatchModel(
        vocab_size=config["vocab_size"],
        formula_dim=config["formula_dim"],
        input_points=config["input_points"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        patch_size=config["patch_size"],
        use_nonuniform_patch=config.get("use_nonuniform_patch", False),
        fingerprint_start_idx=config.get("fingerprint_start_idx"),
        fingerprint_end_idx=config.get("fingerprint_end_idx"),
        fingerprint_patch_size=config.get("fingerprint_patch_size"),
        non_fingerprint_patch_size=config.get("non_fingerprint_patch_size"),
        nonuniform_patch_token_len=config.get("nonuniform_patch_token_len", 16),
        max_tgt_len=config["max_tgt_len"],
        pad_id=config["pad_id"],
        sos_id=config["sos_id"],
        eos_id=config["eos_id"],
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def build_smiles2ir_model(
    model_path: str,
    config: Dict,
    device: torch.device,
    fallback_formula_dim: Optional[int] = None,
) -> SMILES2IRRegressor:
    formula_dim = config.get("formula_dim")
    if formula_dim is None and fallback_formula_dim is not None:
        formula_dim = int(fallback_formula_dim)
    if formula_dim is None:
        formula_dim = 0
    model = SMILES2IRRegressor(
        vocab_size=config["vocab_size"],
        output_points=config["output_points"],
        formula_dim=formula_dim,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        pad_id=config["pad_id"],
        max_len=config["max_len"],
        spectral_decoder_layers=config.get("spectral_decoder_layers", 3),
        decoder_query_len=config.get("decoder_query_len", 325),
        refine_channels=config.get("refine_channels", 64),
    )
    state_dict = torch.load(model_path, map_location="cpu")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        print(
            "Warning: SMILES2IR checkpoint mismatch; loaded with strict=False. "
            f"missing_keys={missing} unexpected_keys={unexpected}"
        )
        print(f"Original load error: {exc}")
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


def encode_smiles_text(smiles: str, token_to_id: Dict[str, int], max_len: int) -> torch.Tensor:
    sos_id = token_to_id.get("<SOS>", 1)
    eos_id = token_to_id.get("<EOS>", 2)
    unk_id = token_to_id.get("<UNK>", 3)

    ids = [sos_id]
    for token in SMILES_TOKEN_PATTERN.findall(smiles):
        ids.append(token_to_id.get(token, unk_id))
    ids.append(eos_id)

    if max_len > 0:
        ids = ids[:max_len]
    if not ids:
        ids = [sos_id, eos_id]

    return torch.tensor(ids, dtype=torch.long)


def rerank_candidates_by_ir_similarity(
    smiles_candidates: List[str],
    target_spectrum: torch.Tensor,
    smiles2ir_model: SMILES2IRRegressor,
    token_to_id: Dict[str, int],
    device: torch.device,
    max_smiles_len: int,
    similarity: str = "cosine",
) -> List[Tuple[str, float]]:
    unique_candidates: List[str] = []
    seen = set()
    for smiles in smiles_candidates:
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)
        unique_candidates.append(smiles)

    if not unique_candidates:
        return []

    encoded = [encode_smiles_text(s, token_to_id=token_to_id, max_len=max_smiles_len) for s in unique_candidates]
    max_batch_len = max(seq.size(0) for seq in encoded)
    pad_id = token_to_id.get("<PAD>", 0)
    padded = []
    for seq in encoded:
        if seq.size(0) < max_batch_len:
            pad = torch.full((max_batch_len - seq.size(0),), pad_id, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=0)
        padded.append(seq)
    smiles_ids = torch.stack(padded, dim=0).to(device)

    with torch.no_grad():
        pred_spectra = smiles2ir_model.predict_from_smiles_ids(smiles_ids)
        target = target_spectrum.unsqueeze(0).expand(pred_spectra.size(0), -1)
        if similarity == "cosine":
            scores = F.cosine_similarity(pred_spectra, target, dim=-1)
        elif similarity == "neg_mse":
            scores = -F.mse_loss(pred_spectra, target, reduction="none").mean(dim=-1)
        else:
            raise ValueError(f"Unsupported similarity: {similarity}")

    ranked = list(zip(unique_candidates, scores.detach().cpu().tolist()))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def choose_rerank_candidate(
    beam_candidates: List[Tuple[str, float]],
    ir_ranked: List[Tuple[str, float]],
    mode: str = "conservative",
    ir_weight: float = 0.25,
    min_abs_similarity: float = 0.80,
    min_delta_similarity: float = 0.03,
) -> Tuple[str, float, bool]:
    if not beam_candidates:
        return "", 0.0, False

    beam_top1 = beam_candidates[0][0]
    if not ir_ranked:
        return beam_top1, 0.0, False

    ir_score_by_smiles = {s: float(v) for s, v in ir_ranked}
    ir_rank = {s: rank for rank, (s, _) in enumerate(ir_ranked)}
    beam_rank = {s: rank for rank, (s, _) in enumerate(beam_candidates)}

    if mode == "ir_only":
        best_smiles, best_score = ir_ranked[0]
        return best_smiles, float(best_score), best_smiles != beam_top1

    if mode == "hybrid":
        best_smiles = beam_top1
        best_score = float("-inf")
        for smiles, _beam_score in beam_candidates:
            b_rank = beam_rank.get(smiles, 10**6)
            i_rank = ir_rank.get(smiles, 10**6)
            score = 1.0 / (1.0 + b_rank) + float(ir_weight) * (1.0 / (1.0 + i_rank))
            if score > best_score:
                best_score = score
                best_smiles = smiles
        return best_smiles, ir_score_by_smiles.get(best_smiles, 0.0), best_smiles != beam_top1

    # conservative
    ir_best_smiles, ir_best_score = ir_ranked[0]
    beam_top1_ir_score = ir_score_by_smiles.get(beam_top1, float("-inf"))
    should_switch = (
        ir_best_smiles != beam_top1
        and float(ir_best_score) >= float(min_abs_similarity)
        and (float(ir_best_score) - float(beam_top1_ir_score)) >= float(min_delta_similarity)
    )
    if should_switch:
        return ir_best_smiles, float(ir_best_score), True
    return beam_top1, float(beam_top1_ir_score), False


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
            m = re.search(r"([A-Z][a-z]?|[cnospb])", smiles[i + 1 : j])
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


def _length_penalized_score(score: float, length: int, alpha: float = 0.7) -> float:
    lp = ((5.0 + max(length, 1)) / 6.0) ** alpha
    return score / lp


def beam_search_single(
    model: BARTPatchModel,
    spectrum: torch.Tensor,
    formula_vec: torch.Tensor,
    beam_size: int,
    max_len: int,
    sos_id: int,
    eos_id: int,
    alpha: float = 0.7,
) -> List[Tuple[List[int], float]]:
    with torch.no_grad():
        memory = model.encode(spectrum.unsqueeze(0), formula_vec.unsqueeze(0))
        beams: List[Tuple[List[int], float, bool]] = [([sos_id], 0.0, False)]

        for _ in range(max_len - 1):
            new_beams: List[Tuple[List[int], float, bool]] = []
            all_finished = True
            for seq, score, finished in beams:
                if finished:
                    new_beams.append((seq, score, True))
                    continue
                all_finished = False
                tgt = torch.tensor([seq], dtype=torch.long, device=spectrum.device)
                logits = model.decode(memory, tgt)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                k = min(beam_size, log_probs.numel())
                topv, topi = torch.topk(log_probs, k=k, dim=-1)
                for logp, token_id in zip(topv.tolist(), topi.tolist()):
                    token_id = int(token_id)
                    new_seq = seq + [token_id]
                    is_finished = token_id == eos_id
                    new_beams.append((new_seq, score + float(logp), is_finished))

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
        out: List[Tuple[List[int], float]] = []
        for seq, raw_score, _ in beams:
            score = _length_penalized_score(raw_score, len(seq) - 1, alpha=alpha)
            out.append((seq, score))
        return out


def evaluate_teacher_forcing(
    model: BARTPatchModel,
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

            decoder_input = smiles_ids[:, :-1]
            target = smiles_ids[:, 1:]
            logits = model(spectra, formula_vec, decoder_input)
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
    ppl = float(torch.exp(torch.tensor(avg_loss)).item())
    token_acc = correct_tokens / max(total_tokens, 1)
    topk_acc = topk_correct_tokens / max(total_tokens, 1)
    return avg_loss, ppl, token_acc, topk_acc


def evaluate_generation(
    model: BARTPatchModel,
    dataset: Dataset,
    id_to_token: Dict[int, str],
    idx_to_elem: Dict[int, str],
    token_to_id: Dict[str, int],
    device: torch.device,
    sample_count: int,
    batch_size: int,
    max_len: int,
    sos_id: int,
    eos_id: int,
    beam_size: int,
    seed: int,
    smiles2ir_model: Optional[SMILES2IRRegressor] = None,
    smiles2ir_max_len: int = 256,
    rerank_similarity: str = "cosine",
    rerank_mode: str = "conservative",
    rerank_ir_weight: float = 0.25,
    rerank_min_abs_similarity: float = 0.80,
    rerank_min_delta_similarity: float = 0.03,
    rerank_top_m: int = 3,
) -> Tuple[float, List[Dict[str, str]], Dict[str, float]]:
    rng = random.Random(seed)
    total = len(dataset)
    sample_count = min(sample_count, total)
    indices = list(range(total))
    rng.shuffle(indices)
    indices = indices[:sample_count]

    samples: List[Dict[str, str]] = []
    exact_match = 0
    beam_top1_exact_match = 0
    elem_set_match = 0
    elem_count_match = 0
    elem_abs_err = 0
    elem_total = 0
    beam_topk_exact_match = 0
    ir_rerank_top1_exact_match = 0
    ir_rerank_score_sum = 0.0
    ir_rerank_score_count = 0
    ir_rerank_switch_count = 0

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
            generated_ids = model.generate(spectra_tensor, formula_tensor, max_len=max_len, sos_id=sos_id, eos_id=eos_id)

            for i in range(len(batch_indices)):
                gt = decode_ids(target_ids_batch[i].tolist(), id_to_token)
                greedy_pred = decode_ids(generated_ids[i].tolist(), id_to_token)
                if gt == greedy_pred:
                    exact_match += 1

                beam_results = beam_search_single(
                    model=model,
                    spectrum=spectra_tensor[i],
                    formula_vec=formula_tensor[i],
                    beam_size=beam_size,
                    max_len=max_len,
                    sos_id=sos_id,
                    eos_id=eos_id,
                )
                beam_candidates: List[Tuple[str, float]] = []
                seen_beam = set()
                for seq, score in beam_results:
                    smiles = decode_ids(seq, id_to_token)
                    if not smiles or smiles in seen_beam:
                        continue
                    seen_beam.add(smiles)
                    beam_candidates.append((smiles, float(score)))

                beam_smiles_full = [s for s, _ in beam_candidates]
                beam_top1_pred = beam_smiles_full[0] if beam_smiles_full else greedy_pred
                if gt == beam_top1_pred:
                    beam_top1_exact_match += 1
                if gt in beam_smiles_full:
                    beam_topk_exact_match += 1

                rerank_pred = beam_top1_pred
                rerank_candidates = beam_candidates[:rerank_top_m] if rerank_top_m > 0 else beam_candidates
                rerank_smiles = [s for s, _ in rerank_candidates]
                if smiles2ir_model is not None and rerank_smiles:
                    ranked_by_ir = rerank_candidates_by_ir_similarity(
                        smiles_candidates=rerank_smiles,
                        target_spectrum=spectra_tensor[i],
                        smiles2ir_model=smiles2ir_model,
                        token_to_id=token_to_id,
                        device=device,
                        max_smiles_len=smiles2ir_max_len,
                        similarity=rerank_similarity,
                    )
                    if ranked_by_ir:
                        rerank_pred, chosen_ir_score, switched = choose_rerank_candidate(
                            beam_candidates=rerank_candidates,
                            ir_ranked=ranked_by_ir,
                            mode=rerank_mode,
                            ir_weight=rerank_ir_weight,
                            min_abs_similarity=rerank_min_abs_similarity,
                            min_delta_similarity=rerank_min_delta_similarity,
                        )
                        ir_rerank_score_sum += float(chosen_ir_score)
                        ir_rerank_score_count += 1
                        if switched:
                            ir_rerank_switch_count += 1
                if smiles2ir_model is not None and gt == rerank_pred:
                    ir_rerank_top1_exact_match += 1

                samples.append(
                    {
                        "gt": gt,
                        "greedy_pred": greedy_pred,
                        "beam_top1_pred": beam_top1_pred,
                        "ir_rerank_pred": rerank_pred,
                    }
                )

                gt_elem_counts = formula_vec_to_counts(formula_batch[i], idx_to_elem)
                pred_elem_counts = parse_smiles_element_counts(rerank_pred if smiles2ir_model is not None else greedy_pred)
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

    em_rate = exact_match / max(len(samples), 1)
    elem_metrics = {
        "beam_top1_exact_match_rate": beam_top1_exact_match / max(len(samples), 1),
        "beam_topk_exact_match_rate": beam_topk_exact_match / max(len(samples), 1),
        "element_set_match_rate": elem_set_match / max(len(samples), 1),
        "element_count_exact_match_rate": elem_count_match / max(len(samples), 1),
        "element_count_mae": elem_abs_err / max(elem_total, 1),
    }
    if smiles2ir_model is not None:
        elem_metrics["ir_rerank_top1_exact_match_rate"] = ir_rerank_top1_exact_match / max(len(samples), 1)
        elem_metrics["ir_rerank_top1_avg_similarity"] = ir_rerank_score_sum / max(ir_rerank_score_count, 1)
        elem_metrics["ir_rerank_switch_rate"] = ir_rerank_switch_count / max(len(samples), 1)
        elem_metrics["ir_rerank_gain_vs_beam_top1"] = (
            elem_metrics["ir_rerank_top1_exact_match_rate"] - elem_metrics["beam_top1_exact_match_rate"]
        )
    return em_rate, samples, elem_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BARTPatch IR+Formula -> SMILES model.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--vocab-path", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--split-path", type=str, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--split", type=str, choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gen-samples", type=int, default=200)
    parser.add_argument("--print-samples", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=120)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--use-ir-rerank", action="store_true")
    parser.add_argument("--smiles2ir-model-path", type=str, default=DEFAULT_SMILES2IR_MODEL_PATH)
    parser.add_argument("--smiles2ir-config-path", type=str, default=DEFAULT_SMILES2IR_CONFIG_PATH)
    parser.add_argument("--rerank-similarity", type=str, choices=["cosine", "neg_mse"], default="cosine")
    parser.add_argument("--rerank-mode", type=str, choices=["conservative", "hybrid", "ir_only"], default="conservative")
    parser.add_argument("--rerank-ir-weight", type=float, default=0.25)
    parser.add_argument("--rerank-min-abs-sim", type=float, default=0.80)
    parser.add_argument("--rerank-min-delta-sim", type=float, default=0.03)
    parser.add_argument("--rerank-top-m", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_to_id, id_to_token = load_vocab(args.vocab_path)
    config = load_config(args.config_path)

    full_dataset = IRDataset(args.data_path, token_to_id)
    idx_to_elem = {v: k for k, v in full_dataset.formula_vocab.items()}

    if args.split == "all":
        eval_dataset = full_dataset
    else:
        split_indices = load_split_indices(args.split_path, args.split)
        eval_dataset = Subset(full_dataset, split_indices)

    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = build_model(args.model_path, config, device)
    smiles2ir_model = None
    smiles2ir_max_len = 256
    if args.use_ir_rerank:
        smiles2ir_config = load_config(args.smiles2ir_config_path)
        smiles2ir_model = build_smiles2ir_model(
            args.smiles2ir_model_path,
            smiles2ir_config,
            device=device,
            fallback_formula_dim=full_dataset.formula_dim,
        )
        smiles2ir_max_len = int(smiles2ir_config.get("max_len", 256))

    avg_loss, ppl, token_acc, topk_acc = evaluate_teacher_forcing(
        model=model,
        loader=loader,
        device=device,
        pad_id=config["pad_id"],
        topk=args.topk,
    )

    em_rate, samples, elem_metrics = evaluate_generation(
        model=model,
        dataset=eval_dataset,
        id_to_token=id_to_token,
        idx_to_elem=idx_to_elem,
        token_to_id=token_to_id,
        device=device,
        sample_count=args.gen_samples,
        batch_size=args.batch_size,
        max_len=args.max_len,
        sos_id=config["sos_id"],
        eos_id=config["eos_id"],
        beam_size=args.beam_size,
        seed=args.seed,
        smiles2ir_model=smiles2ir_model,
        smiles2ir_max_len=smiles2ir_max_len,
        rerank_similarity=args.rerank_similarity,
        rerank_mode=args.rerank_mode,
        rerank_ir_weight=args.rerank_ir_weight,
        rerank_min_abs_similarity=args.rerank_min_abs_sim,
        rerank_min_delta_similarity=args.rerank_min_delta_sim,
        rerank_top_m=args.rerank_top_m,
    )

    print("=" * 70)
    print("BARTPatch model evaluation finished")
    print(f"Device                 : {device}")
    print(f"Model checkpoint       : {args.model_path}")
    print(f"Use nonuniform patch   : {bool(config.get('use_nonuniform_patch', False))}")
    if bool(config.get("use_nonuniform_patch", False)):
        print(
            "Fingerprint patch cfg  : "
            f"[{config.get('fingerprint_start_idx')}:{config.get('fingerprint_end_idx')}) "
            f"fp={config.get('fingerprint_patch_size')} non_fp={config.get('non_fingerprint_patch_size')}"
        )
    if args.use_ir_rerank:
        print(f"SMILES->IR checkpoint  : {args.smiles2ir_model_path}")
        print(f"IR rerank similarity   : {args.rerank_similarity}")
        print(f"IR rerank mode         : {args.rerank_mode}")
        print(f"IR rerank top-m        : {args.rerank_top_m}")
        if args.rerank_mode == "hybrid":
            print(f"IR rerank weight       : {args.rerank_ir_weight}")
        if args.rerank_mode == "conservative":
            print(f"IR rerank min abs sim  : {args.rerank_min_abs_sim}")
            print(f"IR rerank min delta    : {args.rerank_min_delta_sim}")
    print(f"Evaluate split         : {args.split}")
    print(f"Dataset size           : {len(eval_dataset)}")
    print(f"Formula feature dim    : {full_dataset.formula_dim}")
    print("-" * 70)
    print(f"Teacher forcing loss   : {avg_loss:.6f}")
    print(f"Perplexity             : {ppl:.6f}")
    print(f"Token accuracy         : {token_acc:.4%}")
    print(f"Top-{args.topk} token acc     : {topk_acc:.4%}")
    print("-" * 70)
    print(f"Greedy exact-match@{len(samples)} : {em_rate:.4%}")
    print(f"Beam top-1 exact       : {elem_metrics['beam_top1_exact_match_rate']:.4%}")
    print(f"Beam top-{args.beam_size} exact  : {elem_metrics['beam_topk_exact_match_rate']:.4%}")
    if args.use_ir_rerank:
        print(f"IR-rerank top-1 exact  : {elem_metrics['ir_rerank_top1_exact_match_rate']:.4%}")
        print(f"IR-rerank avg score    : {elem_metrics['ir_rerank_top1_avg_similarity']:.6f}")
        print(f"IR-rerank switch rate  : {elem_metrics['ir_rerank_switch_rate']:.4%}")
        print(f"IR-rerank gain vs beam : {elem_metrics['ir_rerank_gain_vs_beam_top1']:+.4%}")
    print(f"Element set match      : {elem_metrics['element_set_match_rate']:.4%}")
    print(f"Element count exact    : {elem_metrics['element_count_exact_match_rate']:.4%}")
    print(f"Element count MAE      : {elem_metrics['element_count_mae']:.6f}")
    print("=" * 70)

    show_n = min(args.print_samples, len(samples))
    if show_n > 0:
        print("\nSample predictions:")
        for i in range(show_n):
            sample = samples[i]
            print(f"[{i + 1}] GT         : {sample['gt']}")
            print(f"    Greedy     : {sample['greedy_pred']}")
            print(f"    BeamTop1   : {sample['beam_top1_pred']}")
            if args.use_ir_rerank:
                print(f"    IR-Rerank  : {sample['ir_rerank_pred']}")


if __name__ == "__main__":
    main()
