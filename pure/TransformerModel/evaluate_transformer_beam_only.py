import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]  # pure
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TransformerModel.TransformerModel import IRFormulaTransformer
from util.dataloader import IRDataset


DEFAULT_MODEL_PATH = "checkpoints/TransformerModel/48.5+74best_transformer_model.pth"
DEFAULT_CONFIG_PATH = "checkpoints/TransformerModel/transformer_config.json"
DEFAULT_DATA_PATH = "data/raw_processed_data.pt"
DEFAULT_VOCAB_PATH = "pure/vocab.json"
DEFAULT_SPLIT_PATH = "checkpoints/data_split.pt"


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


def build_model(model_path: str, config: Dict, device: torch.device) -> IRFormulaTransformer:
    model = IRFormulaTransformer(
        vocab_size=config["vocab_size"],
        formula_dim=config["formula_dim"],
        input_points=config["input_points"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_tgt_len=config["max_tgt_len"],
        max_memory_len=config["max_memory_len"],
        encoder_buffer_layers=config.get("encoder_buffer_layers", 2),
        encoder_buffer_dim_feedforward=config.get("encoder_buffer_dim_feedforward", config["dim_feedforward"]),
        encoder_multiscale_target=config.get("encoder_multiscale_target", "mid"),
        encoder_use_coordconv=config.get("encoder_use_coordconv", False),
        encoder_patch_size=config.get("encoder_patch_size", 4),
        transformer_ffn_type=config.get("transformer_ffn_type", "gelu"),
        use_formula_input=config.get("use_formula_input", True),
        num_functional_groups=config.get("num_functional_groups", 0),
        use_functional_group_head=config.get("use_functional_group_aux", False),
        use_functional_group_token=config.get("use_functional_group_aux", False)
        and config.get("use_functional_group_fusion", False),
        functional_group_head_dim=config.get("functional_group_head_dim", 256),
        functional_group_dropout=config.get("dropout", 0.1),
        functional_group_detach_fusion=config.get("functional_group_detach_fusion", False),
        pad_id=config["pad_id"],
        sos_id=config["sos_id"],
        eos_id=config["eos_id"],
    )
    state_dict = torch.load(model_path, map_location="cpu")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(
            "Warning: checkpoint mismatch; loaded with strict=False. "
            f"missing_keys={list(incompatible.missing_keys)} unexpected_keys={list(incompatible.unexpected_keys)}"
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


def formula_vec_to_counts(formula_vec: torch.Tensor, idx_to_elem: Dict[int, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    vec = formula_vec.detach().cpu().tolist()
    for idx, value in enumerate(vec):
        if value > 0:
            counts[idx_to_elem[idx]] = int(round(float(value)))
    return counts


def _normalize_element_symbol(elem: str) -> str:
    if elem in {"c", "n", "o", "p", "s", "b"}:
        return elem.upper()
    return elem


def _extract_element_from_token(token: str) -> Optional[str]:
    if token in {"Br", "Cl"}:
        return token
    if token in {"B", "C", "N", "O", "P", "S", "F", "I", "H", "b", "c", "n", "o", "p", "s"}:
        return _normalize_element_symbol(token)
    if token.startswith("[") and token.endswith("]"):
        m = re.search(r"([A-Z][a-z]?|[cnospb])", token[1:-1])
        if m:
            return _normalize_element_symbol(m.group(1))
    return None


def _normalize_constraint_counts(formula_counts: Dict[str, int], ignore_elements: Set[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for elem, cnt in formula_counts.items():
        norm_elem = _normalize_element_symbol(elem)
        if norm_elem in ignore_elements:
            continue
        if cnt > 0:
            out[norm_elem] = int(cnt)
    return out


def _is_counts_exact_match(pred_counts: Dict[str, int], target_counts: Dict[str, int]) -> bool:
    keys = set(pred_counts.keys()) | set(target_counts.keys())
    for key in keys:
        if int(pred_counts.get(key, 0)) != int(target_counts.get(key, 0)):
            return False
    return True


def _length_penalized_score(score: float, length: int, alpha: float = 0.7) -> float:
    lp = ((5.0 + max(length, 1)) / 6.0) ** alpha
    return score / lp


def beam_search_single(
    model: IRFormulaTransformer,
    spectrum: torch.Tensor,
    formula_vec: torch.Tensor,
    id_to_token: Dict[int, str],
    beam_size: int,
    max_len: int,
    sos_id: int,
    eos_id: int,
    alpha: float = 0.7,
    use_formula_constrained_beam: bool = False,
    formula_counts: Optional[Dict[str, int]] = None,
    constraint_ignore_elements: Optional[Set[str]] = None,
    constraint_require_exact_match: bool = False,
) -> List[Tuple[List[int], float]]:
    with torch.no_grad():
        memory = model.encoder(spectrum.unsqueeze(0), formula_vec.unsqueeze(0))
        if use_formula_constrained_beam and formula_counts is None:
            raise ValueError("formula_counts must be provided when use_formula_constrained_beam=True")

        ignore_elements = constraint_ignore_elements if constraint_ignore_elements is not None else set()
        normalized_formula_counts = (
            _normalize_constraint_counts(formula_counts or {}, ignore_elements=ignore_elements)
            if use_formula_constrained_beam
            else {}
        )
        beams: List[Tuple[List[int], float, bool, Dict[str, int]]] = [([sos_id], 0.0, False, {})]

        for _ in range(max_len - 1):
            new_beams: List[Tuple[List[int], float, bool, Dict[str, int]]] = []
            all_finished = True

            for seq, score, finished, elem_counts in beams:
                if finished:
                    new_beams.append((seq, score, True, elem_counts))
                    continue

                all_finished = False
                tgt_ids = torch.tensor([seq], dtype=torch.long, device=spectrum.device)
                logits = model.decoder(memory, tgt_ids)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)

                k = min(beam_size, log_probs.numel())
                topv, topi = torch.topk(log_probs, k=k, dim=-1)
                for logp, token_id in zip(topv.tolist(), topi.tolist()):
                    token_id = int(token_id)
                    token_str = id_to_token.get(token_id, "<UNK>")
                    new_seq = seq + [token_id]
                    is_finished = token_id == eos_id
                    new_elem_counts = elem_counts if not use_formula_constrained_beam else dict(elem_counts)

                    if use_formula_constrained_beam:
                        elem = _extract_element_from_token(token_str)
                        if elem is not None and elem not in ignore_elements:
                            next_cnt = int(new_elem_counts.get(elem, 0)) + 1
                            limit = int(normalized_formula_counts.get(elem, 0))
                            if next_cnt > limit:
                                continue
                            new_elem_counts[elem] = next_cnt

                        if is_finished and constraint_require_exact_match:
                            if not _is_counts_exact_match(new_elem_counts, normalized_formula_counts):
                                continue

                    new_beams.append((new_seq, score + float(logp), is_finished, new_elem_counts))

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
        for seq, raw_score, _, _ in beams:
            score = _length_penalized_score(raw_score, len(seq) - 1, alpha=alpha)
            out.append((seq, score))
        return out


def decode_unique_beam_smiles(
    beam_results: List[Tuple[List[int], float]],
    id_to_token: Dict[int, str],
) -> List[str]:
    seen = set()
    out: List[str] = []
    for seq, _score in beam_results:
        smiles = decode_ids(seq, id_to_token)
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)
        out.append(smiles)
    return out


def evaluate_beam_metrics(
    model: IRFormulaTransformer,
    dataset,
    id_to_token: Dict[int, str],
    idx_to_elem: Dict[int, str],
    sample_count: int,
    beam_size: int,
    max_len: int,
    sos_id: int,
    eos_id: int,
    seed: int,
    use_formula_constrained_beam: bool = False,
    constraint_ignore_elements: Optional[Set[str]] = None,
    constraint_require_exact_match: bool = False,
) -> Tuple[int, float, float]:
    total = len(dataset)
    if sample_count <= 0 or sample_count >= total:
        indices = list(range(total))
    else:
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(total, generator=rng).tolist()
        indices = perm[:sample_count]

    top1_exact = 0
    top10_exact = 0
    topk_eval = min(10, beam_size)

    for idx in tqdm(indices, desc="BeamEval", leave=False):
        spectra, formula_vec, target_ids = dataset[idx]
        gt = decode_ids(target_ids.tolist(), id_to_token)
        formula_counts = formula_vec_to_counts(formula_vec, idx_to_elem)
        beam_results = beam_search_single(
            model=model,
            spectrum=spectra.to(next(model.parameters()).device),
            formula_vec=formula_vec.to(next(model.parameters()).device),
            id_to_token=id_to_token,
            beam_size=beam_size,
            max_len=max_len,
            sos_id=sos_id,
            eos_id=eos_id,
            use_formula_constrained_beam=use_formula_constrained_beam,
            formula_counts=formula_counts,
            constraint_ignore_elements=constraint_ignore_elements,
            constraint_require_exact_match=constraint_require_exact_match,
        )
        beam_smiles = decode_unique_beam_smiles(beam_results, id_to_token)
        if beam_smiles and gt == beam_smiles[0]:
            top1_exact += 1
        if gt in beam_smiles[:topk_eval]:
            top10_exact += 1

    denom = max(len(indices), 1)
    return len(indices), top1_exact / denom, top10_exact / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast beam-only evaluation for Transformer IR->SMILES.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--vocab-path", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--split-path", type=str, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--split", type=str, choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--gen-samples", type=int, default=10000)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=120)
    parser.add_argument("--use-formula-constrained-beam", action="store_true")
    parser.add_argument("--constraint-ignore-elements", type=str, default="H")
    parser.add_argument("--constraint-require-exact-match", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_to_id, id_to_token = load_vocab(args.vocab_path)
    config = load_config(args.config_path)
    ignore_elements = {s.strip() for s in args.constraint_ignore_elements.split(",") if s.strip()}

    full_dataset = IRDataset(args.data_path, token_to_id, canonicalize_smiles=True)
    idx_to_elem = {v: k for k, v in full_dataset.formula_vocab.items()}

    if args.split == "all":
        eval_dataset = full_dataset
    else:
        split_indices = load_split_indices(args.split_path, args.split)
        eval_dataset = Subset(full_dataset, split_indices)

    model = build_model(args.model_path, config, device)

    start = time.time()
    eval_count, beam_top1_rate, beam_top10_rate = evaluate_beam_metrics(
        model=model,
        dataset=eval_dataset,
        id_to_token=id_to_token,
        idx_to_elem=idx_to_elem,
        sample_count=args.gen_samples,
        beam_size=args.beam_size,
        max_len=args.max_len,
        sos_id=config["sos_id"],
        eos_id=config["eos_id"],
        seed=args.seed,
        use_formula_constrained_beam=args.use_formula_constrained_beam,
        constraint_ignore_elements=ignore_elements,
        constraint_require_exact_match=args.constraint_require_exact_match,
    )
    elapsed = time.time() - start

    print("=" * 70)
    print("Fast beam-only evaluation finished")
    print(f"Device                 : {device}")
    print(f"Model checkpoint       : {args.model_path}")
    print(f"Evaluate split         : {args.split}")
    print(f"Dataset size           : {len(eval_dataset)}")
    print(f"Generation sample count: {eval_count}")
    print(f"Beam size              : {args.beam_size}")
    print(f"Formula constrained    : {args.use_formula_constrained_beam}")
    if args.use_formula_constrained_beam:
        print(f"Constraint ignore elems: {sorted(ignore_elements)}")
        print(f"Constraint exact match : {args.constraint_require_exact_match}")
    print("-" * 70)
    print(f"Beam top-1 exact       : {beam_top1_rate:.4%}")
    print(f"Beam top-{min(10, args.beam_size)} exact  : {beam_top10_rate:.4%}")
    print(f"Elapsed time (s)       : {elapsed:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
