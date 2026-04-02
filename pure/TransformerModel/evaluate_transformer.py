import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    _MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
except Exception:
    Chem = None
    DataStructs = None
    AllChem = None
    RDLogger = None
    rdFingerprintGenerator = None
    _MORGAN_GENERATOR = None

ROOT = Path(__file__).resolve().parents[1]  # pure
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TransformerModel.TransformerModel import IRFormulaTransformer
from SMILES2IRTransformer.SMILES2IRTransformer import SMILES2IRTransformer
from FunctionalGroupModel.IRFunctionalGroupModel import DEFAULT_FUNCTIONAL_GROUPS, LightweightIRFunctionalGroupClassifier
from FunctionalGroupModel.functional_group_utils import extract_functional_group_labels
from util.dataloader import IRDataset, collate_fn


DEFAULT_MODEL_PATH = "checkpoints/TransformerModel/best_transformer_model.pth"
DEFAULT_CONFIG_PATH = "checkpoints/TransformerModel/transformer_config.json"
DEFAULT_DATA_PATH = "data/raw_processed_data.pt"
DEFAULT_VOCAB_PATH = "pure/vocab.json"
DEFAULT_SPLIT_PATH = "checkpoints/data_split.pt"
DEFAULT_SMILES2IR_MODEL_PATH = "checkpoints/SMILES2IRTransformer/best_smiles2ir_transformer.pth"
DEFAULT_SMILES2IR_CONFIG_PATH = "checkpoints/SMILES2IRTransformer/smiles2ir_transformer_config.json"
DEFAULT_FUNCTIONAL_GROUP_MODEL_PATH = "checkpoints/FunctionalGroupModel/best_functional_group_model.pth"
DEFAULT_FUNCTIONAL_GROUP_CONFIG_PATH = "checkpoints/FunctionalGroupModel/functional_group_config.json"

SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|C|N|O|P|S|F|I|b|n|o|s|p|c|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


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
    for k in keys:
        if int(pred_counts.get(k, 0)) != int(target_counts.get(k, 0)):
            return False
    return True


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
            "Warning: Transformer checkpoint mismatch; loaded with strict=False. "
            f"missing_keys={list(incompatible.missing_keys)} unexpected_keys={list(incompatible.unexpected_keys)}"
        )
        print(f"Original load error: {exc}")
    model = model.to(device)
    model.eval()
    return model


def build_smiles2ir_model(
    model_path: str,
    config: Dict,
    dataset_formula_dim: int,
    device: torch.device,
) -> SMILES2IRTransformer:
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
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(
            "Warning: SMILES2IRTransformer checkpoint mismatch; loaded with strict=False. "
            f"missing_keys={list(incompatible.missing_keys)} unexpected_keys={list(incompatible.unexpected_keys)}"
        )
        print(f"Original load error: {exc}")
    model = model.to(device)
    model.eval()
    return model



def build_functional_group_model(
    model_path: str,
    config_path: Optional[str],
    device: torch.device,
) -> Tuple[LightweightIRFunctionalGroupClassifier, Tuple[str, ...]]:
    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    file_config = {}
    if config_path and Path(config_path).exists():
        file_config = load_config(config_path)

    merged_config = dict(ckpt_config)
    merged_config.update(file_config)
    label_names = tuple(merged_config.get("label_names", checkpoint.get("label_names", DEFAULT_FUNCTIONAL_GROUPS)))

    model = LightweightIRFunctionalGroupClassifier(
        input_points=int(merged_config["input_points"]),
        num_labels=int(merged_config["num_labels"]),
        label_names=label_names,
        stem_channels=int(merged_config.get("stem_channels", 48)),
        hidden_dim=int(merged_config.get("hidden_dim", 192)),
        head_dim=int(merged_config.get("head_dim", 256)),
        dropout=float(merged_config.get("dropout", 0.15)),
        use_coordconv=bool(merged_config.get("use_coordconv", True)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, label_names


def predict_required_functional_groups(
    functional_group_model: LightweightIRFunctionalGroupClassifier,
    spectrum: torch.Tensor,
    label_names: Tuple[str, ...],
    threshold: float,
) -> Tuple[torch.Tensor, List[int], List[str], Dict[str, float]]:
    probs = functional_group_model.predict_proba(spectrum.unsqueeze(0)).squeeze(0).detach().cpu()
    required_indices = [idx for idx, prob in enumerate(probs.tolist()) if float(prob) >= float(threshold)]
    required_names = [label_names[idx] for idx in required_indices]
    required_scores = {label_names[idx]: float(probs[idx].item()) for idx in required_indices}
    return probs, required_indices, required_names, required_scores


def parse_smiles_mol(smiles: str, mol_cache: Dict[str, Optional[object]]) -> Optional[object]:
    if Chem is None or not smiles:
        return None
    if smiles in mol_cache:
        return mol_cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    mol_cache[smiles] = mol
    return mol


def is_valid_smiles(smiles: str, mol_cache: Dict[str, Optional[object]]) -> bool:
    return parse_smiles_mol(smiles, mol_cache) is not None


def _decode_unique_beam_candidates(
    beam_results: List[Tuple[List[int], float]],
    id_to_token: Dict[int, str],
    mol_cache: Dict[str, Optional[object]],
    require_valid_smiles: bool = True,
) -> List[Tuple[str, float]]:
    beam_candidates: List[Tuple[str, float]] = []
    seen_beam = set()
    for seq, score in beam_results:
        smiles = decode_ids(seq, id_to_token)
        if not smiles or smiles in seen_beam:
            continue
        if require_valid_smiles and Chem is not None and not is_valid_smiles(smiles, mol_cache):
            continue
        seen_beam.add(smiles)
        beam_candidates.append((smiles, float(score)))
    return beam_candidates


def _candidate_has_required_functional_groups(
    smiles: str,
    required_label_indices: List[int],
    functional_group_label_names: Tuple[str, ...],
    fg_label_cache: Dict[str, List[int]],
) -> bool:
    if not required_label_indices:
        return True
    if smiles not in fg_label_cache:
        fg_label_cache[smiles] = extract_functional_group_labels(smiles, functional_group_label_names)
    labels = fg_label_cache[smiles]
    return all(int(labels[idx]) == 1 for idx in required_label_indices)
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


def _length_penalized_score(score: float, length: int, alpha: float = 0.7) -> float:
    lp = ((5.0 + max(length, 1)) / 6.0) ** alpha
    return score / lp


def smiles_to_fingerprint(
    smiles: str,
    fp_cache: Dict[str, object],
    mol_cache: Optional[Dict[str, Optional[object]]] = None,
) -> Optional[object]:
    if Chem is None or DataStructs is None or not smiles:
        return None
    if smiles in fp_cache:
        return fp_cache[smiles]
    mol = parse_smiles_mol(smiles, mol_cache if mol_cache is not None else {})
    if mol is None:
        fp_cache[smiles] = None
        return None
    if _MORGAN_GENERATOR is not None:
        fp = _MORGAN_GENERATOR.GetFingerprint(mol)
    elif AllChem is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    else:
        fp = None
    fp_cache[smiles] = fp
    return fp


def tanimoto_similarity_smiles(smiles_a: str, smiles_b: str, fp_cache: Dict[str, object]) -> float:
    fp_a = smiles_to_fingerprint(smiles_a, fp_cache)
    fp_b = smiles_to_fingerprint(smiles_b, fp_cache)
    if fp_a is None or fp_b is None or DataStructs is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def build_pairwise_tanimoto_matrix(smiles_list: List[str], fp_cache: Dict[str, object]) -> List[List[float]]:
    n = len(smiles_list)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim = tanimoto_similarity_smiles(smiles_list[i], smiles_list[j], fp_cache)
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix


def choose_consensus_cluster(
    smiles_candidates: List[str],
    fp_cache: Dict[str, object],
    similarity_threshold: float = 0.55,
    min_cluster_size: int = 3,
) -> Tuple[List[str], List[float], float]:
    if not smiles_candidates:
        return [], [], 0.0

    matrix = build_pairwise_tanimoto_matrix(smiles_candidates, fp_cache)
    n = len(smiles_candidates)
    avg_scores: List[float] = []
    for i in range(n):
        if n == 1:
            avg_scores.append(1.0)
        else:
            avg_scores.append(sum(matrix[i][j] for j in range(n) if j != i) / float(n - 1))

    visited = [False] * n
    components: List[List[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = [start]
        while stack:
            cur = stack.pop()
            for nxt in range(n):
                if visited[nxt] or nxt == cur:
                    continue
                if matrix[cur][nxt] >= similarity_threshold:
                    visited[nxt] = True
                    stack.append(nxt)
                    comp.append(nxt)
        components.append(sorted(comp))

    def component_score(comp: List[int]) -> Tuple[int, float]:
        if len(comp) <= 1:
            return len(comp), 0.0
        vals = []
        for i, idx_i in enumerate(comp):
            for idx_j in comp[i + 1 :]:
                vals.append(matrix[idx_i][idx_j])
        return len(comp), sum(vals) / max(len(vals), 1)

    best_comp: List[int] = []
    best_key = (-1, -1.0)
    for comp in components:
        key = component_score(comp)
        if key > best_key:
            best_comp = comp
            best_key = key

    if len(best_comp) < min_cluster_size:
        return [], avg_scores, 0.0

    comp_sims = []
    for i, idx_i in enumerate(best_comp):
        for idx_j in best_comp[i + 1 :]:
            comp_sims.append(matrix[idx_i][idx_j])
    avg_internal = sum(comp_sims) / max(len(comp_sims), 1)
    return [smiles_candidates[i] for i in best_comp], avg_scores, avg_internal


def rerank_candidates_by_ir_similarity(
    smiles_candidates: List[str],
    target_spectrum: torch.Tensor,
    formula_vec: torch.Tensor,
    smiles2ir_model: SMILES2IRTransformer,
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
    formula_batch = formula_vec.unsqueeze(0).expand(smiles_ids.size(0), -1).to(device)

    with torch.no_grad():
        pred_spectra = smiles2ir_model.predict_from_smiles_ids(smiles_ids, formula_vec=formula_batch)
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
    """
    Returns top beam token sequences (including <SOS>, maybe <EOS>).
    """
    with torch.no_grad():
        memory = model.encoder(spectrum.unsqueeze(0), formula_vec.unsqueeze(0))  # [1, S, D]
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
                logits = model.decoder(memory, tgt_ids)  # [1, T, V]
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)  # [V]

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



def generate_functional_group_constrained_candidates(
    model: IRFormulaTransformer,
    spectrum: torch.Tensor,
    formula_vec: torch.Tensor,
    id_to_token: Dict[int, str],
    initial_beam_size: int,
    max_len: int,
    sos_id: int,
    eos_id: int,
    use_formula_constrained_beam: bool,
    formula_counts: Dict[str, int],
    constraint_ignore_elements: Optional[Set[str]],
    constraint_require_exact_match: bool,
    required_label_indices: List[int],
    functional_group_label_names: Tuple[str, ...],
    fg_label_cache: Dict[str, List[int]],
    mol_cache: Dict[str, Optional[object]],
    beam_multiplier: float = 2.0,
    max_beam_size: int = 40,
    max_regenerations: int = 2,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], Dict[str, object]]:
    current_beam_size = max(int(initial_beam_size), 1)
    regeneration_rounds = 0
    last_raw_candidates: List[Tuple[str, float]] = []
    satisfying_candidates: List[Tuple[str, float]] = []

    while True:
        beam_results = beam_search_single(
            model=model,
            spectrum=spectrum,
            formula_vec=formula_vec,
            id_to_token=id_to_token,
            beam_size=current_beam_size,
            max_len=max_len,
            sos_id=sos_id,
            eos_id=eos_id,
            use_formula_constrained_beam=use_formula_constrained_beam,
            formula_counts=formula_counts,
            constraint_ignore_elements=constraint_ignore_elements,
            constraint_require_exact_match=constraint_require_exact_match,
        )
        last_raw_candidates = _decode_unique_beam_candidates(
            beam_results,
            id_to_token=id_to_token,
            mol_cache=mol_cache,
            require_valid_smiles=True,
        )

        if not required_label_indices:
            satisfying_candidates = list(last_raw_candidates)
        else:
            satisfying_candidates = [
                item
                for item in last_raw_candidates
                if _candidate_has_required_functional_groups(
                    item[0],
                    required_label_indices=required_label_indices,
                    functional_group_label_names=functional_group_label_names,
                    fg_label_cache=fg_label_cache,
                )
            ]

        should_stop = (
            satisfying_candidates
            or regeneration_rounds >= int(max_regenerations)
            or current_beam_size >= int(max_beam_size)
        )
        if should_stop:
            active_candidates = satisfying_candidates if satisfying_candidates else last_raw_candidates
            meta = {
                "triggered": bool(required_label_indices),
                "found_satisfying_candidates": bool(satisfying_candidates),
                "regenerated": regeneration_rounds > 0,
                "regeneration_rounds": regeneration_rounds,
                "searched_beam_size": current_beam_size,
                "raw_candidate_count": len(last_raw_candidates),
                "active_candidate_count": len(active_candidates),
            }
            return last_raw_candidates, active_candidates, meta

        next_beam_size = int(max(current_beam_size + 1, round(current_beam_size * float(beam_multiplier))))
        current_beam_size = min(int(max_beam_size), next_beam_size)
        regeneration_rounds += 1


def evaluate_teacher_forcing(
    model: IRFormulaTransformer,
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
            outputs = model(spectra, formula_vec, decoder_input)
            logits = outputs

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
    model: IRFormulaTransformer,
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
    use_formula_constrained_beam: bool = False,
    constraint_ignore_elements: Optional[Set[str]] = None,
    constraint_require_exact_match: bool = False,
    smiles2ir_model: Optional[SMILES2IRTransformer] = None,
    smiles2ir_max_len: int = 256,
    rerank_similarity: str = "cosine",
    rerank_mode: str = "conservative",
    rerank_ir_weight: float = 0.25,
    rerank_min_abs_similarity: float = 0.80,
    rerank_min_delta_similarity: float = 0.03,
    rerank_top_m: int = 3,
    beam_display_top_k: int = 10,
    consensus_similarity_threshold: float = 0.55,
    consensus_min_cluster_size: int = 3,
    functional_group_model: Optional[LightweightIRFunctionalGroupClassifier] = None,
    functional_group_label_names: Optional[Tuple[str, ...]] = None,
    functional_group_threshold: float = 0.95,
    functional_group_beam_multiplier: float = 2.0,
    functional_group_max_beam_size: int = 40,
    functional_group_max_regenerations: int = 2,
) -> Tuple[float, List[Dict[str, str]], Dict[str, float], List[Dict[str, object]]]:
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
    greedy_tanimoto_sum = 0.0
    beam_top1_tanimoto_sum = 0.0
    beam_topk_best_tanimoto_sum = 0.0
    consensus_top1_exact_match = 0
    consensus_tanimoto_sum = 0.0
    consensus_cluster_size_sum = 0.0
    consensus_cluster_internal_tanimoto_sum = 0.0
    ir_rerank_tanimoto_sum = 0.0
    fg_constraint_trigger_count = 0
    fg_constraint_found_count = 0
    fg_constraint_final_satisfied_count = 0
    fg_constraint_regeneration_count = 0
    fg_constraint_required_group_sum = 0.0
    topk_all_wrong_samples: List[Dict[str, object]] = []
    fp_cache: Dict[str, object] = {}
    mol_cache: Dict[str, Optional[object]] = {}
    fg_label_cache: Dict[str, List[int]] = {}

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
            generated_ids = model.generate(
                spectra_tensor,
                formula_tensor,
                max_len=max_len,
                sos_id=sos_id,
                eos_id=eos_id,
            )

            for i in range(len(batch_indices)):
                gt = decode_ids(target_ids_batch[i].tolist(), id_to_token)
                greedy_pred = decode_ids(generated_ids[i].tolist(), id_to_token)
                if gt == greedy_pred:
                    exact_match += 1
                greedy_tanimoto_sum += tanimoto_similarity_smiles(gt, greedy_pred, fp_cache)

                formula_counts = formula_vec_to_counts(formula_batch[i], idx_to_elem)
                required_fg_indices: List[int] = []
                required_fg_names: List[str] = []
                required_fg_scores: Dict[str, float] = {}
                fg_constraint_meta: Dict[str, object] = {
                    "triggered": False,
                    "found_satisfying_candidates": False,
                    "regenerated": False,
                    "regeneration_rounds": 0,
                    "searched_beam_size": beam_size,
                    "raw_candidate_count": 0,
                    "active_candidate_count": 0,
                }
                active_fg_label_names = functional_group_label_names if functional_group_label_names is not None else DEFAULT_FUNCTIONAL_GROUPS
                if functional_group_model is not None and functional_group_label_names is not None:
                    _, required_fg_indices, required_fg_names, required_fg_scores = predict_required_functional_groups(
                        functional_group_model=functional_group_model,
                        spectrum=spectra_tensor[i],
                        label_names=functional_group_label_names,
                        threshold=functional_group_threshold,
                    )

                raw_beam_candidates, beam_candidates, fg_constraint_meta = generate_functional_group_constrained_candidates(
                    model=model,
                    spectrum=spectra_tensor[i],
                    formula_vec=formula_tensor[i],
                    id_to_token=id_to_token,
                    initial_beam_size=beam_size,
                    max_len=max_len,
                    sos_id=sos_id,
                    eos_id=eos_id,
                    use_formula_constrained_beam=use_formula_constrained_beam,
                    formula_counts=formula_counts,
                    constraint_ignore_elements=constraint_ignore_elements,
                    constraint_require_exact_match=constraint_require_exact_match,
                    required_label_indices=required_fg_indices,
                    functional_group_label_names=active_fg_label_names,
                    fg_label_cache=fg_label_cache,
                    mol_cache=mol_cache,
                    beam_multiplier=functional_group_beam_multiplier,
                    max_beam_size=functional_group_max_beam_size,
                    max_regenerations=functional_group_max_regenerations,
                )
                if fg_constraint_meta["triggered"]:
                    fg_constraint_trigger_count += 1
                    fg_constraint_required_group_sum += float(len(required_fg_names))
                if fg_constraint_meta["found_satisfying_candidates"]:
                    fg_constraint_found_count += 1
                if fg_constraint_meta["regenerated"]:
                    fg_constraint_regeneration_count += 1

                beam_smiles = [s for s, _ in beam_candidates]
                beam_top1_pred = beam_smiles[0] if beam_smiles else ""
                if gt == beam_top1_pred:
                    beam_top1_exact_match += 1
                if gt in beam_smiles:
                    beam_topk_exact_match += 1
                beam_top1_tanimoto_sum += tanimoto_similarity_smiles(gt, beam_top1_pred, fp_cache)
                beam_topk_best_tanimoto_sum += max(
                    [tanimoto_similarity_smiles(gt, s, fp_cache) for s in beam_smiles[:beam_display_top_k]] or [0.0]
                )

                topk_candidates = beam_candidates[:beam_display_top_k]
                topk_smiles = [s for s, _ in topk_candidates]
                consensus_cluster, avg_internal_scores, consensus_internal_tanimoto = choose_consensus_cluster(
                    topk_smiles,
                    fp_cache=fp_cache,
                    similarity_threshold=consensus_similarity_threshold,
                    min_cluster_size=consensus_min_cluster_size,
                )
                consensus_set = set(consensus_cluster)
                if consensus_cluster:
                    consensus_candidates = [item for item in topk_candidates if item[0] in consensus_set]
                    consensus_pred = consensus_candidates[0][0]
                    consensus_cluster_size_sum += float(len(consensus_cluster))
                    consensus_cluster_internal_tanimoto_sum += float(consensus_internal_tanimoto)
                else:
                    consensus_candidates = topk_candidates
                    consensus_pred = beam_top1_pred
                if gt == consensus_pred:
                    consensus_top1_exact_match += 1
                consensus_tanimoto_sum += tanimoto_similarity_smiles(gt, consensus_pred, fp_cache)

                rerank_pred = consensus_pred
                base_candidates = consensus_candidates if consensus_candidates else topk_candidates
                if consensus_candidates:
                    rerank_candidates = base_candidates
                else:
                    rerank_candidates = base_candidates[:rerank_top_m] if rerank_top_m > 0 else base_candidates
                rerank_smiles = [s for s, _ in rerank_candidates]
                if smiles2ir_model is not None and rerank_smiles:
                    ranked_by_ir = rerank_candidates_by_ir_similarity(
                        smiles_candidates=rerank_smiles,
                        target_spectrum=spectra_tensor[i],
                        formula_vec=formula_tensor[i],
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
                if smiles2ir_model is not None:
                    ir_rerank_tanimoto_sum += tanimoto_similarity_smiles(gt, rerank_pred, fp_cache)

                final_fg_satisfied = True
                if functional_group_model is not None and functional_group_label_names is not None and required_fg_names:
                    final_fg_satisfied = _candidate_has_required_functional_groups(
                        rerank_pred,
                        required_label_indices=required_fg_indices,
                        functional_group_label_names=functional_group_label_names,
                        fg_label_cache=fg_label_cache,
                    )
                    if final_fg_satisfied:
                        fg_constraint_final_satisfied_count += 1

                top10_examples = []
                avg_internal_map = {smiles: avg_internal_scores[idx] for idx, smiles in enumerate(topk_smiles)}
                for rank, (smiles, score) in enumerate(topk_candidates, start=1):
                    top10_examples.append(
                        {
                            "rank": rank,
                            "smiles": smiles,
                            "beam_score": float(score),
                            "tanimoto_to_gt": tanimoto_similarity_smiles(gt, smiles, fp_cache),
                            "avg_top10_tanimoto": float(avg_internal_map.get(smiles, 0.0)),
                            "in_consensus_cluster": smiles in consensus_set,
                            "satisfies_required_functional_groups": (
                                _candidate_has_required_functional_groups(
                                    smiles,
                                    required_label_indices=required_fg_indices,
                                    functional_group_label_names=functional_group_label_names if functional_group_label_names is not None else tuple(),
                                    fg_label_cache=fg_label_cache,
                                )
                                if functional_group_model is not None and functional_group_label_names is not None and required_fg_names
                                else True
                            ),
                        }
                    )

                samples.append(
                    {
                        "sample_index": int(batch_indices[i]),
                        "gt": gt,
                        "greedy_pred": greedy_pred,
                        "beam_top1_pred": beam_top1_pred,
                        "consensus_pred": consensus_pred,
                        "ir_rerank_pred": rerank_pred,
                        "top10_candidates": top10_examples,
                        "consensus_cluster_size": len(consensus_cluster),
                        "consensus_cluster_internal_tanimoto": float(consensus_internal_tanimoto),
                        "required_functional_groups": required_fg_names,
                        "required_functional_group_probs": required_fg_scores,
                        "functional_group_constraint_triggered": bool(fg_constraint_meta["triggered"]),
                        "functional_group_constraint_found": bool(fg_constraint_meta["found_satisfying_candidates"]),
                        "functional_group_constraint_regenerated": bool(fg_constraint_meta["regenerated"]),
                        "functional_group_constraint_final_satisfied": bool(final_fg_satisfied),
                        "functional_group_constraint_beam_size": int(fg_constraint_meta["searched_beam_size"]),
                    }
                )

                if gt not in topk_smiles:
                    topk_all_wrong_samples.append(
                        {
                            "sample_index": int(batch_indices[i]),
                            "gt": gt,
                            "greedy_pred": greedy_pred,
                            "beam_top1_pred": beam_top1_pred,
                            "consensus_pred": consensus_pred,
                            "ir_rerank_pred": rerank_pred,
                            "consensus_cluster_size": int(len(consensus_cluster)),
                            "consensus_cluster_internal_tanimoto": float(consensus_internal_tanimoto),
                            "required_functional_groups": list(required_fg_names),
                            "required_functional_group_probs": dict(required_fg_scores),
                            "functional_group_constraint_triggered": bool(fg_constraint_meta["triggered"]),
                            "functional_group_constraint_found": bool(fg_constraint_meta["found_satisfying_candidates"]),
                            "functional_group_constraint_regenerated": bool(fg_constraint_meta["regenerated"]),
                            "functional_group_constraint_final_satisfied": bool(final_fg_satisfied),
                            "functional_group_constraint_beam_size": int(fg_constraint_meta["searched_beam_size"]),
                            "topk_candidates": list(top10_examples),
                        }
                    )

                gt_elem_counts = formula_vec_to_counts(formula_batch[i], idx_to_elem)
                pred_elem_counts = parse_smiles_element_counts(rerank_pred if smiles2ir_model is not None else consensus_pred)

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
        "greedy_tanimoto_mean": greedy_tanimoto_sum / max(len(samples), 1),
        "beam_top1_exact_match_rate": beam_top1_exact_match / max(len(samples), 1),
        "beam_topk_exact_match_rate": beam_topk_exact_match / max(len(samples), 1),
        "beam_top1_tanimoto_mean": beam_top1_tanimoto_sum / max(len(samples), 1),
        "beam_topk_best_tanimoto_mean": beam_topk_best_tanimoto_sum / max(len(samples), 1),
        "consensus_top1_exact_match_rate": consensus_top1_exact_match / max(len(samples), 1),
        "consensus_top1_tanimoto_mean": consensus_tanimoto_sum / max(len(samples), 1),
        "consensus_cluster_size_mean": consensus_cluster_size_sum / max(len(samples), 1),
        "consensus_cluster_internal_tanimoto_mean": (
            consensus_cluster_internal_tanimoto_sum / max(len(samples), 1)
        ),
        "element_set_match_rate": elem_set_match / max(len(samples), 1),
        "element_count_exact_match_rate": elem_count_match / max(len(samples), 1),
        "element_count_mae": elem_abs_err / max(elem_total, 1),
    }
    if functional_group_model is not None:
        elem_metrics["functional_group_constraint_trigger_rate"] = fg_constraint_trigger_count / max(len(samples), 1)
        elem_metrics["functional_group_constraint_found_rate_when_triggered"] = (
            fg_constraint_found_count / max(fg_constraint_trigger_count, 1)
        )
        elem_metrics["functional_group_constraint_final_satisfied_rate_when_triggered"] = (
            fg_constraint_final_satisfied_count / max(fg_constraint_trigger_count, 1)
        )
        elem_metrics["functional_group_constraint_regeneration_rate_when_triggered"] = (
            fg_constraint_regeneration_count / max(fg_constraint_trigger_count, 1)
        )
        elem_metrics["functional_group_constraint_avg_required_groups_when_triggered"] = (
            fg_constraint_required_group_sum / max(fg_constraint_trigger_count, 1)
        )
    if smiles2ir_model is not None:
        elem_metrics["ir_rerank_top1_exact_match_rate"] = ir_rerank_top1_exact_match / max(len(samples), 1)
        elem_metrics["ir_rerank_top1_avg_similarity"] = ir_rerank_score_sum / max(ir_rerank_score_count, 1)
        elem_metrics["ir_rerank_switch_rate"] = ir_rerank_switch_count / max(len(samples), 1)
        elem_metrics["ir_rerank_tanimoto_mean"] = ir_rerank_tanimoto_sum / max(len(samples), 1)
        elem_metrics["ir_rerank_gain_vs_beam_top1"] = (
            elem_metrics["ir_rerank_top1_exact_match_rate"] - elem_metrics["beam_top1_exact_match_rate"]
        )
    elem_metrics["topk_all_wrong_rate"] = len(topk_all_wrong_samples) / max(len(samples), 1)
    elem_metrics["topk_all_wrong_count"] = len(topk_all_wrong_samples)
    return em_rate, samples, elem_metrics, topk_all_wrong_samples


def export_topk_failures_to_excel(
    failures: List[Dict[str, object]],
    output_path: str,
    top_k: int,
) -> Tuple[str, str]:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        from openpyxl import Workbook
    except Exception as exc:
        csv_path = str(Path(output_path).with_suffix(".csv"))
        summary_headers = [
            "sample_index",
            "gt",
            "greedy_pred",
            "beam_top1_pred",
            "consensus_pred",
            "ir_rerank_pred",
            "consensus_cluster_size",
            "consensus_cluster_internal_tanimoto",
            "required_functional_groups",
            "required_functional_group_probs",
            "functional_group_constraint_triggered",
            "functional_group_constraint_found",
            "functional_group_constraint_regenerated",
            "functional_group_constraint_final_satisfied",
            "functional_group_constraint_beam_size",
        ]
        for rank in range(1, top_k + 1):
            summary_headers.extend(
                [
                    f"top{rank}_smiles",
                    f"top{rank}_beam_score",
                    f"top{rank}_tanimoto_to_gt",
                    f"top{rank}_avg_top10_tanimoto",
                    f"top{rank}_in_consensus_cluster",
                    f"top{rank}_fg_ok",
                ]
            )

        import csv

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(summary_headers)
            for item in failures:
                row = [
                    item["sample_index"],
                    item["gt"],
                    item["greedy_pred"],
                    item["beam_top1_pred"],
                    item["consensus_pred"],
                    item["ir_rerank_pred"],
                    item["consensus_cluster_size"],
                    item["consensus_cluster_internal_tanimoto"],
                    ",".join(item.get("required_functional_groups", [])),
                    json.dumps(item.get("required_functional_group_probs", {}), ensure_ascii=False),
                    item["functional_group_constraint_triggered"],
                    item["functional_group_constraint_found"],
                    item["functional_group_constraint_regenerated"],
                    item["functional_group_constraint_final_satisfied"],
                    item["functional_group_constraint_beam_size"],
                ]
                candidates = item.get("topk_candidates", [])
                for rank in range(top_k):
                    if rank < len(candidates):
                        cand = candidates[rank]
                        row.extend(
                            [
                                cand.get("smiles", ""),
                                cand.get("beam_score", ""),
                                cand.get("tanimoto_to_gt", ""),
                                cand.get("avg_top10_tanimoto", ""),
                                cand.get("in_consensus_cluster", False),
                                cand.get("satisfies_required_functional_groups", True),
                            ]
                        )
                    else:
                        row.extend(["", "", "", "", "", ""])
                writer.writerow(row)
        return csv_path, f"openpyxl unavailable, exported CSV instead ({exc})"

    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "summary"
    candidate_sheet = workbook.create_sheet(title="topk_candidates")

    summary_headers = [
        "sample_index",
        "gt",
        "greedy_pred",
        "beam_top1_pred",
        "consensus_pred",
        "ir_rerank_pred",
        "consensus_cluster_size",
        "consensus_cluster_internal_tanimoto",
        "required_functional_groups",
        "required_functional_group_probs",
        "functional_group_constraint_triggered",
        "functional_group_constraint_found",
        "functional_group_constraint_regenerated",
        "functional_group_constraint_final_satisfied",
        "functional_group_constraint_beam_size",
    ]
    for rank in range(1, top_k + 1):
        summary_headers.extend(
            [
                f"top{rank}_smiles",
                f"top{rank}_beam_score",
                f"top{rank}_tanimoto_to_gt",
                f"top{rank}_avg_top10_tanimoto",
                f"top{rank}_in_consensus_cluster",
                f"top{rank}_fg_ok",
            ]
        )
    summary_sheet.append(summary_headers)

    candidate_sheet.append(
        [
            "sample_index",
            "gt",
            "rank",
            "smiles",
            "beam_score",
            "tanimoto_to_gt",
            "avg_top10_tanimoto",
            "in_consensus_cluster",
            "satisfies_required_functional_groups",
        ]
    )

    for item in failures:
        row = [
            item["sample_index"],
            item["gt"],
            item["greedy_pred"],
            item["beam_top1_pred"],
            item["consensus_pred"],
            item["ir_rerank_pred"],
            item["consensus_cluster_size"],
            item["consensus_cluster_internal_tanimoto"],
            ",".join(item.get("required_functional_groups", [])),
            json.dumps(item.get("required_functional_group_probs", {}), ensure_ascii=False),
            item["functional_group_constraint_triggered"],
            item["functional_group_constraint_found"],
            item["functional_group_constraint_regenerated"],
            item["functional_group_constraint_final_satisfied"],
            item["functional_group_constraint_beam_size"],
        ]
        candidates = item.get("topk_candidates", [])
        for rank in range(top_k):
            if rank < len(candidates):
                cand = candidates[rank]
                row.extend(
                    [
                        cand.get("smiles", ""),
                        cand.get("beam_score", ""),
                        cand.get("tanimoto_to_gt", ""),
                        cand.get("avg_top10_tanimoto", ""),
                        cand.get("in_consensus_cluster", False),
                        cand.get("satisfies_required_functional_groups", True),
                    ]
                )
                candidate_sheet.append(
                    [
                        item["sample_index"],
                        item["gt"],
                        cand.get("rank", rank + 1),
                        cand.get("smiles", ""),
                        cand.get("beam_score", ""),
                        cand.get("tanimoto_to_gt", ""),
                        cand.get("avg_top10_tanimoto", ""),
                        cand.get("in_consensus_cluster", False),
                        cand.get("satisfies_required_functional_groups", True),
                    ]
                )
            else:
                row.extend(["", "", "", "", "", ""])
        summary_sheet.append(row)

    workbook.save(output_path)
    return output_path, "xlsx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Transformer IR+Formula -> SMILES model.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--vocab-path", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--split-path", type=str, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--split", type=str, choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gen-samples", type=int, default=10000)
    parser.add_argument("--print-samples", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=120)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--use-formula-constrained-beam", action="store_true")
    parser.add_argument("--constraint-ignore-elements", type=str, default="H")
    parser.add_argument("--constraint-require-exact-match", action="store_true")
    parser.add_argument("--use-ir-rerank", action="store_true")
    parser.add_argument("--smiles2ir-model-path", type=str, default=DEFAULT_SMILES2IR_MODEL_PATH)
    parser.add_argument("--smiles2ir-config-path", type=str, default=DEFAULT_SMILES2IR_CONFIG_PATH)
    parser.add_argument("--rerank-similarity", type=str, choices=["cosine", "neg_mse"], default="cosine")
    parser.add_argument("--rerank-mode", type=str, choices=["conservative", "hybrid", "ir_only"], default="conservative")
    parser.add_argument("--rerank-ir-weight", type=float, default=0.25)
    parser.add_argument("--rerank-min-abs-sim", type=float, default=0.80)
    parser.add_argument("--rerank-min-delta-sim", type=float, default=0.03)
    parser.add_argument("--rerank-top-m", type=int, default=3)
    parser.add_argument("--beam-display-top-k", type=int, default=10)
    parser.add_argument("--consensus-sim-threshold", type=float, default=0.55)
    parser.add_argument("--consensus-min-cluster-size", type=int, default=3)
    parser.add_argument("--use-functional-group-constraint", action="store_true")
    parser.add_argument("--functional-group-model-path", type=str, default=DEFAULT_FUNCTIONAL_GROUP_MODEL_PATH)
    parser.add_argument("--functional-group-config-path", type=str, default=DEFAULT_FUNCTIONAL_GROUP_CONFIG_PATH)
    parser.add_argument("--functional-group-threshold", type=float, default=0.95)
    parser.add_argument("--functional-group-beam-multiplier", type=float, default=2.0)
    parser.add_argument("--functional-group-max-beam-size", type=int, default=40)
    parser.add_argument("--functional-group-max-regenerations", type=int, default=2)
    parser.add_argument("--save-top10-failures", action="store_true")
    parser.add_argument(
        "--top10-failures-path",
        type=str,
        default="checkpoints/TransformerModel/top10_all_wrong.xlsx",
    )
    parser.add_argument("--seed", type=int, default=42)
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

    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = build_model(args.model_path, config, device)
    smiles2ir_model = None
    smiles2ir_max_len = 256
    if args.use_ir_rerank:
        smiles2ir_config = load_config(args.smiles2ir_config_path)
        smiles2ir_model = build_smiles2ir_model(
            args.smiles2ir_model_path,
            smiles2ir_config,
            dataset_formula_dim=full_dataset.formula_dim,
            device=device,
        )
        smiles2ir_max_len = int(smiles2ir_config.get("max_len", 256))

    functional_group_model = None
    functional_group_label_names = None
    if args.use_functional_group_constraint:
        functional_group_model, functional_group_label_names = build_functional_group_model(
            args.functional_group_model_path,
            args.functional_group_config_path,
            device=device,
        )

    avg_loss, ppl, token_acc, topk_acc = evaluate_teacher_forcing(
        model,
        loader,
        device,
        pad_id=config["pad_id"],
        topk=args.topk,
    )

    em_rate, pairs, elem_metrics, top10_failures = evaluate_generation(
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
        use_formula_constrained_beam=args.use_formula_constrained_beam,
        constraint_ignore_elements=ignore_elements,
        constraint_require_exact_match=args.constraint_require_exact_match,
        smiles2ir_model=smiles2ir_model,
        smiles2ir_max_len=smiles2ir_max_len,
        rerank_similarity=args.rerank_similarity,
        rerank_mode=args.rerank_mode,
        rerank_ir_weight=args.rerank_ir_weight,
        rerank_min_abs_similarity=args.rerank_min_abs_sim,
        rerank_min_delta_similarity=args.rerank_min_delta_sim,
        rerank_top_m=args.rerank_top_m,
        beam_display_top_k=args.beam_display_top_k,
        consensus_similarity_threshold=args.consensus_sim_threshold,
        consensus_min_cluster_size=args.consensus_min_cluster_size,
        functional_group_model=functional_group_model,
        functional_group_label_names=functional_group_label_names,
        functional_group_threshold=args.functional_group_threshold,
        functional_group_beam_multiplier=args.functional_group_beam_multiplier,
        functional_group_max_beam_size=args.functional_group_max_beam_size,
        functional_group_max_regenerations=args.functional_group_max_regenerations,
    )

    exported_failure_path = None
    exported_failure_note = None
    if args.save_top10_failures:
        exported_failure_path, exported_failure_note = export_topk_failures_to_excel(
            failures=top10_failures,
            output_path=args.top10_failures_path,
            top_k=args.beam_display_top_k,
        )

    print("=" * 70)
    print("Transformer model evaluation finished")
    print(f"Device                 : {device}")
    print(f"Model checkpoint       : {args.model_path}")
    print(f"Formula constrained beam: {args.use_formula_constrained_beam}")
    print(f"RDKit available         : {Chem is not None}")
    if args.use_formula_constrained_beam:
        print(f"Constraint ignore elems: {sorted(ignore_elements)}")
        print(f"Constraint exact match : {args.constraint_require_exact_match}")
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
    if args.use_functional_group_constraint:
        print(f"FG constraint ckpt     : {args.functional_group_model_path}")
        print(f"FG threshold           : {args.functional_group_threshold}")
        print(f"FG beam multiplier     : {args.functional_group_beam_multiplier}")
        print(f"FG max beam size       : {args.functional_group_max_beam_size}")
        print(f"FG max regenerations   : {args.functional_group_max_regenerations}")
    print(f"Consensus top-k        : {args.beam_display_top_k}")
    print(f"Consensus sim threshold: {args.consensus_sim_threshold}")
    print(f"Consensus min cluster  : {args.consensus_min_cluster_size}")
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
    print(f"Greedy tanimoto mean   : {elem_metrics['greedy_tanimoto_mean']:.6f}")
    print(f"Beam top-1 exact       : {elem_metrics['beam_top1_exact_match_rate']:.4%}")
    print(f"Beam top-1 tanimoto    : {elem_metrics['beam_top1_tanimoto_mean']:.6f}")
    print(f"Beam top-{args.beam_size} exact  : {elem_metrics['beam_topk_exact_match_rate']:.4%}")
    print(f"Beam top-{args.beam_display_top_k} best tanimoto : {elem_metrics['beam_topk_best_tanimoto_mean']:.6f}")
    print(f"Top-{args.beam_display_top_k} all-wrong rate : {elem_metrics['topk_all_wrong_rate']:.4%}")
    print(f"Top-{args.beam_display_top_k} all-wrong count: {elem_metrics['topk_all_wrong_count']}")
    print(f"Consensus top-1 exact  : {elem_metrics['consensus_top1_exact_match_rate']:.4%}")
    print(f"Consensus tanimoto     : {elem_metrics['consensus_top1_tanimoto_mean']:.6f}")
    print(f"Consensus cluster size : {elem_metrics['consensus_cluster_size_mean']:.4f}")
    print(f"Consensus cluster sim  : {elem_metrics['consensus_cluster_internal_tanimoto_mean']:.6f}")
    if args.use_functional_group_constraint:
        print(f"FG trigger rate        : {elem_metrics['functional_group_constraint_trigger_rate']:.4%}")
        print(f"FG found rate          : {elem_metrics['functional_group_constraint_found_rate_when_triggered']:.4%}")
        print(f"FG final satisfied     : {elem_metrics['functional_group_constraint_final_satisfied_rate_when_triggered']:.4%}")
        print(f"FG regeneration rate   : {elem_metrics['functional_group_constraint_regeneration_rate_when_triggered']:.4%}")
        print(f"FG avg required groups : {elem_metrics['functional_group_constraint_avg_required_groups_when_triggered']:.4f}")
    if args.use_ir_rerank:
        print(f"IR-rerank top-1 exact  : {elem_metrics['ir_rerank_top1_exact_match_rate']:.4%}")
        print(f"IR-rerank avg score    : {elem_metrics['ir_rerank_top1_avg_similarity']:.6f}")
        print(f"IR-rerank tanimoto     : {elem_metrics['ir_rerank_tanimoto_mean']:.6f}")
        print(f"IR-rerank switch rate  : {elem_metrics['ir_rerank_switch_rate']:.4%}")
        print(f"IR-rerank gain vs beam : {elem_metrics['ir_rerank_gain_vs_beam_top1']:+.4%}")
    print(f"Element set match      : {elem_metrics['element_set_match_rate']:.4%}")
    print(f"Element count exact    : {elem_metrics['element_count_exact_match_rate']:.4%}")
    print(f"Element count MAE      : {elem_metrics['element_count_mae']:.6f}")
    if exported_failure_path is not None:
        print(f"Top-{args.beam_display_top_k} failures file : {exported_failure_path}")
        if exported_failure_note not in {None, 'xlsx'}:
            print(f"Top-{args.beam_display_top_k} failures note : {exported_failure_note}")
    print("=" * 70)

    show_n = min(args.print_samples, len(pairs))
    if show_n > 0:
        print("\nSample predictions:")
        for i in range(show_n):
            sample = pairs[i]
            print(f"[{i + 1}] GT         : {sample['gt']}")
            print(f"    Greedy     : {sample['greedy_pred']}")
            print(f"    BeamTop1   : {sample['beam_top1_pred']}")
            print(f"    Consensus  : {sample['consensus_pred']}")
            if args.use_ir_rerank:
                print(f"    IR-Rerank  : {sample['ir_rerank_pred']}")
            print(
                f"    Cluster    : size={sample['consensus_cluster_size']} "
                f"avg_sim={sample['consensus_cluster_internal_tanimoto']:.4f}"
            )
            if args.use_functional_group_constraint:
                req = sample["required_functional_groups"]
                req_scores = sample["required_functional_group_probs"]
                req_text = ", ".join([f"{name}({req_scores[name]:.2f})" for name in req]) if req else "(none)"
                print(f"    FG Req     : {req_text}")
                print(
                    f"    FG Status  : triggered={sample['functional_group_constraint_triggered']} "
                    f"found={sample['functional_group_constraint_found']} "
                    f"regenerated={sample['functional_group_constraint_regenerated']} "
                    f"final_ok={sample['functional_group_constraint_final_satisfied']} "
                    f"beam={sample['functional_group_constraint_beam_size']}"
                )
            print("    Top-10:")
            for cand in sample["top10_candidates"]:
                tag = " [CONSENSUS]" if cand["in_consensus_cluster"] else ""
                fg_tag = " [FG-OK]" if cand.get("satisfies_required_functional_groups", True) else " [FG-MISS]"
                print(
                    f"      {cand['rank']:>2}. {cand['smiles']} | "
                    f"beam={cand['beam_score']:.4f} "
                    f"tanimoto_gt={cand['tanimoto_to_gt']:.4f} "
                    f"avg_top10={cand['avg_top10_tanimoto']:.4f}{tag}{fg_tag}"
                )


if __name__ == "__main__":
    main()


































