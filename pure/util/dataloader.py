import os
import random
import re
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

try:
    from rdkit import Chem
except Exception:
    Chem = None

try:
    from FunctionalGroupModel.IRFunctionalGroupModel import DEFAULT_FUNCTIONAL_GROUPS
    from FunctionalGroupModel.functional_group_utils import extract_functional_group_labels
except Exception:
    DEFAULT_FUNCTIONAL_GROUPS = ()
    extract_functional_group_labels = None


FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula_to_counts(formula: str) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    if not isinstance(formula, str):
        return counts
    for elem, num_str in FORMULA_PATTERN.findall(formula):
        num = float(num_str) if num_str else 1.0
        counts[elem] = counts.get(elem, 0.0) + num
    return counts


def build_formula_vocab(formulas: List[str]) -> Dict[str, int]:
    elem_set = set()
    for formula in formulas:
        elem_set.update(parse_formula_to_counts(formula).keys())
    return {elem: idx for idx, elem in enumerate(sorted(elem_set))}


class IRDataset(Dataset):
    def __init__(
        self,
        pt_path,
        tokenizer,
        formula_vocab: Optional[Dict[str, int]] = None,
        data: Optional[Dict] = None,
        canonicalize_smiles: bool = False,
        randomize_smiles: bool = False,
        randomize_prob: float = 0.0,
        seed: int = 42,
        return_functional_groups: bool = False,
        functional_group_labels: Optional[torch.Tensor] = None,
        functional_group_label_names: Optional[Sequence[str]] = None,
        functional_group_cache_path: Optional[str] = None,
    ):
        self.data = data if data is not None else torch.load(pt_path)
        self.tokenizer = tokenizer
        self.canonicalize_smiles = bool(canonicalize_smiles)
        self.randomize_smiles = bool(randomize_smiles)
        self.randomize_prob = float(randomize_prob)
        self._rng = random.Random(seed)
        self._use_rdkit = Chem is not None
        self.return_functional_groups = bool(return_functional_groups)
        self.functional_group_label_names = tuple(
            functional_group_label_names if functional_group_label_names is not None else DEFAULT_FUNCTIONAL_GROUPS
        )

        if not isinstance(self.data, dict):
            raise ValueError(f"Unexpected data format: {type(self.data)}")
        if "molecular_formula" not in self.data:
            raise KeyError("molecular_formula not found in dataset")

        formulas = self.data["molecular_formula"]
        self.formula_vocab = formula_vocab if formula_vocab is not None else build_formula_vocab(formulas)
        self.formula_dim = len(self.formula_vocab)

        self._canonical_cache = None
        if self.canonicalize_smiles and not self.randomize_smiles and self._use_rdkit:
            smiles_list = self.data.get("smiles", [])
            self._canonical_cache = [self._canonicalize_smiles(s) for s in smiles_list]

        self.functional_group_labels = None
        if self.return_functional_groups:
            self.functional_group_labels = self._get_functional_group_labels(
                provided_labels=functional_group_labels,
                cache_path=functional_group_cache_path,
            )

    def __len__(self):
        return len(self.data["smiles"])

    def __getitem__(self, index):
        smiles = self.data["smiles"][index]
        ir_spectra = self.data["ir_spectra"][index]
        formula = self.data["molecular_formula"][index]

        if self._canonical_cache is not None:
            smiles = self._canonical_cache[index]
        else:
            smiles = self._normalize_smiles(smiles)

        if hasattr(self.tokenizer, "encode"):
            encoded_smiles = self.tokenizer.encode(smiles)
        else:
            encoded_smiles = self.encode_smiles(smiles, self.tokenizer)

        formula_vec = self.encode_formula(formula)
        if self.functional_group_labels is not None:
            return ir_spectra, formula_vec, encoded_smiles, self.functional_group_labels[index]
        return ir_spectra, formula_vec, encoded_smiles

    def encode_formula(self, formula: str) -> torch.Tensor:
        vec = torch.zeros(self.formula_dim, dtype=torch.float32)
        counts = parse_formula_to_counts(formula)
        for elem, cnt in counts.items():
            if elem in self.formula_vocab:
                vec[self.formula_vocab[elem]] = cnt
        return vec

    def _canonicalize_smiles(self, smiles: str) -> str:
        if not self._use_rdkit or not isinstance(smiles, str):
            return smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, canonical=True)

    def _randomize_smiles(self, smiles: str) -> str:
        if not self._use_rdkit or not isinstance(smiles, str):
            return smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, doRandom=True)

    def _normalize_smiles(self, smiles: str) -> str:
        if self.randomize_smiles and self.randomize_prob > 0 and self._rng.random() < self.randomize_prob:
            randomized = self._randomize_smiles(smiles)
            if isinstance(randomized, str) and len(randomized) > 0:
                return randomized
        if self.canonicalize_smiles:
            return self._canonicalize_smiles(smiles)
        return smiles

    def get_normalized_smiles_list(self) -> List[str]:
        if self._canonical_cache is not None:
            return list(self._canonical_cache)
        return [self._normalize_smiles(smiles) for smiles in self.data["smiles"]]

    def _get_functional_group_labels(
        self,
        provided_labels: Optional[torch.Tensor],
        cache_path: Optional[str],
    ) -> torch.Tensor:
        if provided_labels is not None:
            labels = provided_labels.detach().cpu().float()
            if labels.size(0) != len(self):
                raise ValueError(
                    f"functional_group_labels length {labels.size(0)} does not match dataset length {len(self)}"
                )
            return labels

        if extract_functional_group_labels is None:
            raise ImportError(
                "Functional-group labels requested but functional_group_utils could not be imported."
            )

        if cache_path and os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            if (
                isinstance(cached, dict)
                and tuple(cached.get("label_names", ())) == self.functional_group_label_names
                and int(cached.get("num_samples", -1)) == len(self)
            ):
                labels = cached.get("labels")
                if isinstance(labels, torch.Tensor) and labels.size(0) == len(self):
                    return labels.float()

        smiles_list = self.get_normalized_smiles_list()
        labels = torch.tensor(
            [extract_functional_group_labels(smiles, self.functional_group_label_names) for smiles in smiles_list],
            dtype=torch.float32,
        )

        if cache_path:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            torch.save(
                {
                    "label_names": list(self.functional_group_label_names),
                    "num_samples": len(self),
                    "labels": labels,
                },
                cache_path,
            )
        return labels

    def encode_smiles(self, smiles, vocab_dict):
        token_pattern = r"(\[[^\]]+\]|Br?|Cl?|C|N|O|P|S|F|I|b|n|o|s|p|c|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(token_pattern)

        tokens = regex.findall(smiles)
        ids = [vocab_dict["<SOS>"]]
        for token in tokens:
            ids.append(vocab_dict.get(token, vocab_dict["<UNK>"]))
        ids.append(vocab_dict["<EOS>"])

        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    spectra_list = []
    formula_list = []
    smiles_list = []
    functional_group_list = []
    has_functional_groups = len(batch[0]) >= 4

    for item in batch:
        spectra, formula_vec, smiles = item[:3]
        spectra_list.append(spectra)
        formula_list.append(formula_vec)
        smiles_list.append(smiles)
        if has_functional_groups:
            functional_group_list.append(item[3])

    spectra_batch = torch.stack(spectra_list, dim=0)
    formula_batch = torch.stack(formula_list, dim=0)

    max_len = max(seq.size(0) for seq in smiles_list)
    padded_smiles = []

    for seq in smiles_list:
        pad_length = max_len - seq.size(0)
        if pad_length > 0:
            padded_seq = torch.cat([seq, torch.full((pad_length,), 0, dtype=seq.dtype)], dim=0)
        else:
            padded_seq = seq
        padded_smiles.append(padded_seq)

    smiles_batch = torch.stack(padded_smiles, dim=0)
    if has_functional_groups:
        functional_group_batch = torch.stack(functional_group_list, dim=0)
        return spectra_batch, formula_batch, smiles_batch, functional_group_batch
    return spectra_batch, formula_batch, smiles_batch
