import json
import re

import torch

try:
    from rdkit import Chem
except Exception:
    Chem = None


class SMILESTokenizer:
    def __init__(self, smiles_list):
        self.token_pattern = (
            r"(\[[^\]]+\]|Br?|Cl?|C|N|O|P|S|F|I|b|n|o|s|p|c|\(|\)|\."
            r"|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        )
        self.regex = re.compile(self.token_pattern)

        all_tokens = []
        for s in smiles_list:
            all_tokens.extend(self.tokenize(s))

        unique_chars = sorted(list(set(all_tokens)))

        self.vocab = {char: i + 4 for i, char in enumerate(unique_chars)}
        self.vocab["<PAD>"] = 0
        self.vocab["<SOS>"] = 1
        self.vocab["<EOS>"] = 2
        self.vocab["<UNK>"] = 3
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, smiles):
        return [token for token in self.regex.findall(smiles)]

    def encode(self, smiles, max_len=None):
        tokens = self.tokenize(smiles)
        ids = [self.vocab["<SOS>"]]
        for t in tokens:
            ids.append(self.vocab.get(t, self.vocab["<UNK>"]))
        ids.append(self.vocab["<EOS>"])
        if max_len:
            if len(ids) < max_len:
                ids += [self.vocab["<PAD>"]] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
        return torch.tensor(ids)

    def decode(self, ids):
        res = []
        for i in ids:
            token = self.inv_vocab.get(i.item(), "")
            if token == "<EOS>":
                break
            if token not in ["<SOS>", "<PAD>", "<UNK>"]:
                res.append(token)
        return "".join(res)


def _canonicalize_smiles(smiles: str) -> str:
    if Chem is None or not isinstance(smiles, str):
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def build_vocab(datapath, canonicalize_smiles: bool = False):
    data = torch.load(datapath)
    smiles_list = data["smiles"]
    if canonicalize_smiles and Chem is not None:
        smiles_list = [_canonicalize_smiles(s) for s in smiles_list]
    print(f"Loaded {len(smiles_list)} SMILES entries for vocab.")
    tokenizer = SMILESTokenizer(smiles_list)
    print(f"Vocab size: {len(tokenizer.vocab)}")
    with open("pure/vocab.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer.vocab, f)
    print("Saved vocab to pure/vocab.json")
