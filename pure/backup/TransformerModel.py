import math
from typing import Optional

import torch
import torch.nn as nn


class ConvFeatureEncoder(nn.Module):
    """
    Encode IR spectrum + molecular formula into a memory sequence for Transformer decoder.
    """

    def __init__(
        self,
        formula_dim: int,
        d_model: int = 512,
        max_memory_len: int = 1024,
        input_points: int = 1652,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_memory_len = max_memory_len
        self.input_points = input_points

        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.formula_proj = nn.Sequential(
            nn.Linear(formula_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.memory_pos_emb = nn.Embedding(max_memory_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ir_spectrum: torch.Tensor, formula_vec: torch.Tensor) -> torch.Tensor:
        # ir_spectrum: [B, L]
        x = ir_spectrum.unsqueeze(1) if ir_spectrum.dim() == 2 else ir_spectrum
        feat = self.conv(x)  # [B, d_model, S]
        feat = feat.transpose(1, 2)  # [B, S, d_model]

        formula_token = self.formula_proj(formula_vec).unsqueeze(1)  # [B, 1, d_model]
        memory = torch.cat([formula_token, feat], dim=1)  # [B, S+1, d_model]

        mem_len = memory.size(1)
        if mem_len > self.max_memory_len:
            raise ValueError(f"memory length {mem_len} > max_memory_len {self.max_memory_len}")

        pos_ids = torch.arange(mem_len, device=memory.device).unsqueeze(0)
        memory = memory + self.memory_pos_emb(pos_ids)
        memory = self.norm(memory)
        return memory


class TransformerSMILESDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_tgt_len: int = 256,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.max_tgt_len = max_tgt_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_pos_emb = nn.Embedding(max_tgt_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((size, size), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, memory: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        # memory: [B, S, D], tgt_ids: [B, T]
        bsz, tgt_len = tgt_ids.size()
        if tgt_len > self.max_tgt_len:
            raise ValueError(f"tgt_len {tgt_len} > max_tgt_len {self.max_tgt_len}")

        pos_ids = torch.arange(tgt_len, device=tgt_ids.device).unsqueeze(0).expand(bsz, tgt_len)
        tgt_emb = self.token_emb(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.tgt_pos_emb(pos_ids)
        tgt_emb = self.norm(self.dropout(tgt_emb))

        tgt_mask = self._causal_mask(tgt_len, tgt_ids.device)
        tgt_key_padding_mask = tgt_ids.eq(self.pad_id)

        # transformer decoder expects [T, B, D]
        tgt = tgt_emb.transpose(0, 1)
        mem = memory.transpose(0, 1)

        decoded = self.decoder(
            tgt=tgt,
            memory=mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        decoded = decoded.transpose(0, 1)  # [B, T, D]
        logits = self.fc_out(decoded)  # [B, T, V]
        return logits


class IRFormulaTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        formula_dim: int,
        input_points: int = 1652,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_tgt_len: int = 256,
        max_memory_len: int = 1024,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.encoder = ConvFeatureEncoder(
            formula_dim=formula_dim,
            d_model=d_model,
            max_memory_len=max_memory_len,
            input_points=input_points,
        )
        self.decoder = TransformerSMILESDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_tgt_len=max_tgt_len,
            pad_id=pad_id,
        )

    def forward(self, ir_spectrum: torch.Tensor, formula_vec: torch.Tensor, target_smiles: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(ir_spectrum, formula_vec)
        logits = self.decoder(memory, target_smiles)
        return logits

    @torch.no_grad()
    def generate(
        self,
        ir_spectrum: torch.Tensor,
        formula_vec: torch.Tensor,
        max_len: int = 120,
        sos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        sos_id = self.sos_id if sos_id is None else sos_id
        eos_id = self.eos_id if eos_id is None else eos_id

        bsz = ir_spectrum.size(0)
        memory = self.encoder(ir_spectrum, formula_vec)

        generated = torch.full((bsz, 1), sos_id, dtype=torch.long, device=ir_spectrum.device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=ir_spectrum.device)

        for _ in range(max_len - 1):
            logits = self.decoder(memory, generated)
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(eos_id)
            if torch.all(finished):
                break

        return generated
