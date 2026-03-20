import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SMILESEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        formula_dim: int = 0,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.formula_dim = int(formula_dim)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.enc_norm = nn.LayerNorm(self.d_model)

        if self.formula_dim > 0:
            self.formula_proj = nn.Sequential(
                nn.Linear(self.formula_dim, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )
            self.missing_formula_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.formula_proj = None
            self.missing_formula_token = None

    def forward(
        self,
        smiles_ids: torch.Tensor,
        formula_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if smiles_ids.dim() != 2:
            raise ValueError(f"Expected smiles_ids shape [B, T], got {tuple(smiles_ids.shape)}")

        bsz, seq_len = smiles_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"smiles length {seq_len} exceeds max_len {self.max_len}")

        token_pad_mask = smiles_ids.eq(self.pad_id)  # [B, T]
        pos_ids = torch.arange(seq_len, device=smiles_ids.device).unsqueeze(0).expand(bsz, seq_len)

        x = self.token_emb(smiles_ids) * math.sqrt(self.d_model)
        x = self.input_norm(self.input_dropout(x + self.pos_emb(pos_ids)))
        memory = self.encoder(x, src_key_padding_mask=token_pad_mask)
        memory = self.enc_norm(memory)

        if self.formula_proj is not None:
            if formula_vec is not None:
                if formula_vec.dim() != 2 or formula_vec.size(0) != bsz:
                    raise ValueError(
                        f"Expected formula_vec shape [B, F], got {tuple(formula_vec.shape)} for batch {bsz}"
                    )
                formula_token = self.formula_proj(formula_vec).unsqueeze(1)
            else:
                formula_token = self.missing_formula_token.expand(bsz, 1, -1)

            memory = torch.cat([formula_token, memory], dim=1)
            formula_mask = torch.zeros((bsz, 1), dtype=torch.bool, device=smiles_ids.device)
            memory_pad_mask = torch.cat([formula_mask, token_pad_mask], dim=1)
        else:
            memory_pad_mask = token_pad_mask

        return memory, memory_pad_mask


class IRQueryDecoder(nn.Module):
    def __init__(
        self,
        output_points: int,
        query_len: Optional[int] = None,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_points = int(output_points)
        self.query_len = int(query_len) if query_len is not None else int(output_points)
        self.d_model = int(d_model)

        self.query_emb = nn.Embedding(self.query_len, self.d_model)
        self.query_norm = nn.LayerNorm(self.d_model)
        self.query_dropout = nn.Dropout(dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.dec_norm = nn.LayerNorm(self.d_model)
        self.out_proj = nn.Linear(self.d_model, 1)

    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = memory.size(0)
        pos_ids = torch.arange(self.query_len, device=memory.device)
        tgt = self.query_emb(pos_ids).unsqueeze(0).expand(bsz, -1, -1)
        tgt = self.query_norm(self.query_dropout(tgt))

        out = self.decoder(tgt=tgt, memory=memory, memory_key_padding_mask=memory_key_padding_mask)
        out = self.dec_norm(out)
        ir = self.out_proj(out).squeeze(-1)  # [B, Q]
        if self.query_len != self.output_points:
            ir = F.interpolate(ir.unsqueeze(1), size=self.output_points, mode="linear", align_corners=False).squeeze(1)
        return ir


class SMILES2IRTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        output_points: int,
        formula_dim: int = 0,
        d_model: int = 512,
        nhead: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 256,
        decoder_query_len: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = SMILESEncoder(
            vocab_size=vocab_size,
            formula_dim=formula_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            pad_id=pad_id,
        )
        self.decoder = IRQueryDecoder(
            output_points=output_points,
            query_len=decoder_query_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, smiles_ids: torch.Tensor, formula_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        memory, memory_pad_mask = self.encoder(smiles_ids, formula_vec=formula_vec)
        return self.decoder(memory, memory_key_padding_mask=memory_pad_mask)

    @torch.no_grad()
    def predict_from_smiles_ids(
        self,
        smiles_ids: torch.Tensor,
        formula_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        return self.forward(smiles_ids, formula_vec=formula_vec)
