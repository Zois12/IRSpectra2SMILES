import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionDecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        q = self.norm1(q + self.dropout(self_out))

        cross_out, _ = self.cross_attn(
            q,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        q = self.norm2(q + self.dropout(cross_out))

        ffn_out = self.ffn(q)
        q = self.norm3(q + self.dropout(ffn_out))
        return q


class SMILES2IRRegressor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        output_points: int,
        formula_dim: int = 0,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.15,
        pad_id: int = 0,
        max_len: int = 256,
        spectral_decoder_layers: int = 3,
        decoder_query_len: int = 325,
        refine_channels: int = 64,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.output_points = int(output_points)
        self.formula_dim = int(formula_dim)
        self.d_model = int(d_model)
        self.pad_id = int(pad_id)
        self.max_len = int(max_len)
        self.decoder_query_len = int(decoder_query_len)

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        self.input_dropout = nn.Dropout(dropout)

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

        self.pool_query = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.spectral_queries = nn.Parameter(torch.randn(1, self.decoder_query_len, self.d_model) * 0.02)
        self.decoder_blocks = nn.ModuleList(
            [
                CrossAttentionDecoderBlock(
                    d_model=self.d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(spectral_decoder_layers)
            ]
        )
        self.coarse_out = nn.Linear(self.d_model, 1)
        self.global_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.output_points),
        )
        self.refine = nn.Sequential(
            nn.Conv1d(1, refine_channels, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(refine_channels, refine_channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(refine_channels, 1, kernel_size=5, padding=2),
        )

    def _encode_memory(
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
        x = self.input_dropout(x + self.pos_emb(pos_ids))
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

    def _attention_pool(self, memory: torch.Tensor, memory_pad_mask: torch.Tensor) -> torch.Tensor:
        # memory: [B, T, D]
        q = self.pool_query.expand(memory.size(0), -1, -1)  # [B, 1, D]
        scores = torch.matmul(q, memory.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)  # [B, T]
        scores = scores.masked_fill(memory_pad_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)  # [B, D]
        return pooled

    def forward(self, smiles_ids: torch.Tensor, formula_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        memory, memory_pad_mask = self._encode_memory(smiles_ids, formula_vec=formula_vec)
        pooled = self._attention_pool(memory, memory_pad_mask)

        q = self.spectral_queries.expand(memory.size(0), -1, -1)
        for block in self.decoder_blocks:
            q = block(q, memory, memory_key_padding_mask=memory_pad_mask)

        coarse = self.coarse_out(q).squeeze(-1)  # [B, Q]
        coarse_full = F.interpolate(
            coarse.unsqueeze(1),
            size=self.output_points,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

        global_pred = self.global_head(pooled)
        fused = coarse_full + global_pred
        refine_delta = self.refine(fused.unsqueeze(1)).squeeze(1)
        return fused + 0.5 * refine_delta

    @torch.no_grad()
    def predict_from_smiles_ids(
        self,
        smiles_ids: torch.Tensor,
        formula_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        return self.forward(smiles_ids, formula_vec=formula_vec)
