import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from TransformerModel.TransformerModel import ConvFeatureEncoder


class BiLSTMDecoder(nn.Module):
    # NOTE: Unidirectional LSTM for autoregressive decoding; name kept for compatibility.
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_tgt_len: int = 256,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_tgt_len = max_tgt_len
        self.pad_id = pad_id

        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else d_model // 2
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.out_dim = self.hidden_dim

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_pos_emb = nn.Embedding(max_tgt_len, d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attn_proj = nn.Linear(self.out_dim, d_model) if self.out_dim != d_model else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.fc_out = nn.Sequential(
            nn.Linear(self.out_dim + d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, memory: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        # memory: [B, S, D], tgt_ids: [B, T]
        bsz, tgt_len = tgt_ids.size()
        if tgt_len > self.max_tgt_len:
            raise ValueError(f"tgt_len {tgt_len} > max_tgt_len {self.max_tgt_len}")

        pos_ids = torch.arange(tgt_len, device=tgt_ids.device).unsqueeze(0).expand(bsz, tgt_len)
        tgt_emb = self.token_emb(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = self.norm(self.dropout(tgt_emb + self.tgt_pos_emb(pos_ids)))

        dec_out, _ = self.lstm(tgt_emb)  # [B, T, H]
        dec_for_attn = self.attn_proj(dec_out)
        attn_scores = torch.matmul(dec_for_attn, memory.transpose(1, 2)) / math.sqrt(self.d_model)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, memory)

        combined = torch.cat([dec_out, context], dim=-1)
        logits = self.fc_out(combined)
        return logits


class IRFormulaBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        formula_dim: int,
        input_points: int = 1652,
        d_model: int = 512,
        nhead: int = 8,
        decoder_hidden_dim: Optional[int] = None,
        decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_tgt_len: int = 256,
        max_memory_len: int = 1024,
        encoder_buffer_layers: int = 2,
        encoder_buffer_dim_feedforward: Optional[int] = None,
        encoder_multiscale_target: str = "mid",
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        buffer_ffn = dim_feedforward if encoder_buffer_dim_feedforward is None else int(encoder_buffer_dim_feedforward)
        self.encoder = ConvFeatureEncoder(
            formula_dim=formula_dim,
            d_model=d_model,
            max_memory_len=max_memory_len,
            input_points=input_points,
            nhead=nhead,
            buffer_layers=encoder_buffer_layers,
            buffer_dim_feedforward=buffer_ffn,
            dropout=dropout,
            multiscale_target=encoder_multiscale_target,
        )

        self.decoder = BiLSTMDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_layers,
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
