import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAddNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([x, sublayer_out], dim=-1)))
        out = x + self.dropout(g * sublayer_out)
        return self.norm(out)


class PatchEmbedding1D(nn.Module):
    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        bsz, _, length = x.shape
        pad_len = (self.patch_size - (length % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode="constant", value=0.0)
        patches = self.proj(x)  # [B, D, N]
        return patches.transpose(1, 2)  # [B, N, D]


def build_nonuniform_patch_spans(
    input_points: int,
    fingerprint_start_idx: int,
    fingerprint_end_idx: int,
    fingerprint_patch_size: int,
    non_fingerprint_patch_size: int,
) -> List[Tuple[int, int]]:
    if input_points <= 0:
        raise ValueError(f"input_points must be positive, got {input_points}")
    if fingerprint_patch_size <= 0:
        raise ValueError(f"fingerprint_patch_size must be positive, got {fingerprint_patch_size}")
    if non_fingerprint_patch_size <= 0:
        raise ValueError(f"non_fingerprint_patch_size must be positive, got {non_fingerprint_patch_size}")

    start = max(0, min(int(fingerprint_start_idx), input_points))
    end = max(0, min(int(fingerprint_end_idx), input_points))
    if start > end:
        start, end = end, start

    spans: List[Tuple[int, int]] = []

    def add_range(range_start: int, range_end: int, patch_size: int):
        pos = range_start
        while pos < range_end:
            nxt = min(pos + patch_size, range_end)
            spans.append((pos, nxt))
            pos = nxt

    add_range(0, start, non_fingerprint_patch_size)
    add_range(start, end, fingerprint_patch_size)
    add_range(end, input_points, non_fingerprint_patch_size)
    if not spans:
        spans.append((0, input_points))
    return spans


class NonUniformPatchEmbedding1D(nn.Module):
    def __init__(self, patch_spans: List[Tuple[int, int]], d_model: int, patch_token_len: int = 16):
        super().__init__()
        if len(patch_spans) == 0:
            raise ValueError("patch_spans must not be empty")
        if patch_token_len <= 0:
            raise ValueError(f"patch_token_len must be positive, got {patch_token_len}")

        self.patch_spans = [(int(s), int(e)) for s, e in patch_spans]
        self.patch_token_len = int(patch_token_len)
        self.proj = nn.Linear(self.patch_token_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L] or [B, 1, L]
        if x.dim() == 3:
            if x.size(1) != 1:
                raise ValueError(f"Expected channel size 1 for 3D input, got {x.size(1)}")
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Expected input shape [B, L], got {tuple(x.shape)}")

        parts: List[torch.Tensor] = []
        for start, end in self.patch_spans:
            seg = x[:, start:end].unsqueeze(1)  # [B, 1, seg_len]
            if seg.size(-1) != self.patch_token_len:
                seg = F.interpolate(seg, size=self.patch_token_len, mode="linear", align_corners=False)
            parts.append(seg.squeeze(1))  # [B, patch_token_len]

        patch_mat = torch.stack(parts, dim=1)  # [B, N, patch_token_len]
        return self.proj(patch_mat)  # [B, N, D]


class BartLikeEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.addnorm1 = GatedAddNorm(d_model, dropout)
        self.addnorm2 = GatedAddNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.addnorm1(x, attn_out)
        ffn_out = self.ffn(x)
        x = self.addnorm2(x, ffn_out)
        return x


class BartLikeDecoderLayer(nn.Module):
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
        self.addnorm1 = GatedAddNorm(d_model, dropout)
        self.addnorm2 = GatedAddNorm(d_model, dropout)
        self.addnorm3 = GatedAddNorm(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_attn_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_attn_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = self.addnorm1(x, self_out)

        cross_out, _ = self.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = self.addnorm2(x, cross_out)

        ffn_out = self.ffn(x)
        x = self.addnorm3(x, ffn_out)
        return x


class BARTPatchModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        formula_dim: int,
        input_points: int = 1652,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        patch_size: int = 16,
        use_nonuniform_patch: bool = False,
        fingerprint_start_idx: Optional[int] = None,
        fingerprint_end_idx: Optional[int] = None,
        fingerprint_patch_size: Optional[int] = None,
        non_fingerprint_patch_size: Optional[int] = None,
        nonuniform_patch_token_len: int = 16,
        max_tgt_len: int = 256,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.formula_dim = formula_dim
        self.input_points = input_points
        self.d_model = d_model
        self.patch_size = patch_size
        self.use_nonuniform_patch = bool(use_nonuniform_patch)
        self.max_tgt_len = max_tgt_len
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        if self.use_nonuniform_patch:
            default_fp_start = int(round(input_points * 0.68))
            default_fp_end = int(round(input_points * 0.97))
            fp_start = default_fp_start if fingerprint_start_idx is None else int(fingerprint_start_idx)
            fp_end = default_fp_end if fingerprint_end_idx is None else int(fingerprint_end_idx)
            fp_patch = max(1, patch_size // 2) if fingerprint_patch_size is None else int(fingerprint_patch_size)
            non_fp_patch = patch_size if non_fingerprint_patch_size is None else int(non_fingerprint_patch_size)

            self.fingerprint_start_idx = fp_start
            self.fingerprint_end_idx = fp_end
            self.fingerprint_patch_size = fp_patch
            self.non_fingerprint_patch_size = non_fp_patch
            self.nonuniform_patch_token_len = int(nonuniform_patch_token_len)

            self.patch_spans = build_nonuniform_patch_spans(
                input_points=input_points,
                fingerprint_start_idx=fp_start,
                fingerprint_end_idx=fp_end,
                fingerprint_patch_size=fp_patch,
                non_fingerprint_patch_size=non_fp_patch,
            )
            self.patch_embed = NonUniformPatchEmbedding1D(
                patch_spans=self.patch_spans,
                d_model=d_model,
                patch_token_len=self.nonuniform_patch_token_len,
            )
            max_patches = len(self.patch_spans)
        else:
            self.fingerprint_start_idx = None
            self.fingerprint_end_idx = None
            self.fingerprint_patch_size = None
            self.non_fingerprint_patch_size = None
            self.nonuniform_patch_token_len = None
            self.patch_spans = None
            self.patch_embed = PatchEmbedding1D(patch_size=patch_size, d_model=d_model)
            max_patches = math.ceil(input_points / patch_size)

        self.formula_proj = nn.Sequential(
            nn.Linear(formula_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.mem_pos_emb = nn.Embedding(max_patches + 1, d_model)
        self.tgt_token_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_pos_emb = nn.Embedding(max_tgt_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [
                BartLikeEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                BartLikeDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, ir_spectrum: torch.Tensor, formula_vec: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(ir_spectrum)  # [B, N, D]
        formula_token = self.formula_proj(formula_vec).unsqueeze(1)  # [B, 1, D]
        memory = torch.cat([formula_token, patches], dim=1)

        mem_len = memory.size(1)
        pos_ids = torch.arange(mem_len, device=memory.device).unsqueeze(0)
        memory = self.dropout(memory + self.mem_pos_emb(pos_ids))

        for layer in self.encoder_layers:
            memory = layer(memory)
        return self.enc_norm(memory)

    def _build_causal_mask(self, tgt_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), diagonal=1)

    def decode(self, memory: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        bsz, tgt_len = tgt_ids.shape
        if tgt_len > self.max_tgt_len:
            raise ValueError(f"tgt_len={tgt_len} exceeds max_tgt_len={self.max_tgt_len}")

        pos_ids = torch.arange(tgt_len, device=tgt_ids.device).unsqueeze(0).expand(bsz, tgt_len)
        x = self.tgt_token_emb(tgt_ids) * math.sqrt(self.d_model)
        x = self.dropout(x + self.tgt_pos_emb(pos_ids))

        tgt_attn_mask = self._build_causal_mask(tgt_len, tgt_ids.device)
        tgt_key_padding_mask = tgt_ids.eq(self.pad_id)

        for layer in self.decoder_layers:
            x = layer(
                x,
                memory,
                tgt_attn_mask=tgt_attn_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=None,
            )

        x = self.dec_norm(x)
        return self.fc_out(x)

    def forward(self, ir_spectrum: torch.Tensor, formula_vec: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        memory = self.encode(ir_spectrum, formula_vec)
        logits = self.decode(memory, target_ids)
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

        memory = self.encode(ir_spectrum, formula_vec)
        bsz = ir_spectrum.size(0)
        generated = torch.full((bsz, 1), sos_id, dtype=torch.long, device=ir_spectrum.device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=ir_spectrum.device)

        for _ in range(max_len - 1):
            logits = self.decode(memory, generated)
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(eos_id)
            if torch.all(finished):
                break

        return generated
