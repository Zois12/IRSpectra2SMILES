import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        ffn_type: str = "gelu",
    ):
        super().__init__()
        self.ffn_type = str(ffn_type).lower()
        if self.ffn_type not in {"gelu", "glu"}:
            raise ValueError(f"Unsupported transformer FFN type: {ffn_type}")

        hidden_in = dim_feedforward * 2 if self.ffn_type == "glu" else dim_feedforward
        self.linear1 = nn.Linear(d_model, hidden_in)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        if self.ffn_type == "glu":
            x = F.glu(x, dim=-1)
        else:
            x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GLUTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
        ffn_type: str = "glu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = TransformerFeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            ffn_type=ffn_type,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def _self_attention_block(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        return x

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.norm_first:
            x = x + self.dropout1(self._self_attention_block(self.norm1(x), src_mask, src_key_padding_mask))
            x = x + self.dropout2(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout1(self._self_attention_block(x, src_mask, src_key_padding_mask)))
            x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class GLUTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: GLUTransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(encoder_layer) for _ in range(num_layers))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return x


class GLUTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = False,
        ffn_type: str = "glu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = TransformerFeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            ffn_type=ffn_type,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def _self_attention_block(
        self,
        x: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        return x

    def _cross_attention_block(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x, _ = self.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        return x

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.norm_first:
            tgt = tgt + self.dropout1(self._self_attention_block(self.norm1(tgt), tgt_mask, tgt_key_padding_mask))
            tgt = tgt + self.dropout2(self._cross_attention_block(self.norm2(tgt), memory, memory_key_padding_mask))
            tgt = tgt + self.dropout3(self.ffn(self.norm3(tgt)))
        else:
            tgt = self.norm1(tgt + self.dropout1(self._self_attention_block(tgt, tgt_mask, tgt_key_padding_mask)))
            tgt = self.norm2(
                tgt + self.dropout2(self._cross_attention_block(tgt, memory, memory_key_padding_mask))
            )
            tgt = self.norm3(tgt + self.dropout3(self.ffn(tgt)))
        return tgt


class GLUTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: GLUTransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(decoder_layer) for _ in range(num_layers))

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return tgt


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
        dilation: int = 1,
    ):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out + residual)
        return out


class ResidualPatchCNN1D(nn.Module):
    """
    Lightweight residual CNN followed by explicit non-overlapping patching.
    Returns patched feature map [B, d_model, S].
    """

    def __init__(
        self,
        d_model: int = 512,
        dropout: float = 0.1,
        multiscale_target: str = "mid",
        use_coordconv: bool = False,
        patch_size: int = 4,
    ):
        super().__init__()
        del multiscale_target  # kept for backward-compatible constructor signature
        self.use_coordconv = bool(use_coordconv)
        self.patch_size = max(int(patch_size), 1)

        in_channels = 2 if self.use_coordconv else 1
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            ResidualBlock1D(64, 64, stride=1, dropout=dropout),
            ResidualBlock1D(64, 128, stride=2, dropout=dropout),
            ResidualBlock1D(128, 128, stride=1, dropout=dropout, dilation=2),
            ResidualBlock1D(128, d_model, stride=2, dropout=dropout, dilation=2),
        )
        self.patch_proj = nn.Sequential(
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=False,
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_coordconv:
            coord = torch.linspace(-1.0, 1.0, x.size(-1), device=x.device, dtype=x.dtype)
            coord = coord.unsqueeze(0).unsqueeze(0).expand(x.size(0), 1, -1)
            x = torch.cat([x, coord], dim=1)
        x = self.stem(x)
        x = self.blocks(x)
        if x.size(-1) < self.patch_size:
            x = F.pad(x, (0, self.patch_size - x.size(-1)))
        x = self.patch_proj(x)
        return x


class ConvFeatureEncoder(nn.Module):
    """
    Encode IR spectrum + optional molecular formula into a memory sequence.
    """

    def __init__(
        self,
        formula_dim: int,
        d_model: int = 512,
        max_memory_len: int = 1024,
        input_points: int = 1652,
        nhead: int = 8,
        buffer_layers: int = 2,
        buffer_dim_feedforward: int = 2048,
        dropout: float = 0.1,
        multiscale_target: str = "mid",
        use_coordconv: bool = False,
        patch_size: int = 4,
        transformer_ffn_type: str = "gelu",
        use_formula_input: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_memory_len = max_memory_len
        self.input_points = input_points
        self.buffer_layers = int(buffer_layers)
        self.patch_size = max(int(patch_size), 1)
        self.transformer_ffn_type = str(transformer_ffn_type).lower()
        self.use_formula_input = bool(use_formula_input)

        self.cnn = ResidualPatchCNN1D(
            d_model=d_model,
            dropout=dropout,
            multiscale_target=multiscale_target,
            use_coordconv=use_coordconv,
            patch_size=self.patch_size,
        )
        if self.use_formula_input:
            self.formula_proj = nn.Sequential(
                nn.Linear(formula_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.formula_proj = None

        self.memory_pos_emb = nn.Embedding(max_memory_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        if self.buffer_layers > 0:
            if self.transformer_ffn_type == "glu":
                encoder_layer = GLUTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=buffer_dim_feedforward,
                    dropout=dropout,
                    norm_first=True,
                    ffn_type=self.transformer_ffn_type,
                )
                self.buffer_encoder = GLUTransformerEncoder(encoder_layer, num_layers=self.buffer_layers)
            else:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=buffer_dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.buffer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.buffer_layers)
            self.buffer_norm = nn.LayerNorm(d_model)
        else:
            self.buffer_encoder = None
            self.buffer_norm = None

    def encode_spectrum(self, ir_spectrum: torch.Tensor) -> torch.Tensor:
        # ir_spectrum: [B, L]
        x = ir_spectrum.unsqueeze(1) if ir_spectrum.dim() == 2 else ir_spectrum
        feat = self.cnn(x)  # [B, d_model, S]
        return feat.transpose(1, 2)  # [B, S, d_model]

    def project_formula(self, formula_vec: torch.Tensor) -> torch.Tensor:
        if not self.use_formula_input or self.formula_proj is None:
            raise RuntimeError("Formula projection requested while use_formula_input=False.")
        return self.formula_proj(formula_vec).unsqueeze(1)  # [B, 1, d_model]

    def build_memory(
        self,
        spectral_tokens: torch.Tensor,
        formula_vec: Optional[torch.Tensor],
        extra_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = []
        prefix_len = 0
        if self.use_formula_input:
            if formula_vec is None:
                raise ValueError("formula_vec must be provided when use_formula_input=True")
            formula_token = self.project_formula(formula_vec)
            tokens.append(formula_token)
            prefix_len = 1
        if extra_tokens is not None:
            if extra_tokens.dim() != 3 or extra_tokens.size(0) != spectral_tokens.size(0) or extra_tokens.size(2) != self.d_model:
                raise ValueError(
                    "extra_tokens must have shape [B, K, d_model] matching encoder batch and d_model"
                )
            tokens.append(extra_tokens)
            prefix_len += extra_tokens.size(1)
        tokens.append(spectral_tokens)
        memory = torch.cat(tokens, dim=1)

        # Clamp memory length to avoid overflow when high-resolution multi-scale is used.
        if memory.size(1) > self.max_memory_len:
            prefix_part = memory[:, :prefix_len, :]
            spectral_part = memory[:, prefix_len:, :].transpose(1, 2)  # [B, D, S]
            target_spec_len = max(1, self.max_memory_len - prefix_len)
            spectral_part = F.adaptive_avg_pool1d(spectral_part, output_size=target_spec_len).transpose(1, 2)
            memory = torch.cat([prefix_part, spectral_part], dim=1)

        mem_len = memory.size(1)
        pos_ids = torch.arange(mem_len, device=memory.device).unsqueeze(0)
        memory = self.norm(self.dropout(memory + self.memory_pos_emb(pos_ids)))

        if self.buffer_encoder is not None:
            memory = self.buffer_encoder(memory)
            memory = self.buffer_norm(memory)
        return memory

    def forward(
        self,
        ir_spectrum: torch.Tensor,
        formula_vec: Optional[torch.Tensor],
        extra_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        spectral_tokens = self.encode_spectrum(ir_spectrum)
        return self.build_memory(spectral_tokens, formula_vec, extra_tokens=extra_tokens)

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
        transformer_ffn_type: str = "gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.max_tgt_len = max_tgt_len
        self.transformer_ffn_type = str(transformer_ffn_type).lower()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_pos_emb = nn.Embedding(max_tgt_len, d_model)

        if self.transformer_ffn_type == "glu":
            decoder_layer = GLUTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=False,
                ffn_type=self.transformer_ffn_type,
            )
            self.decoder = GLUTransformerDecoder(decoder_layer, num_layers=num_layers)
            self.decoder_batch_first = True
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=False,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            self.decoder_batch_first = False
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

        if self.decoder_batch_first:
            decoded = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        else:
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
        encoder_buffer_layers: int = 2,
        encoder_buffer_dim_feedforward: Optional[int] = None,
        encoder_multiscale_target: str = "mid",
        encoder_use_coordconv: bool = False,
        encoder_patch_size: int = 4,
        transformer_ffn_type: str = "gelu",
        use_formula_input: bool = True,
        num_functional_groups: int = 0,
        use_functional_group_head: bool = False,
        use_functional_group_token: bool = False,
        functional_group_head_dim: int = 256,
        functional_group_dropout: Optional[float] = None,
        functional_group_detach_fusion: bool = False,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
    ):
        super().__init__()
        del (
            num_functional_groups,
            use_functional_group_head,
            use_functional_group_token,
            functional_group_head_dim,
            functional_group_dropout,
            functional_group_detach_fusion,
        )
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.use_formula_input = bool(use_formula_input)
        self.transformer_ffn_type = str(transformer_ffn_type).lower()

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
            use_coordconv=encoder_use_coordconv,
            patch_size=encoder_patch_size,
            transformer_ffn_type=self.transformer_ffn_type,
            use_formula_input=self.use_formula_input,
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
            transformer_ffn_type=self.transformer_ffn_type,
        )

    def _encode_memory(
        self,
        ir_spectrum: torch.Tensor,
        formula_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.encoder(ir_spectrum, formula_vec)

    def forward(
        self,
        ir_spectrum: torch.Tensor,
        formula_vec: Optional[torch.Tensor],
        target_smiles: torch.Tensor,
        return_aux: bool = False,
    ):
        memory = self._encode_memory(ir_spectrum, formula_vec)
        logits = self.decoder(memory, target_smiles)
        if return_aux:
            return {
                "logits": logits,
                "memory": memory,
            }
        return logits

    @torch.no_grad()
    def generate(
        self,
        ir_spectrum: torch.Tensor,
        formula_vec: Optional[torch.Tensor],
        max_len: int = 120,
        sos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        sos_id = self.sos_id if sos_id is None else sos_id
        eos_id = self.eos_id if eos_id is None else eos_id

        bsz = ir_spectrum.size(0)
        memory = self._encode_memory(ir_spectrum, formula_vec)

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
