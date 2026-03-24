import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Encode IR spectrum + molecular formula into a memory sequence for Transformer decoder.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.max_memory_len = max_memory_len
        self.input_points = input_points
        self.buffer_layers = int(buffer_layers)
        self.patch_size = max(int(patch_size), 1)

        self.cnn = ResidualPatchCNN1D(
            d_model=d_model,
            dropout=dropout,
            multiscale_target=multiscale_target,
            use_coordconv=use_coordconv,
            patch_size=self.patch_size,
        )
        self.formula_proj = nn.Sequential(
            nn.Linear(formula_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.memory_pos_emb = nn.Embedding(max_memory_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        if self.buffer_layers > 0:
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
        return self.formula_proj(formula_vec).unsqueeze(1)  # [B, 1, d_model]

    def build_memory(
        self,
        spectral_tokens: torch.Tensor,
        formula_vec: torch.Tensor,
        extra_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        formula_token = self.project_formula(formula_vec)
        tokens = [formula_token]
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
        formula_vec: torch.Tensor,
        extra_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        spectral_tokens = self.encode_spectrum(ir_spectrum)
        return self.build_memory(spectral_tokens, formula_vec, extra_tokens=extra_tokens)


class SpectralTokenPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model * 2)

    def forward(self, spectral_tokens: torch.Tensor) -> torch.Tensor:
        if spectral_tokens.dim() != 3:
            raise ValueError(f"Expected spectral_tokens [B, S, D], got {tuple(spectral_tokens.shape)}")
        mean_pool = spectral_tokens.mean(dim=1)
        max_pool = spectral_tokens.max(dim=1).values
        return self.norm(torch.cat([mean_pool, max_pool], dim=-1))


class FunctionalGroupHead(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(int(hidden_dim), 64)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_features)


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
        encoder_buffer_layers: int = 2,
        encoder_buffer_dim_feedforward: Optional[int] = None,
        encoder_multiscale_target: str = "mid",
        encoder_use_coordconv: bool = False,
        encoder_patch_size: int = 4,
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
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.num_functional_groups = int(num_functional_groups)
        self.use_functional_group_head = bool(use_functional_group_head)
        self.use_functional_group_token = bool(use_functional_group_token)
        self.functional_group_detach_fusion = bool(functional_group_detach_fusion)

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
        )
        fg_dropout = dropout if functional_group_dropout is None else float(functional_group_dropout)
        self.spectral_pool = None
        self.functional_group_head = None
        self.functional_group_token_proj = None
        if self.use_functional_group_head:
            if self.num_functional_groups <= 0:
                raise ValueError("num_functional_groups must be > 0 when use_functional_group_head=True")
            self.spectral_pool = SpectralTokenPooling(d_model)
            self.functional_group_head = FunctionalGroupHead(
                input_dim=d_model * 2,
                num_labels=self.num_functional_groups,
                hidden_dim=functional_group_head_dim,
                dropout=fg_dropout,
            )
            if self.use_functional_group_token:
                self.functional_group_token_proj = nn.Sequential(
                    nn.Linear(self.num_functional_groups, d_model),
                    nn.GELU(),
                    nn.LayerNorm(d_model),
                )
        elif self.use_functional_group_token:
            raise ValueError("use_functional_group_token=True requires use_functional_group_head=True")
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

    def _encode_with_functional_groups(
        self,
        ir_spectrum: torch.Tensor,
        formula_vec: torch.Tensor,
    ):
        spectral_tokens = self.encoder.encode_spectrum(ir_spectrum)
        functional_group_logits = None
        functional_group_probs = None
        extra_tokens = None

        if self.functional_group_head is not None and self.spectral_pool is not None:
            pooled = self.spectral_pool(spectral_tokens)
            functional_group_logits = self.functional_group_head(pooled)
            functional_group_probs = torch.sigmoid(functional_group_logits)
            if self.functional_group_token_proj is not None:
                fusion_input = functional_group_probs.detach() if self.functional_group_detach_fusion else functional_group_probs
                extra_tokens = self.functional_group_token_proj(fusion_input).unsqueeze(1)

        memory = self.encoder.build_memory(spectral_tokens, formula_vec, extra_tokens=extra_tokens)
        return memory, functional_group_logits, functional_group_probs

    def forward(
        self,
        ir_spectrum: torch.Tensor,
        formula_vec: torch.Tensor,
        target_smiles: torch.Tensor,
        return_aux: bool = False,
    ):
        memory, functional_group_logits, functional_group_probs = self._encode_with_functional_groups(
            ir_spectrum,
            formula_vec,
        )
        logits = self.decoder(memory, target_smiles)
        if return_aux:
            return {
                "logits": logits,
                "functional_group_logits": functional_group_logits,
                "functional_group_probs": functional_group_probs,
                "memory": memory,
            }
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
        memory, _, _ = self._encode_with_functional_groups(ir_spectrum, formula_vec)

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

    @torch.no_grad()
    def predict_functional_groups(
        self,
        ir_spectrum: torch.Tensor,
        formula_vec: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        _, _, functional_group_probs = self._encode_with_functional_groups(ir_spectrum, formula_vec)
        return functional_group_probs
