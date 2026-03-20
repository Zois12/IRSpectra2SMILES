import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from TransformerModel.TransformerModel import MultiScaleResNet1D


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = t.device
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SMILESConditionEncoder(nn.Module):
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

    def _with_formula(
        self,
        memory: torch.Tensor,
        token_pad_mask: torch.Tensor,
        formula_vec: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.formula_proj is None:
            return memory, token_pad_mask

        bsz = memory.size(0)
        if formula_vec is not None:
            if formula_vec.dim() != 2 or formula_vec.size(0) != bsz:
                raise ValueError(f"Expected formula_vec shape [B, F], got {tuple(formula_vec.shape)}")
            formula_token = self.formula_proj(formula_vec).unsqueeze(1)
        else:
            formula_token = self.missing_formula_token.expand(bsz, 1, -1)

        memory = torch.cat([formula_token, memory], dim=1)
        formula_mask = torch.zeros((bsz, 1), dtype=torch.bool, device=memory.device)
        memory_pad_mask = torch.cat([formula_mask, token_pad_mask], dim=1)
        return memory, memory_pad_mask

    def encode_ids(
        self,
        smiles_ids: torch.Tensor,
        formula_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if smiles_ids.dim() != 2:
            raise ValueError(f"Expected smiles_ids shape [B, T], got {tuple(smiles_ids.shape)}")

        bsz, seq_len = smiles_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"smiles length {seq_len} exceeds max_len {self.max_len}")

        token_pad_mask = smiles_ids.eq(self.pad_id)
        pos_ids = torch.arange(seq_len, device=smiles_ids.device).unsqueeze(0).expand(bsz, seq_len)

        x = self.token_emb(smiles_ids) * math.sqrt(self.d_model)
        x = self.input_norm(self.input_dropout(x + self.pos_emb(pos_ids)))
        memory = self.encoder(x, src_key_padding_mask=token_pad_mask)
        memory = self.enc_norm(memory)

        return self._with_formula(memory, token_pad_mask, formula_vec)

    def encode_soft(
        self,
        soft_emb: torch.Tensor,
        pad_mask: torch.Tensor,
        formula_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if soft_emb.dim() != 3:
            raise ValueError(f"Expected soft_emb shape [B, T, D], got {tuple(soft_emb.shape)}")
        bsz, seq_len, _ = soft_emb.shape
        if seq_len > self.max_len:
            raise ValueError(f"soft_emb length {seq_len} exceeds max_len {self.max_len}")

        pos_ids = torch.arange(seq_len, device=soft_emb.device).unsqueeze(0).expand(bsz, seq_len)
        x = soft_emb * math.sqrt(self.d_model)
        x = self.input_norm(self.input_dropout(x + self.pos_emb(pos_ids)))
        memory = self.encoder(x, src_key_padding_mask=pad_mask)
        memory = self.enc_norm(memory)

        return self._with_formula(memory, pad_mask, formula_vec)


class IRLatentAutoencoder(nn.Module):
    def __init__(
        self,
        input_points: int,
        latent_len: int = 256,
        latent_dim: int = 512,
        dropout: float = 0.1,
        multiscale_target: str = "mid",
    ):
        super().__init__()
        self.input_points = int(input_points)
        self.latent_len = int(latent_len)
        self.latent_dim = int(latent_dim)

        self.backbone = MultiScaleResNet1D(
            d_model=self.latent_dim,
            dropout=dropout,
            multiscale_target=multiscale_target,
        )
        self.enc_norm = nn.LayerNorm(self.latent_dim)

        self.dec_proj = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.latent_dim, 1, kernel_size=1),
        )

    def encode(self, ir: torch.Tensor) -> torch.Tensor:
        if ir.dim() == 2:
            x = ir.unsqueeze(1)
        else:
            x = ir
        feat = self.backbone(x)  # [B, D, S]
        feat = F.adaptive_avg_pool1d(feat, self.latent_len)
        z = feat.transpose(1, 2)
        z = self.enc_norm(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        feat = z.transpose(1, 2)
        feat = F.interpolate(feat, size=self.input_points, mode="linear", align_corners=False)
        out = self.dec_proj(feat).squeeze(1)
        return out

    def forward(self, ir: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(ir))


class ConditionalDiffusionModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        time_dim: Optional[int] = None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        time_dim = self.latent_dim if time_dim is None else int(time_dim)

        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.dec_norm = nn.LayerNorm(self.latent_dim)
        self.out_proj = nn.Linear(self.latent_dim, self.latent_dim)

        self.null_token = nn.Parameter(torch.zeros(1, 1, self.latent_dim))

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond_memory: torch.Tensor,
        cond_pad_mask: Optional[torch.Tensor] = None,
        cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        bsz, seq_len, _ = z_t.shape

        if cond_drop_prob > 0 and self.training:
            drop_mask = torch.rand(bsz, device=z_t.device) < cond_drop_prob
            if drop_mask.any():
                null_mem = self.null_token.expand(bsz, cond_memory.size(1), -1)
                cond_memory = cond_memory.clone()
                cond_memory[drop_mask] = null_mem[drop_mask]
                if cond_pad_mask is not None:
                    cond_pad_mask = cond_pad_mask.clone()
                    cond_pad_mask[drop_mask] = False

        time_emb = self.time_mlp(self.time_embed(t))
        z_in = z_t + time_emb.unsqueeze(1)
        out = self.decoder(tgt=z_in, memory=cond_memory, memory_key_padding_mask=cond_pad_mask)
        out = self.dec_norm(out)
        return self.out_proj(out)


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.02)


def build_schedule(timesteps: int, device: torch.device) -> DiffusionSchedule:
    betas = cosine_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
    )


def extract_schedule_coeff(coeff: torch.Tensor, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    out = coeff.gather(0, t)
    while len(out.shape) < len(shape):
        out = out.unsqueeze(-1)
    return out


def q_sample(z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, schedule: DiffusionSchedule) -> torch.Tensor:
    sqrt_alpha = extract_schedule_coeff(schedule.sqrt_alphas_cumprod, t, z0.shape)
    sqrt_one_minus = extract_schedule_coeff(schedule.sqrt_one_minus_alphas_cumprod, t, z0.shape)
    return sqrt_alpha * z0 + sqrt_one_minus * noise


@dataclass
class DiffusionBundle:
    autoencoder: IRLatentAutoencoder
    condition_encoder: SMILESConditionEncoder
    diffusion: ConditionalDiffusionModel
    schedule_steps: int

    @staticmethod
    def build_from_config(config: Dict, device: torch.device) -> "DiffusionBundle":
        autoencoder = IRLatentAutoencoder(
            input_points=config["input_points"],
            latent_len=config["latent_len"],
            latent_dim=config["latent_dim"],
            dropout=config.get("dropout", 0.1),
            multiscale_target=config.get("multiscale_target", "mid"),
        )
        condition_encoder = SMILESConditionEncoder(
            vocab_size=config["vocab_size"],
            formula_dim=config.get("formula_dim", 0),
            d_model=config["cond_dim"],
            nhead=config.get("cond_nhead", 8),
            num_layers=config.get("cond_layers", 6),
            dim_feedforward=config.get("cond_ffn", 2048),
            dropout=config.get("cond_dropout", 0.1),
            max_len=config.get("max_len", 256),
            pad_id=config.get("pad_id", 0),
        )
        diffusion = ConditionalDiffusionModel(
            latent_dim=config["latent_dim"],
            nhead=config.get("diff_nhead", 8),
            num_layers=config.get("diff_layers", 8),
            dim_feedforward=config.get("diff_ffn", 2048),
            dropout=config.get("diff_dropout", 0.1),
            time_dim=config.get("time_dim"),
        )

        autoencoder = autoencoder.to(device)
        condition_encoder = condition_encoder.to(device)
        diffusion = diffusion.to(device)

        return DiffusionBundle(
            autoencoder=autoencoder,
            condition_encoder=condition_encoder,
            diffusion=diffusion,
            schedule_steps=int(config.get("timesteps", 1000)),
        )
