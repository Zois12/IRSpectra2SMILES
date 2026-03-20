from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_FUNCTIONAL_GROUPS = (
    "alcohol",
    "phenol",
    "carboxylic_acid",
    "ester",
    "ether",
    "aldehyde",
    "ketone",
    "amine",
    "amide",
    "nitrile",
    "nitro",
    "alkene",
    "alkyne",
    "aromatic_ring",
    "halide",
)


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool1d(x, output_size=1)
        scale = F.gelu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class DepthwiseSeparableConv1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ResidualIRBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
        )
        self.conv2 = DepthwiseSeparableConv1D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            dropout=dropout,
        )
        self.se = SqueezeExcite1D(out_channels, reduction=8)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.out_bn = nn.BatchNorm1d(out_channels)
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.out_bn(out + residual)
        return self.out_act(out)


class AttentiveStatsPool1D(nn.Module):
    def __init__(self, channels: int, attn_hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, attn_hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attn_hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        attn_logits = self.attn(x)
        attn = torch.softmax(attn_logits, dim=-1)
        mean = torch.sum(attn * x, dim=-1)
        second = torch.sum(attn * (x ** 2), dim=-1)
        std = torch.sqrt(torch.clamp(second - mean ** 2, min=1e-6))
        return torch.cat([mean, std], dim=1)


class LightweightIRFunctionalGroupClassifier(nn.Module):
    """
    Lightweight multi-label classifier for IR -> functional groups.

    Expected input:
    - ir_spectrum: [B, L] or [B, 1, L]

    Output:
    - logits: [B, num_functional_groups]
    """

    def __init__(
        self,
        input_points: int = 1625,
        num_labels: int = len(DEFAULT_FUNCTIONAL_GROUPS),
        label_names: Optional[Sequence[str]] = None,
        stem_channels: int = 48,
        hidden_dim: int = 192,
        head_dim: int = 256,
        dropout: float = 0.15,
        use_coordconv: bool = True,
    ):
        super().__init__()
        self.input_points = int(input_points)
        self.num_labels = int(num_labels)
        self.label_names = tuple(label_names) if label_names is not None else DEFAULT_FUNCTIONAL_GROUPS
        self.use_coordconv = bool(use_coordconv)

        if len(self.label_names) != self.num_labels:
            raise ValueError(
                f"label_names length {len(self.label_names)} does not match num_labels {self.num_labels}"
            )

        in_channels = 2 if self.use_coordconv else 1
        branch_channels = stem_channels // 3
        remaining_channels = stem_channels - branch_channels * 2

        self.stem_small = nn.Conv1d(
            in_channels, branch_channels, kernel_size=5, stride=2, padding=2, bias=False
        )
        self.stem_mid = nn.Conv1d(
            in_channels, branch_channels, kernel_size=11, stride=2, padding=5, bias=False
        )
        self.stem_large = nn.Conv1d(
            in_channels, remaining_channels, kernel_size=21, stride=2, padding=10, bias=False
        )
        self.stem_bn = nn.BatchNorm1d(stem_channels)
        self.stem_act = nn.GELU()

        self.block1 = ResidualIRBlock(
            in_channels=stem_channels,
            out_channels=96,
            kernel_size=7,
            stride=2,
            dilation=1,
            dropout=dropout,
        )
        self.block2 = ResidualIRBlock(
            in_channels=96,
            out_channels=hidden_dim,
            kernel_size=5,
            stride=2,
            dilation=1,
            dropout=dropout,
        )
        self.block3 = ResidualIRBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=2,
            dilation=2,
            dropout=dropout,
        )
        self.block4 = ResidualIRBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            dilation=3,
            dropout=dropout,
        )

        self.pool = AttentiveStatsPool1D(hidden_dim, attn_hidden=max(hidden_dim // 2, 64))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, self.num_labels),
        )

    def _add_coord_channel(self, x: torch.Tensor) -> torch.Tensor:
        coord = torch.linspace(-1.0, 1.0, x.size(-1), device=x.device, dtype=x.dtype)
        coord = coord.view(1, 1, -1).expand(x.size(0), 1, -1)
        return torch.cat([x, coord], dim=1)

    def extract_features(self, ir_spectrum: torch.Tensor) -> torch.Tensor:
        x = ir_spectrum.unsqueeze(1) if ir_spectrum.dim() == 2 else ir_spectrum
        if x.dim() != 3:
            raise ValueError(f"Expected ir_spectrum shape [B, L] or [B, 1, L], got {tuple(ir_spectrum.shape)}")

        if self.use_coordconv:
            x = self._add_coord_channel(x)

        stem = torch.cat(
            [self.stem_small(x), self.stem_mid(x), self.stem_large(x)],
            dim=1,
        )
        stem = self.stem_act(self.stem_bn(stem))

        x = self.block1(stem)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.pool(x)

    def forward(self, ir_spectrum: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(ir_spectrum)
        return self.head(features)

    @torch.no_grad()
    def predict_proba(self, ir_spectrum: torch.Tensor) -> torch.Tensor:
        logits = self.forward(ir_spectrum)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_labels(self, ir_spectrum: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        probs = self.predict_proba(ir_spectrum)
        return (probs >= threshold).to(dtype=torch.long)


def functional_group_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} does not match targets shape {tuple(targets.shape)}")

    targets = targets.float()
    if label_smoothing > 0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
