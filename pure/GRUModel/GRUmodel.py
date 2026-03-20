import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class IREncoder(nn.Module):
    def __init__(self, input_points=1652, latent_dim=512):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = ResidualBlock1D(64, 128, stride=2)
        self.layer2 = ResidualBlock1D(128, 256, stride=2)
        self.layer3 = ResidualBlock1D(256, 512, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(51 * 512, latent_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        return self.fc(x)


class FusionEncoder(nn.Module):
    def __init__(self, ir_encoder: IREncoder, formula_dim: int, latent_dim: int):
        super().__init__()
        self.ir_encoder = ir_encoder
        self.formula_proj = nn.Sequential(
            nn.Linear(formula_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + 128, latent_dim),
            nn.ReLU(),
        )

    def forward(self, ir_spectrum: torch.Tensor, formula_vec: torch.Tensor) -> torch.Tensor:
        ir_latent = self.ir_encoder(ir_spectrum)
        formula_latent = self.formula_proj(formula_vec)
        fused = torch.cat([ir_latent, formula_latent], dim=1)
        return self.fusion(fused)


class SMILESDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_to_h0 = nn.Linear(latent_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, latent_vector, target_seq, teacher_forcing_ratio=0.5):
        batch_size = latent_vector.size(0)
        max_len = target_seq.size(1)
        vocab_size = self.fc_out.out_features

        h = self.latent_to_h0(latent_vector).unsqueeze(0)
        outputs = torch.zeros(batch_size, max_len, vocab_size, device=latent_vector.device)
        input_token = target_seq[:, 0]

        for t in range(1, max_len):
            embedded = self.embedding(input_token).unsqueeze(1)
            output, h = self.gru(embedded, h)
            prediction = self.fc_out(output.squeeze(1))
            outputs[:, t, :] = prediction

            is_teacher = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = target_seq[:, t] if is_teacher else top1

        return outputs


class IR2SMILES(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, ir_spectrum, formula_vec, target_smiles):
        latent = self.encoder(ir_spectrum, formula_vec)
        return self.decoder(latent, target_smiles)
