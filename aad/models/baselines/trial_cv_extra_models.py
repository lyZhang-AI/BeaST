import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSpatialStem(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=(9, 1), padding=(4, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2), groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )

    def forward(self, x):
        return self.net(x)


class AGSLNet(nn.Module):
    """Adaptive graph and spectral-local baseline for trial-CV experiments."""

    def __init__(self, num_channels=64, num_classes=2, hidden=32):
        super().__init__()
        self.stem = TemporalSpatialStem(hidden)
        self.channel_proj = nn.Linear(hidden, hidden)
        self.graph_score = nn.Parameter(torch.randn(num_channels, num_channels) * 0.02)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: [B, 1, T, C]
        feat = self.stem(x).mean(dim=2).transpose(1, 2)  # [B, C, H]
        adj = torch.softmax(F.relu(self.graph_score), dim=-1)
        feat = torch.matmul(adj, feat)
        feat = self.channel_proj(feat).mean(dim=1)
        return self.classifier(feat)


class XANet(nn.Module):
    """Cross-axis attention network for EEG windows."""

    def __init__(self, num_classes=2, hidden=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(hidden),
            nn.ELU(),
            nn.Conv2d(hidden, hidden, kernel_size=(64, 1), groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ELU(),
        )
        self.time_attn = nn.MultiheadAttention(hidden, num_heads=4, dropout=0.1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: [B, 1, C, T]
        feat = self.stem(x).squeeze(2).transpose(1, 2)  # [B, T, H]
        feat, _ = self.time_attn(feat, feat, feat, need_weights=False)
        return self.classifier(feat.mean(dim=1))


class GraphEEG(nn.Module):
    """Compact graph EEG baseline with learnable adjacency."""

    def __init__(self, num_channels=64, num_classes=2, hidden=32):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=(15, 1), padding=(7, 0), bias=False),
            nn.BatchNorm2d(hidden),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        self.adj = nn.Parameter(torch.eye(num_channels))
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: [B, 1, T, C]
        feat = self.temporal(x).mean(dim=2).transpose(1, 2)  # [B, C, H]
        adj = torch.softmax(F.relu(self.adj), dim=-1)
        feat = torch.matmul(adj, feat)
        feat = self.node_mlp(feat).mean(dim=1)
        return self.classifier(feat)
