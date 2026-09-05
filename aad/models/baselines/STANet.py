import numpy as np
import math
import random
import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from torch.backends import cudnn







# Spatial Attention Module
class SpatialAtt(nn.Module):
    def __init__(self):
        # self.patch_size = patch_size
        super().__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 3, (1, 1), (1, 1)),
            nn.ELU(),
        )

        self.pooling1 = nn.AdaptiveAvgPool2d((1, None))

        self.FC = nn.Sequential(
            nn.Linear(64, 16),
            nn.ELU(),
            nn.Linear(16, 64),
        )

        self.ConvBlock = nn.Sequential(
            nn.Conv2d(1, 5, (1, 64), (1, 1)),
            nn.ELU(),
            nn.MaxPool2d((4, 1))
        )

    def forward(self, x: Tensor) -> Tensor:
        x = torch.transpose(x, 3, 2)
        b, _, _, _ = x.shape
        out = self.Conv1(x)
        # print("shape", out.shape)
        out, _ = torch.max(out, dim=1)
        out = torch.unsqueeze(out, dim=1)
        # print("out shape", out.shape)
        x = x * out
        x = self.FC(x)
        # print("x shape", x.shape)
        x = self.ConvBlock(x)
        x = torch.squeeze(x, dim=-1)
        x = torch.transpose(x, 2, 1)
        # print(x.shape)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 修改线性层的输入和输出维度
        self.keys = nn.Linear(emb_size, 50)
        self.queries = nn.Linear(emb_size, 50)
        self.values = nn.Linear(emb_size, emb_size)  # values的维度保持不变
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = 50 ** (1 / 2)  # 修改scaling的计算，使其与新的维度相匹配
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=1,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size,n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(128),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # print("FC", x.shape)
        x = x.contiguous().view(x.size(0), -1)
        # print("FC2", x.shape)
        out = self.fc(x)
        return out


class STANet(nn.Sequential):
    def __init__(self, emb_size=5, depth=1, n_classes=2, **kwargs):
        super().__init__(

            SpatialAtt(),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )
