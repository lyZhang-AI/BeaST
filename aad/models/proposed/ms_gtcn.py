import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from aad.models.proposed.ms_tcn import MultiScale_TemporalConv as MS_TCN
from aad.models.proposed.mlp import MLP
from aad.models.proposed.activation import activation_factory
from aad.graphs.tools import k_adjacency, normalize_adjacency_matrix


class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        N, C, T, V = x.shape
        x = self.unfold(x)
        x = x.view(N, C, self.window_size, -1, V).permute(0,1,3,2,4).contiguous()
        x = x.view(N, C, -1, self.window_size * V)
        return x


class SpatialTemporal_MS_GCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 disentangled_agg=True,
                 use_Ares=True,
                 residual=False,
                 dropout=0,
                 activation='relu'):

        super().__init__()
        self.num_scales = num_scales
        self.window_size = window_size
        self.use_Ares = use_Ares
        A = self.build_spatial_temporal_graph(A_binary, window_size)

        if disentangled_agg:
            A_scales = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
            A_scales = np.concatenate([normalize_adjacency_matrix(g) for g in A_scales])
        else:
            A_scales = [normalize_adjacency_matrix(A) for k in range(num_scales)]
            A_scales = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_scales)]
            A_scales = np.concatenate(A_scales)

        self.A_scales = torch.Tensor(A_scales)

        self.V = len(A_binary)

        if use_Ares:
            self.A_res = nn.init.uniform_(nn.Parameter(torch.randn(self.A_scales.shape)), -1e-6, 1e-6)
        else:
            self.A_res = torch.tensor(0)

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation='linear')
        if not residual:
            self.residual = self._zero_residual
        elif in_channels == out_channels:
            self.residual = self._identity_residual
        else:
            self.residual = MLP(in_channels, [out_channels], activation='linear')

        self.act = activation_factory(activation)

    def _zero_residual(self, x):
        return 0

    def _identity_residual(self, x):
        return x
    def build_spatial_temporal_graph(self, A_binary, window_size):
        assert isinstance(A_binary, np.ndarray), 'A_binary should be of type `np.ndarray`'
        V = len(A_binary)
        V_large = V * window_size
        A_binary_with_I = A_binary + np.eye(len(A_binary), dtype=A_binary.dtype)
        A_large = np.tile(A_binary_with_I, (window_size, window_size)).copy()
        return A_large

    def forward(self, x):
        N, C, T, V = x.shape    # T = number of windows

        A = self.A_scales.to(x.dtype).to(x.device) + self.A_res.to(x.dtype).to(x.device)

        res = self.residual(x)
        agg = torch.einsum('vu,nctu->nctv', A, x)
        agg = agg.view(N, C, T, self.num_scales, V)
        agg = agg.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)

        out = self.mlp(agg)
        out += res
        return self.act(out)

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, in_channels, heads=8, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.query = nn.Linear(in_channels, in_channels * heads)
        self.key = nn.Linear(in_channels, in_channels * heads)
        self.value = nn.Linear(in_channels, in_channels * heads)
        self.out_proj = nn.Linear(in_channels * heads, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, T, V, C = x.shape  # [batch, time, nodes, channels]

        query = self.query(x).view(N, T, V, self.heads, C)
        key = self.key(x).view(N, T, V, self.heads, C)
        value = self.value(x).view(N, T, V, self.heads, C)

        attention_scores = torch.einsum('ntvhd,ntvhd->nthv', query, key) / (C ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch, time, heads, nodes]

        attention_output = torch.einsum('nthv,ntvhd->ntvhd', attention_weights, value)
        attention_output = attention_output.contiguous().view(N, T, V, -1)  # Merge heads

        output = self.out_proj(attention_output)
        output = self.dropout(output)
        return output

class SpatialTemporal_MS_GCN_with_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, A_binary, num_scales, window_size, disentangled_agg=True, use_Ares=True, residual=False, dropout=0, activation='relu'):
        super().__init__()
        self.num_scales = num_scales
        self.window_size = window_size
        self.use_Ares = use_Ares
        self.attention_layer = AttentionLayer(in_channels, heads=8, dropout=dropout)

        A = self.build_spatial_temporal_graph(A_binary, window_size)

        if disentangled_agg:
            A_scales = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
            A_scales = np.concatenate([normalize_adjacency_matrix(g) for g in A_scales])
        else:
            A_scales = [normalize_adjacency_matrix(A) for k in range(num_scales)]
            A_scales = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_scales)]
            A_scales = np.concatenate(A_scales)

        self.A_scales = torch.Tensor(A_scales)

        if use_Ares:
            self.A_res = nn.init.uniform_(nn.Parameter(torch.randn(self.A_scales.shape)), -1e-6, 1e-6)
        else:
            self.A_res = torch.tensor(0)

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation='linear')

        if not residual:
            self.residual = self._zero_residual
        elif in_channels == out_channels:
            self.residual = self._identity_residual
        else:
            self.residual = MLP(in_channels, [out_channels], activation='linear')

        self.act = activation_factory(activation)

    def _zero_residual(self, x):
        return 0

    def _identity_residual(self, x):
        return x

    def build_spatial_temporal_graph(self, A_binary, window_size):
        V = len(A_binary)
        A_binary_with_I = A_binary + np.eye(len(A_binary), dtype=A_binary.dtype)
        A_large = np.tile(A_binary_with_I, (window_size, window_size)).copy()
        return A_large

    def forward(self, x):
        N, C, T, V = x.shape

        attention_output = self.attention_layer(x.permute(0, 2, 3, 1))  # [N, T, V, C]
        attention_output = attention_output.permute(0, 3, 1, 2)  # Back to [N, C, T, V]

        A = self.A_scales.to(x.dtype).to(x.device) + self.A_res.to(x.dtype).to(x.device)

        res = self.residual(attention_output)
        agg = torch.einsum('vu,nctu->nctv', A, attention_output)

        agg = agg.view(N, C, T, self.num_scales, V)
        agg = agg.permute(0, 3, 1, 2, 4).contiguous().view(N, self.num_scales * C, T, V)

        out = self.mlp(agg)
        out += res
        return self.act(out)
