import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import scipy.sparse as sp
import scipy
import torch.nn.functional as F
from einops import rearrange

def top_rank(attention_score, keep_ratio):
    """基于给定的attention_score, 对每个图进行pooling操作.
    为了直观体现pooling过程，我们将每个图单独进行池化，最后再将它们级联起来进行下一步计算

    Arguments:
    ----------
        attention_score：torch.Tensor
            使用GCN计算出的注意力分数，Z = GCN(A, X)
        graph_indicator：torch.Tensor
            指示每个节点属于哪个图
        keep_ratio: float
            要保留的节点比例，保留的节点数量为int(N * keep_ratio)
    """
    shape=attention_score.shape
    graph_node_num = attention_score.shape[1]
    keep_graph_node_num = int(keep_ratio * graph_node_num)
    graph_mask = attention_score.new_zeros(shape,
                                            dtype=torch.bool)

    values, indices = attention_score.topk(keep_graph_node_num, dim=1, largest=True, sorted=True)
    mask=graph_mask.scatter(1,indices,True)

    return mask



class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0
        self.lin1=nn.Linear(input_dim, hidden_dim)
        self.bn1=nn.BatchNorm1d(hidden_dim)
        self.relu1=nn.ReLU()
        self.lin2=nn.Linear(hidden_dim, output_dim)
        self.bn2=nn.BatchNorm1d(output_dim)
        self.relu2=nn.ReLU()

    def forward(self, v, a):
        v_aggregate = torch.bmm(a, v)
        v_aggregate += self.epsilon * v # assumes that the adjacency matrix includes self-loop
        v_combine=self.lin1(v_aggregate)
        v_combine=rearrange(v_combine, 'b n c -> b c n')
        v_combine=self.bn1(v_combine)
        v_combine=rearrange(v_combine, 'b c n -> b n c')
        v_combine=self.relu1(v_combine)
        v_combine=self.lin2(v_combine)
        v_combine=rearrange(v_combine, 'b n c -> b c n')
        v_combine = self.bn2(v_combine)
        v_combine=rearrange(v_combine, 'b c n -> b n c')
        v_combine=self.relu2(v_combine)

        return v_combine


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, keep_ratio, activation=torch.tanh):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.activation = activation
        self.attn_gcn = LayerGIN(input_dim, input_dim//2, 1)

    def forward(self, adjacency, input_feature):
        attn_score = self.attn_gcn( input_feature,adjacency).squeeze()
        attn_score = self.activation(attn_score)
        mask = top_rank(attn_score, self.keep_ratio)
        batch=mask.shape[0]
        t=input_feature[mask]#.view(batch,-1,self.input_dim)
        t2=attn_score[mask].view(-1,1)#.view(-1, 1)
        hidden = input_feature[mask] * attn_score[mask].view(-1, 1)
        return hidden.view(batch,-1,self.input_dim)#, mask_adjacency