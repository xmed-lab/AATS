"""
Author: Lvxg
Date: Feb 26, 2024
"""
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import pdb
    
# class RelativePosition(nn.Module):

#     def __init__(self, num_units, max_relative_position):
#         super().__init__()
#         self.num_units = num_units
#         self.max_relative_position = max_relative_position
#         self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
#         nn.init.xavier_uniform_(self.embeddings_table)

#     def forward(self, length_q, length_k):
#         range_vec_q = torch.arange(length_q)
#         range_vec_k = torch.arange(length_k)
#         distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
#         distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
#         final_mat = distance_mat_clipped + self.max_relative_position
#         final_mat = torch.LongTensor(final_mat).cuda()
#         embeddings = self.embeddings_table[final_mat].cuda()

#         return embeddings

# class MultiHeadAttentionLayer(nn.Module):
#     def __init__(self, hid_dim, n_heads, dropout, device):
#         super().__init__()
        
#         assert hid_dim % n_heads == 0
        
#         self.hid_dim = hid_dim
#         self.n_heads = n_heads
#         self.head_dim = hid_dim // n_heads
#         self.max_relative_position = 2

#         self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
#         self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

#         self.fc_q = nn.Linear(hid_dim, hid_dim)
#         self.fc_k = nn.Linear(hid_dim, hid_dim)
#         self.fc_v = nn.Linear(hid_dim, hid_dim)
        
#         self.fc_o = nn.Linear(hid_dim, hid_dim)
        
#         self.dropout = nn.Dropout(dropout)
        
#         self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
#     def forward(self, query, key, value, mask = None):
#         #query: [batch size, query len, hid dim]
#         #key: [batch size, key len, hid dim]
#         #value: [batch size, value len, hid dim]
#         batch_size = query.shape[0]
#         len_k = key.shape[1]
#         len_q = query.shape[1]
#         len_v = value.shape[1]

#         query = self.fc_q(query)
#         key = self.fc_k(key)
#         value = self.fc_v(value)

#         r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

#         r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
#         r_k2 = self.relative_position_k(len_q, len_k)
#         attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
#         attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
#         attn = (attn1 + attn2) / self.scale

#         if mask is not None:
#             attn = attn.masked_fill(mask == 0, -1e10)

#         attn = self.dropout(torch.softmax(attn, dim = -1))

#         #attn: [batch size, n heads, query len, key len]
#         r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         weight1 = torch.matmul(attn, r_v1)
#         r_v2 = self.relative_position_v(len_q, len_v)
#         weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
#         weight2 = torch.matmul(weight2, r_v2)
#         weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)
#         # x: [batch size, n heads, query len, head dim]
#         x = weight1 + weight2
#         # x: [batch size, query len, n heads, head dim]
#         x = x.permute(0, 2, 1, 3).contiguous()
#         # x: [batch size, query len, hid dim]   
#         x = x.view(batch_size, -1, self.hid_dim)
#         # x: [batch size, query len, hid dim]
#         x = self.fc_o(x)

#         return x
    
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            #print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            #print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            #print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Feat2Graph(nn.Module):
    def __init__(self, num_feats):
        super(Feat2Graph, self).__init__()
        self.wq = nn.Linear(num_feats, num_feats)
        self.wk = nn.Linear(num_feats, num_feats)

    def forward(self, x):
        qx = self.wq(x)
        kx = self.wk(x)

        dot_mat = qx.matmul(kx.transpose(-1, -2))
        adj = F.normalize(dot_mat.square(), p=1, dim=-1)
        return x, adj

class GCN(nn.Module):
    def __init__(self, in_feat, nhid, out_feat, dropout=False, init="xavier"):
        super(GCN, self).__init__()
        self.graph = Feat2Graph(in_feat)

        self.gc1 = GraphConvolution(in_feat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nhid, init=init)
        self.gc3 = GraphConvolution(nhid, out_feat, init=init)
        self.dropout = dropout


    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x):
        x_in = x

        x, adj = self.graph(x)
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))

        return x
    
class GAT(nn.Module):
    def __init__(self, nfeat=512, nhid=512, nfeatout=512, dropout=0.1, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.graph = Feat2Graph(nfeat)
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nfeatout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x_in):
        x, adj = self.graph(x_in)
        x = F.dropout(x, self.dropout)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)