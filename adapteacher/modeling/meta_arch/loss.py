"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(GraphConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        dim_in = 1024
        feat_dim = 1024
        self.head_1 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        self.head_2 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, t_feat, s_feat, graph_cn, labels=None, mask=None):    

        qx = graph_cn.graph.wq(s_feat)
        kx = graph_cn.graph.wk(s_feat)        
        sim_mat = qx.matmul(kx.transpose(-1, -2))
        dot_mat = sim_mat.detach().clone()

        thresh = 0.5
        dot_mat -= dot_mat.min(1, keepdim=True)[0]
        dot_mat /= dot_mat.max(1, keepdim=True)[0]
        mask = ((dot_mat>thresh)*1).detach().clone()
        mask.fill_diagonal_(1)

        anchor_feat = self.head_1(s_feat)
        contrast_feat = self.head_2(s_feat)

        anchor_feat = F.normalize(anchor_feat, dim=1)
        contrast_feat = F.normalize(contrast_feat, dim=1)

        ss_anchor_dot_contrast = torch.div(torch.matmul(anchor_feat, contrast_feat.T), self.temperature)  ##### torch.Size([6, 6])
        logits_max, _ = torch.max(ss_anchor_dot_contrast, dim=1, keepdim=True)  ##### torch.Size([6, 1]) - contains max value along dim=1
        ss_graph_logits = ss_anchor_dot_contrast - logits_max.detach()

        ss_graph_all_logits = torch.exp(ss_graph_logits)
        ss_log_prob = ss_graph_logits - torch.log(ss_graph_all_logits.sum(1, keepdim=True))
        ss_mean_log_prob_pos = (mask * ss_log_prob).sum(1) / mask.sum(1)  
    
        # loss
        ss_loss = - (self.temperature / self.base_temperature) * ss_mean_log_prob_pos
        ss_loss = ss_loss.mean()

        return ss_loss
    

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, weights=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(features.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(features.device)
        else:
            mask = mask.float().to(features.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        if weights is not None:
            loss = (loss.view(anchor_count, batch_size) * weights).sum() / weights.sum()
        else:
            loss = loss.view(anchor_count, batch_size).mean()

        return loss