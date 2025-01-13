"""
Author: Lvxg
Date: Feb 26, 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import SGD 
from torch.autograd import Variable
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from adapteacher.modeling.meta_arch.GNN import GAT
from adapteacher.structures.resnet_3x3 import ResNet, BasicBlock 
from adapteacher.structures.attentions import AggregateAttention
from detectron2.structures import ImageList, Instances
from tqdm import tqdm
import numpy as np
import warnings
import random
import os
warnings.filterwarnings("ignore", category=UserWarning)

PIXEL_MEAN =  torch.tensor([123.675, 116.280, 103.530]).view(-1, 1, 1)
PIXEL_STD =  torch.tensor([58.395, 57.120, 57.375]).view(-1, 1, 1)

def feat2prob(feat, center, alpha=1.0):
    q = 1.0 / (1.0 + torch.sum(
        torch.pow(feat.unsqueeze(1) - center, 2), 2) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def init_prob_kmeans(model, eval_loader, args, data_len):
    torch.manual_seed(1)
    # cluster parameter initiate
    model.eval()
    layer = args.SEMISUPNET.DIS_TYPE
    B = args.SOLVER.IMG_PER_BATCH_UNLABEL
    nclass = args.MODEL.ROI_HEADS.NUM_CLASSES
    loader_len = int(np.ceil(data_len / B))
    feats = np.zeros((loader_len * B, 512))

    for i in tqdm(range(loader_len), desc="Init prob with Kmeans"):
        _, _, _, wk = next(eval_loader)
        x = preprocess_image(wk)
        x = x.to(args.MODEL.DEVICE)
        feat = model(x.tensor)
        feat_pooled = F.adaptive_avg_pool2d(feat[layer], (1, 1))
        # idx = idx.data.cpu().numpy()
        feats[np.arange(i * B, (i + 1) * B), :] = feat_pooled.squeeze(-1).squeeze(-1).data.cpu().numpy()
    # evaluate clustering performance
    # pca = PCA(n_components=args.n_clusters)
    # feats = pca.fit_transform(feats)
    kmeans = KMeans(n_clusters=nclass, n_init=20)
    y_pred = kmeans.fit_predict(feats) 
    center = torch.from_numpy(kmeans.cluster_centers_).to(args.MODEL.DEVICE)
    init_probs = feat2prob(torch.from_numpy(feats).to(args.MODEL.DEVICE), center)
    p_targets = target_distribution(init_probs)

    return center, p_targets

def preprocess_image(batched_inputs):
    """
    Normalize, pad and batch the input images.
    """
    images = [x["image"] for x in batched_inputs]
    images = [(x - PIXEL_MEAN) / PIXEL_STD for x in images]
    images = ImageList.from_tensors(images, 0)
    return images


class DUC(nn.Module):
    """
    Deep Unsupervise Clustering
    """
    def __init__(self, cfg):
        super(DUC, self).__init__()
        self.cfg = cfg
        self.rampup_coefficient = 10.0
        self.rampup_length = 5
        self.device = self.cfg.MODEL.DEVICE
        self.nclass = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # self.model = GAT().to(self.device)
        # self.model = ResNet(BasicBlock, [2,2,2,2]).to(self.device)
        self.model = AggregateAttention(512, self.nclass).to(self.device)
        self.model.center = None
        self.p_targets = None
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, feats, labels, p_targets):

        feats, nodes = self.model(feats, labels)
        if self.model.center is None:
            init_center, p_targets = self.init_prob_kmeans(nodes, self.nclass)
            # self.model.center = nn.Parameter(torch.Tensor(init_center.size()))
            # self.model.center.data = torch.tensor(init_center).float().to(self.device)
            self.model.center = nn.Parameter(torch.tensor(init_center).float().to(self.device))
            self.p_targets = p_targets

        if p_targets is None:
            p_targets = self.p_targets

        # w = self.rampup_coefficient * sigmoid_rampup(iter, self.rampup_length) 
        probs = feat2prob(nodes, self.model.center)
        # pred_loss = self.celoss(probs, torch.arange(self.nclass).to(self.device))
        pred_loss = self.celoss(self.model.center, torch.arange(self.nclass).to(self.device))
        sharp_loss = F.kl_div(probs.log(), p_targets.float().to(self.device))
        loss = sharp_loss + 0.1 * pred_loss

        return loss, feats, probs

    def init_prob_kmeans(self, feats, n_clusters):
        # evaluate clustering performance
        # pca = PCA(n_components=n_clusters)
        # feats = pca.fit_transform(feats)
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(feats.detach().cpu().numpy()) 
        # acc, nmi, ari = cluster_acc(targets, y_pred), nmi_score(targets, y_pred), ari_score(targets, y_pred)
        # print('Init acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
        center = torch.from_numpy(kmeans.cluster_centers_).to(self.device)
        probs = feat2prob(feats, center)
        p_targets = target_distribution(probs.detach())
        return center, p_targets

