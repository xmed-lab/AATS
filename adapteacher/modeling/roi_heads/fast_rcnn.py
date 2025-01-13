# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    FastRCNNOutputs,
)
from adapteacher.structures.relation import PositionalEmbedding, RankEmbedding_3d

# focal loss

class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions
        losses = FastRCNNFocalLoss(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
        ).losses()

        return losses
    
    
class FastRCNNFocaltLossOutputLayers_branch(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape, fc_features = 1024, n_relations=0):
        super(FastRCNNFocaltLossOutputLayers_branch, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.n_relations = n_relations
        self.if_relation = cfg.SEMISUPNET.RELATION
        if (self.n_relations) and self.if_relation:
            self.dim_g = int(fc_features/n_relations)
            fully_connected1 = nn.Linear(fc_features, fc_features)
            relu1 = nn.ReLU(inplace=True)
            fully_connected2 = nn.Linear(fc_features, fc_features)
            relu2 = nn.ReLU(inplace=True)
            self.relation1= RelationModule(n_relations = n_relations, appearance_feature_dim=fc_features,
                                        key_feature_dim = self.dim_g, geo_feature_dim = self.dim_g)

            # self.relation2 = RelationModule(n_relations=n_relations, appearance_feature_dim=fc_features,
            #                             key_feature_dim=self.dim_g, geo_feature_dim=self.dim_g)
            
            self.fc1 = nn.Sequential(fully_connected1, relu1) #, relation1,
            # self.fc2 = nn.Sequential(fully_connected2, relu2)# relation2)
            # self.fc_bbox1 = nn.Linear(fc_features, 4)
            # self.fc_bbox2 = nn.Linear(fc_features, 4)

    def forward(self, x, branch):
        if ('strong' in branch or 'supervised' in branch) and self.if_relation:
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            pool = self.relation1(self.fc1(x), RankEmbedding_3d().to(x.device))
            # pool = self.relation2(self.fc2(pool), PositionalEmbedding(self.fc_bbox2(pool)))

            scores = self.cls_score(pool)
            proposal_deltas = self.bbox_pred(pool)
            return scores, proposal_deltas
            
        else:
            return super().forward(x)


    def losses(self, predictions, proposals, weights=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions
        losses = FastRCNNFocalLoss(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
            weights=weights
        ).losses()

        return losses


class FastRCNNFocalLoss(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        num_classes=80,
        weights=None,
    ):
        super(FastRCNNFocalLoss, self).__init__(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
            box_reg_loss_type,
        )
        self.num_classes = num_classes
        self.weights = weights

    def losses(self):
        return {
            "loss_cls": self.comput_focal_loss(),
            "loss_box_reg": self.box_reg_loss(),
        }

    def comput_focal_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
            total_loss = FC_loss(input=self.pred_class_logits, target=self.gt_classes, weights=self.weights)
            if self.weights is not None:
                total_loss = total_loss / self.weights.sum()
            else:
                total_loss = total_loss / self.gt_classes.shape[0]

            return total_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target, weights):
        # focal loss
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE

        # lvxg
        if weights is not None:
            loss *= weights 
        return loss.sum()


class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=1024,key_feature_dim = 64, geo_feature_dim = 64):
        super(RelationUnit, self).__init__()
        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        self.WG = nn.Linear(geo_feature_dim, 1, bias=True)
        self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_a, position_embedding):
        N,_ = f_a.size()
        # _, _, num_classes, _ = position_embedding.size() 

        position_embedding = position_embedding.view(-1,self.dim_g)
        # position_embedding = position_embedding.view(N, N, -1)
        # position_embedding = torch.mean(position_embedding, dim=2)  # [1024, 1024]

        w_g = self.relu(self.WG(position_embedding))
        w_g = w_g.view(N,N)

        w_k = self.WK(f_a)
        w_k = w_k.view(N,1,self.dim_k)

        w_q = self.WQ(f_a)
        w_q = w_q.view(1,N,self.dim_k)

        scaled_dot = torch.sum((w_k*w_q),-1 )
        scaled_dot = scaled_dot / np.sqrt(self.dim_k)
        w_a = scaled_dot.view(N,N)

        w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a
        w_mn = torch.nn.Softmax(dim=1)(w_mn)

        w_v = self.WV(f_a)

        w_mn = w_mn.view(N,N,1)
        w_v = w_v.view(N,1,-1)

        output = w_mn*w_v

        output = torch.sum(output,-2)
        return output
    
class RelationModule(nn.Module):
    def __init__(self,n_relations = 16, appearance_feature_dim=1024,key_feature_dim = 64, geo_feature_dim = 64, isDuplication = False):
        super(RelationModule, self).__init__()
        self.isDuplication=isDuplication
        self.Nr = n_relations
        self.dim_g = geo_feature_dim
        self.relation = nn.ModuleList()
        for N in range(self.Nr):
            self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim))
    def forward(self, f_a, position_embedding): # input_data
        # if(self.isDuplication):
        #     f_a, embedding_f_a, position_embedding =input_data
        # else:
        #     f_a, position_embedding = input_data
        isFirst=True
        for N in range(self.Nr):
            if(isFirst):
                if(self.isDuplication):
                    concat = self.relation[N](embedding_f_a,position_embedding)
                else:
                    concat = self.relation[N](f_a,position_embedding)
                isFirst=False
            else:
                if(self.isDuplication):
                    concat = torch.cat((concat, self.relation[N](embedding_f_a, position_embedding)), -1)
                else:
                    concat = torch.cat((concat, self.relation[N](f_a, position_embedding)), -1)
        return concat+f_a