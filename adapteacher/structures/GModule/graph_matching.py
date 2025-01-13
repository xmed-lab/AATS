import torch
from torch import nn
import torch.nn.functional as F

import sklearn.cluster as cluster
from adapteacher.structures.GModule.losses import BCEFocalLoss
from adapteacher.structures.GModule.affinity import Affinity
from adapteacher.structures.attentions import MultiHeadAttention, HyperGraph
from adapteacher.modeling.meta_arch.loss import SupConLoss

class V2GConv(torch.nn.Module):
    # Project the sampled visual features to the graph embeddings:
    # visual features: [0,+INF) -> graph embedding: (-INF, +INF)
    def __init__(self, opt, in_channels, out_channel, mode='in'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        if mode == 'in':
            num_convs = opt.MODEL.MIDDLE_HEAD.NUM_CONVS_IN
        elif mode == 'out':
            num_convs = opt.MODEL.MIDDLE_HEAD.NUM_CONVS_OUT
        else:
            num_convs = opt.MODEL.FCOS.NUM_CONVS
            print('undefined num_conv in middle head')

        middle_tower = []
        for i in range(num_convs):
            middle_tower.append(
                nn.Conv2d(
                    in_channels,
                    out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            if mode == 'in':
                if opt.MODEL.MIDDLE_HEAD.IN_NORM == 'GN':
                    middle_tower.append(nn.GroupNorm(32, in_channels))
                elif opt.MODEL.MIDDLE_HEAD.IN_NORM == 'IN':
                    middle_tower.append(nn.InstanceNorm2d(in_channels))
                elif opt.MODEL.MIDDLE_HEAD.IN_NORM == 'BN':
                    middle_tower.append(nn.BatchNorm2d(in_channels))
            if i != (num_convs - 1):
                middle_tower.append(nn.ReLU())

        self.add_module('middle_tower', nn.Sequential(*middle_tower))

        for modules in [self.middle_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        middle_tower = []
        for l, feature in enumerate(x):
            middle_tower.append(self.middle_tower(feature))
        return middle_tower

def build_V2G_linear(dim):
    head_in_ln = nn.Sequential(
        nn.Linear(dim, dim),
        nn.LayerNorm(dim, elementwise_affine=False),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.LayerNorm(dim, elementwise_affine=False),
    )
    return head_in_ln

class GModule(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(GModule, self).__init__()

        init_item = []
        
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.with_hyper_graph = False

        # One-to-one (o2o) matching or many-to-many (m2m) matching?
        self.matching_cfg = 'o2o'  # 'o2o' and 'm2m'        
        
        # add quadratic matching constraints.
        self.with_quadratic_matching = True

        # Several weights hyper-parameters
        self.weight_matching = 0.1
        self.weight_nodes = 0.1
        self.weight_dis = 0.1
        self.lambda_dis = 0.01

        # Detailed settings
        self.with_node_dis = False
        self.with_global_graph = False # F

        # Pre-processing for the vision-to-graph transformation
        self.head_in_ln = build_V2G_linear(512)
        init_item.append('head_in_ln')

        # node classification layers
        self.node_cls_middle = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )
        init_item.append('node_cls_middle')

        self.cross_domain_graph = MultiHeadAttention(in_channels, 1, dropout=0.1, version='v2')  # Cross Graph Interaction

        if self.with_hyper_graph:
            self.intra_domain_graph = HyperGraph(emb_dim=256, K_neigs=self.num_hyper_edge, num_layer=self.num_hypergnn_layer)  # Intra-domain graph aggregation
        else:
            self.intra_domain_graph = MultiHeadAttention(in_channels, 1, dropout=0.1, version='v2')  # Intra-domain graph aggregation

        # Semantic-aware Node Affinity
        self.node_affinity = Affinity(d=in_channels)
        self.InstNorm_layer = nn.InstanceNorm2d(1)

        # Structure-aware Matching Loss
        # Different matching loss choices
        self.matching_loss_cfg = 'MSE'
        if self.matching_loss_cfg == 'L1':
            self.matching_loss = nn.L1Loss(reduction='sum')
        elif self.matching_loss_cfg == 'MSE':
            self.matching_loss = nn.MSELoss(reduction='sum')
        elif self.matching_loss_cfg == 'BCE':
            self.matching_loss = BCEFocalLoss()
        self.quadratic_loss = torch.nn.L1Loss(reduction='mean')
        self.supconloss = SupConLoss(contrast_mode='one')

        if self.with_node_dis:
            self.node_dis_2 = nn.Sequential(
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            init_item.append('node_dis')
            self.loss_fn = nn.BCEWithLogitsLoss()
        self._init_weight(init_item)

    def _init_weight(self, init_item=None):
        
        if 'node_dis' in init_item:
            for i in self.node_dis_2:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
            self.logger.info('node_dis initialized')
        if 'node_cls_middle' in init_item:
            for i in self.node_cls_middle:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
        if 'head_in_ln' in init_item:
            for i in self.head_in_ln:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)

    def forward(self, features, labels, center):
        '''
        We have equal number of source/target feature maps
        features: [sr_feats, tg_feats]
        targets: [sr_targets, None]

        '''
        feat_loss = self._forward_train(features, labels, center)
        return feat_loss

    def _forward_train(self, features, labels, center):
        features_s, features_t = features
        labels_s, labels_t = labels
        middle_head_loss = {}

        if features_s.dim() == 1:
            features_s = features_s.unsqueeze(0)
        if features_t.dim() == 1:
            features_t = features_t.unsqueeze(0)

        # STEP1: Graph Completement    
        features_s, labels_s = self.nodes_complete(features_s, labels_s, center)
        
        # STEP2: Inter Graph Interaction
        nodes_s, edges_s = self._forward_intra_domain_graph(features_s)
        nodes_t, edges_t = self._forward_intra_domain_graph(features_t)

        # STEP3: Conduct Cross Graph Interaction (CGI)
        nodes_s, nodes_t = self._forward_cross_domain_graph(nodes_s, nodes_t)

        # STEP4: Generate node loss
        node_loss = self._forward_node_loss(
            torch.cat([nodes_s, nodes_t], dim=0),
            torch.cat([labels_s, labels_t], dim=0)
        )
        middle_head_loss.update({'loss_node': self.weight_nodes * node_loss})

        # STEP5: Node Affinity and Quadratic Constrain
        if self.matching_cfg != 'none':
            matching_loss_affinity, affinity = self._forward_aff(nodes_s, nodes_t, labels_s, labels_t)
            middle_head_loss.update({'loss_mat_aff': self.weight_matching * matching_loss_affinity})

            if self.with_quadratic_matching:
                matching_loss_quadratic = self._forward_qu(nodes_s, nodes_t, edges_s.detach(), edges_t.detach(), affinity)
                middle_head_loss.update({'loss_mat_qu': matching_loss_quadratic})

        return middle_head_loss

    def _forward_intra_domain_graph(self, nodes):
        nodes, edges = self.intra_domain_graph([nodes, nodes, nodes])
        return nodes, edges

    def _forward_cross_domain_graph(self, nodes_1, nodes_2):

        n_1 = len(nodes_1)
        n_2 = len(nodes_2)
        global_nodes = torch.cat([nodes_1, nodes_2], dim=0)
        global_nodes = self.cross_domain_graph([global_nodes, global_nodes, global_nodes])[0]

        nodes1_enahnced = global_nodes[:n_1]
        nodes2_enahnced = global_nodes[n_1:]
        
        return nodes1_enahnced, nodes2_enahnced

    def _forward_node_loss(self, nodes, labels, weights=None):

        labels = labels.long()
        assert len(nodes) == len(labels)

        if weights is None:  # Source domain
        
            logits = self.node_cls_middle(nodes)

            node_loss = F.cross_entropy(logits, labels,
                                        reduction='mean')
        else:  # Target domain
            if self.with_cond_cls:
                sr_embeds = self.node_cls_middle(self.sr_seed)
                logits = self.dynamic_fc(nodes, sr_embeds)
            else:
                logits = self.node_cls_middle(nodes)

            node_loss = F.cross_entropy(logits, labels.long(),
                                        reduction='none')
            node_loss = (node_loss * weights).float().mean() if self.with_score_weight else node_loss.float().mean()

        return node_loss

    def nodes_complete(self, nodes, labels, center):

        uni_labels = labels.unique(sorted=True).long()
        all_labels = torch.arange(self.num_classes).to(nodes.device)
        is_present = (all_labels.unsqueeze(1) == uni_labels).any(dim=1)
        missing_labels = all_labels[~is_present]

        for label in missing_labels:
            center_comp = center[label].unsqueeze(0)
            nodes = torch.cat([nodes, center_comp], 0)
            labels = torch.cat((labels, label.unsqueeze(0)))
            assert nodes.shape[0] == len(labels)

        return nodes, labels
              
    def _forward_aff(self, nodes_1, nodes_2, labels_side1, labels_side2):
        if self.matching_cfg == 'o2o':
            M = self.node_affinity(nodes_1, nodes_2)
            matching_target = torch.mm(self.one_hot(labels_side1), self.one_hot(labels_side2).t())

            M = self.InstNorm_layer(M[None, None, :, :])
            M = self.sinkhorn_iter(M[:, 0, :, :], n_iters=20).squeeze().exp()

            TP_mask = (matching_target == 1).float().to(M.device)
            indx = (M * TP_mask).max(-1)[1]
            TP_samples = M[range(M.size(0)), indx].view(-1, 1)
            TP_target = torch.full(TP_samples.shape, 1, dtype=torch.float, device=TP_samples.device).float()

            FP_samples = M[matching_target == 0].view(-1, 1)
            FP_target = torch.full(FP_samples.shape, 0, dtype=torch.float, device=FP_samples.device).float()

            # TODO Find a better reduction strategy
            TP_loss = self.matching_loss(TP_samples, TP_target.float()) / len(TP_samples)
            FP_loss = self.matching_loss(FP_samples, FP_target.float()) / torch.sum(FP_samples).detach()
            matching_loss = TP_loss + FP_loss

        elif self.matching_cfg == 'm2m':  # Refer to the Appendix
            M = self.node_affinity(nodes_1, nodes_2)
            matching_target = torch.mm(self.one_hot(labels_side1), self.one_hot(labels_side2).t())
            matching_loss = self.matching_loss(M.sigmoid(), matching_target.float()).mean()
        else:
            M = None
            matching_loss = 0
        return matching_loss, M

    def _forward_inference(self, images, features):
        return features

    def _forward_qu(self, nodes_1, nodes_2, edges_1, edges_2, affinity):

        if self.with_hyper_graph:

            # hypergraph matching (high order)
            translated_indx = list(range(1, self.num_hyper_edge))+[int(0)]
            mathched_index = affinity.argmax(0)
            matched_node_1 = nodes_1[mathched_index]
            matched_edge_1 = edges_1.t()[mathched_index]
            matched_edge_1[matched_edge_1 > 0] = 1

            matched_node_2 =nodes_2
            matched_edge_2 =edges_2.t()
            matched_edge_2[matched_edge_2 > 0] = 1
            n_nodes = matched_node_1.size(0)

            angle_dis_list = []
            for i in range(n_nodes):
                triangle_1 = nodes_1[matched_edge_1[i, :].bool()]  # 3 x 256
                triangle_1_tmp = triangle_1[translated_indx]
                # print(triangle_1.size(), triangle_1_tmp.size())
                sin1 = torch.sqrt(1.- F.cosine_similarity(triangle_1, triangle_1_tmp).pow(2)).sort()[0]
                triangle_2 = nodes_2[matched_edge_2[i, :].bool()]  # 3 x 256
                triangle_2_tmp = triangle_2[translated_indx]
                sin2 = torch.sqrt(1.- F.cosine_similarity(triangle_2, triangle_2_tmp).pow(2)).sort()[0]
                angle_dis = (-1 / self.angle_eps  * (sin1 - sin2).abs().sum()).exp()
                angle_dis_list.append(angle_dis.view(1,-1))

            angle_dis_list = torch.cat(angle_dis_list)
            loss = angle_dis_list.mean()
        else:
            # common graph matching (2nd order)
            R = torch.mm(edges_1, affinity) - torch.mm(affinity, edges_2)
            loss = self.quadratic_loss(R, R.new_zeros(R.size()))
        return loss

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def sinkhorn_iter(self, log_alpha, n_iters=5, slack=True, eps=-1):
        ''' Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

        Args:
            log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
            n_iters (int): Number of normalization iterations
            slack (bool): Whether to include slack row and column
            eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K)

        Modified from original source taken from:
            Learning Latent Permutations with Gumbel-Sinkhorn Networks
            https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
        '''
        prev_alpha = None
        if slack:
            zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
            log_alpha_padded = zero_pad(log_alpha[:, None, :, :])
            log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

            for i in range(n_iters):
                # Row normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                    dim=1)
                # Column normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                    dim=2)
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()
            log_alpha = log_alpha_padded[:, :-1, :-1]
        else:
            for i in range(n_iters):
                # Row normalization (i.e. each row sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))
                # Column normalization (i.e. each column sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha).clone()
        return log_alpha

    def dynamic_fc(self, features, kernel_par):
        weight = kernel_par
        return torch.nn.functional.linear(features, weight, bias=None)

    def dynamic_conv(self, features, kernel_par):
        weight = kernel_par.view(self.num_classes, -1, 1, 1)
        return torch.nn.functional.conv2d(features, weight)

    def one_hot(self, x):
        return torch.eye(self.num_classes)[x.long().cpu(), :].to(x.device)
    
    