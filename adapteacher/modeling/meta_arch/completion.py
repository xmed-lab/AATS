import torch
from torch import nn
import torch.nn.functional as F

import sklearn.cluster as cluster

def update_seed(self, features, labels, boxes, centers):

    k = 20  # conduct clustering when we have enough graph nodes
    for cls in sr_labels.unique().long():
        bs = sr_nodes[sr_labels == cls].detach()

        if len(bs) > k and self.with_cluster_update:
            # TODO Use Pytorch-based GPU version
            sp = cluster.SpectralClustering(2, affinity='nearest_neighbors', n_jobs=-1,
                                            assign_labels='kmeans', random_state=1234, n_neighbors=len(bs) // 2)
            seed_cls = self.sr_seed[cls]
            indx = sp.fit_predict(torch.cat([seed_cls[None, :], bs]).cpu().numpy())
            indx = (indx == indx[0])[1:]
            bs = bs[indx].mean(0)
        else:
            bs = bs.mean(0)

        momentum = torch.nn.functional.cosine_similarity(bs.unsqueeze(0), self.sr_seed[cls].unsqueeze(0))
        self.sr_seed[cls] = self.sr_seed[cls] * momentum + bs * (1.0 - momentum)

    if tg_nodes is not None:
        for cls in tg_labels.unique().long():
            bs = tg_nodes[tg_labels == cls].detach()
            if len(bs) > k and self.with_cluster_update:
                seed_cls = self.tg_seed[cls]
                sp = cluster.SpectralClustering(2, affinity='nearest_neighbors', n_jobs=-1,
                                                assign_labels='kmeans', random_state=1234, n_neighbors=len(bs) // 2)
                indx = sp.fit_predict(torch.cat([seed_cls[None, :], bs]).cpu().numpy())
                indx = (indx == indx[0])[1:]
                bs = bs[indx].mean(0)
            else:
                bs = bs.mean(0)
            momentum = torch.nn.functional.cosine_similarity(bs.unsqueeze(0), self.tg_seed[cls].unsqueeze(0))
            self.tg_seed[cls] = self.tg_seed[cls] * momentum + bs * (1.0 - momentum)

def feature_comple():
     sr_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).to(tg_nodes_c.device) + sr_nodes_c \
        if len(tg_nodes_c) < 5 \
            else torch.normal(mean=sr_nodes_c,
                std=tg_nodes_c.std(0).unsqueeze(0).expand(sr_nodes_c.size())).to(sr_nodes_c.device)
