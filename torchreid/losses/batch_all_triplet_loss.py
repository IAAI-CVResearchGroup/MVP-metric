# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import pdb

class BA_TripletLoss(nn.Module):
    """Triplet loss with Batch_all.

    """

    def __init__(self, margin=200.0):
        super(BA_TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())  
        dist = dist.clamp(min=1e-12)  

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())  
        loss_sum = 0
        count = 0
        for i in range(n):
            for ind in range(len(dist[i][mask[i]])):
                dist_an = []
                if dist[i][mask[i]][ind].item() > 1e-9:
                    dist_ap = dist[i][mask[i]][ind]
                else:
                    continue
                count += 1
                dist_an.append(dist[i][mask[i] == 0].unsqueeze(0))
                dist_an = torch.cat(dist_an)
                neg_pairs = dist_an.size(1)

                loss_ = torch.max(torch.zeros_like(dist_an), dist_ap - dist_an + self.margin)
                loss_sum += torch.sum(loss_)
        loss = loss_sum / (count*neg_pairs)
        return loss
