# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, margin=200, same_margin = 200, diff_margin = 400, no_use_norm_triplet = False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.same_margin = same_margin
        self.diff_margin = diff_margin
        self.no_use_norm_triplet = no_use_norm_triplet

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())  # 1*dist + (-2)*inputs@inputs.t()  @:矩阵乘
      #  dist = dist.clamp(min=1e-12)  # for numerical stability #clamp:限制元素范围，大于min

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())  # eq：比较元素是否相等
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # unsqueeze(0)在第0维增加一个维度
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        if self.no_use_norm_triplet == False:
            y = torch.ones_like(dist_an)  # 维度与dist_an相同的全为1的矩阵
            self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
            loss = self.ranking_loss(dist_an, dist_ap, y)  # max(0,-y*(dist_an-dist_ap)+margin)
        else:
            y = torch.ones_like(dist_an)
            self.ranking_loss = nn.MarginRankingLoss(margin=-self.same_margin)
            loss = self.ranking_loss(torch.zeros_like(dist_an), dist_ap, y)  # max(0, y * (dist_ap) - same_margin)
            self.ranking_loss = nn.MarginRankingLoss(margin=self.diff_margin)
            loss = loss + self.ranking_loss(dist_an, torch.zeros_like(dist_ap), y) # max(0, -y *(dist_an) + diff_margin)
        return loss