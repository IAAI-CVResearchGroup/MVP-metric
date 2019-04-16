# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class LiftedLoss(nn.Module):
    def __init__(self, margin = 1.0):
        super(LiftedLoss, self).__init__()
        self.margin = margin
    
    def forward(self, score, target):
        """
          Lifted loss, per "Deep Metric Learning via Lifted Structured Feature Embedding" by Song et al
          Implemented in `pytorch`
        """

        loss = 0
        counter = 0

        bsz = score.size(0)
        mag = (score ** 2).sum(1).expand(bsz, bsz)
        sim = score.mm(score.transpose(0, 1))

        dist = (mag + mag.transpose(0, 1) - 2 * sim)
        dist = torch.nn.functional.relu(dist).sqrt()

        for i in range(bsz):
            t_i = target[i]

            for j in range(i + 1, bsz):
                t_j = target[j]

                if t_i == t_j:
                    l_ni = (self.margin - dist[i][target != t_i]).exp().sum()
                    l_nj = (self.margin - dist[j][target != t_j]).exp().sum()
                    l_n  = (l_ni + l_nj).log()

                    l_p  = dist[i,j]

                    loss += torch.nn.functional.relu(l_n + l_p) ** 2
                    counter += 1

        counter = max(1, counter)
        return loss / (2 * counter)
