from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import pdb
import numpy as np
import math

class KM_algorithm:
    def __init__(self, groundMetric):
        self.mp = groundMetric
        self.n = groundMetric.shape[0]
        self.link = np.zeros(self.n).astype(np.int16)
        self.lx = np.zeros(self.n)
        self.ly = np.zeros(self.n)
        self.sla = np.zeros(self.n)
        self.visx = np.zeros(self.n).astype(np.bool)
        self.visy = np.zeros(self.n).astype(np.bool)
        
    def DFS(self, x):
        self.visx[x] = True
        for y in range(self.n):
            if self.visy[y]:
                continue
            tmp = self.lx[x] + self.ly[y] - self.mp[x][y]
            if math.fabs(tmp) < 1e-5:
                self.visy[y] = True
                if self.link[y] == -1 or self.DFS(self.link[y]):
                    self.link[y] = x
                    return True
            elif self.sla[y] + 1e-5 > tmp: 
                self.sla[y] = tmp  
        return False
    
    def run(self):
        for index in range(self.n):
            self.link[index] = -1
            self.ly[index] = 0.0
            self.lx[index] = np.max(self.mp[index])
        
        for x in range(self.n):
            self.sla = np.zeros(self.n) + 1e10
            while True:
                self.visx = np.zeros(self.n).astype(np.bool)
                self.visy = np.zeros(self.n).astype(np.bool)
                if self.DFS(x): 
                    break
                d = 1e10
                for i in range(self.n):
                    if self.visy[i] == False:
                        d = min(d, self.sla[i])
                for i in range(self.n):
                    if self.visx[i]:
                        self.lx[i] -= d
                    if self.visy[i]:
                        self.ly[i] += d
                    else:
                        self.sla[i] -= d
        
        res = 0
        T = np.zeros((self.n, self.n))
        for i in range(self.n):
            if self.link[i] != -1:
                T[self.link[i]][i] = 1.0 / self.n
        return T
            
            

class MVPLoss(nn.Module):
    """
    MVPloss.
    Args:
    - margin (float): negative margin for triplet.
    - same_margin (float): pos margin for triplet.
    - use_auto_samemargin (boolean): whether to use learnable samemargin.
    """

    def __init__(self, margin = 200.0, lamb=10.0, same_margin = 0.0, no_use_km = False, use_auto_samemargin = False):
        super(MVPLoss, self).__init__()
        self.margin = margin
        self.relative_margin = margin - same_margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.use_auto_samemargin  = use_auto_samemargin
        
        if use_auto_samemargin == True:
            self.auto_samemargin = torch.autograd.Variable(torch.Tensor([same_margin]).cuda(),requires_grad=True)
        else:
            self.auto_samemargin = same_margin

    def forward(self, inputs_batch1, inputs_batch2, targets_batch1, targets_batch2, mode = 'both'):
        
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        self.mode = mode

        simLabel = self.simLabelGeneration(targets_batch1, targets_batch2)
        GMFlatten, GM = self.calculateGroundMetricContrastive(inputs_batch1, inputs_batch2, simLabel)
        KM = KM_algorithm(GM.data.cpu().numpy())
        T = KM.run()
        T_flatten = torch.autograd.Variable(torch.from_numpy(T.reshape([-1]))).float().cuda()
        loss = torch.sum(GMFlatten.mul(T_flatten))
           
        return loss

    def simLabelGeneration(self, label1, label2):
        batch_size = label1.size(0)

        label_expand_batch1 = label1.view(batch_size, 1).repeat(1, batch_size) 
        label_expand_batch2 = label2.view(batch_size, 1).repeat(batch_size, 1) 

        simLabel = torch.eq(label_expand_batch1.view(batch_size*batch_size, -1),
                            label_expand_batch2.view(batch_size*batch_size, -1)).float()
        simLabelMatrix = simLabel.view(batch_size, batch_size)

        return simLabelMatrix

    def calculateGroundMetricContrastive(self, batchFea1, batchFea2, labelMatrix):
        """
        calculate the ground metric between two batch of features
        """
        batch_size = batchFea1.size(0)
        squareBatchFea1 = torch.sum(batchFea1.pow(2), 1)
        squareBatchFea1 = squareBatchFea1.view(batch_size, -1)

        squareBatchFea2 = torch.sum(batchFea2.pow(2), 1)
        squareBatchFea2 = squareBatchFea2.view(-1, batch_size)

        correlationTerm = batchFea1.mm(batchFea2.t())

        groundMetric = squareBatchFea1 - 2 * correlationTerm + squareBatchFea2

        hinge_groundMetric = torch.max(torch.zeros(batch_size, batch_size).cuda(), self.auto_samemargin + self.relative_margin - groundMetric)                                           
        same_class_groundMetric = torch.max(torch.zeros(batch_size, batch_size).cuda(), groundMetric - self.auto_samemargin)

        GM_positivePair = labelMatrix.mul(same_class_groundMetric)
        GM_negativePair = (1 - labelMatrix).mul(hinge_groundMetric)
        
        if self.mode == 'both':
            GM = GM_negativePair + GM_positivePair 
        elif self.mode == 'pos':
            GM = GM_positivePair + (1 - labelMatrix).mul(-10000000.0)
        else:
            GM = GM_negativePair + labelMatrix.mul(-10000000.0)
        GMFlatten = GM.view(-1)
        return GMFlatten, GM