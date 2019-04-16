#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017

@author: luohao
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API

probFea: all feature vectors of the query set, shape = (image_size, feature_dim)
galFea: all feature vectors of the gallery set, shape = (image_size, feature_dim)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""


import numpy as np
from scipy.spatial.distance import cdist
import torch

def re_ranking(probFea,galFea,k1,k2,lambda_value, MemorySave = False, Minibatch = 2000):
    print('lambda_value is', lambda_value)
    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]    
    feat = np.append(probFea,galFea,axis = 0)
    feat = torch.cuda.HalfTensor(feat)
    print('computing original distance')
    if MemorySave:
        original_dist = np.zeros(shape = [all_num,all_num],dtype = np.float16)
        i = 0
        while True:
            it = i + Minibatch
            if it < np.shape(feat)[0]:
                original_dist[i:it,] = np.power(cdist(feat[i:it,],feat),2).astype(np.float16)
            else:
                original_dist[i:,:] = np.power(cdist(feat[i:,],feat),2).astype(np.float16)
                break
            i = it
    else:
        torch.cuda.empty_cache()
        original_dist = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
        original_dist = (original_dist + original_dist.t() - 2 * feat.mm(feat.t()))
        original_dist = torch.transpose(original_dist / torch.max(original_dist, 0)[0], 1, 0).cpu().numpy()
        torch.cuda.empty_cache()


        
    gallery_num = original_dist.shape[0]
    #original_dist = np.transpose(original_dist/np.max(original_dist,axis = 0))

    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)#升序排列(返回为索引)

    
    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1] #i的k1近邻
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1] #i的k1近邻的k1近邻
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi] #互为k1近邻

        # expansion
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            # 求k_reciprocal_index[j]的互k1/2近邻
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            # 如果candidate_k_reciprocal_index与k_reciprocal_index的交集足够大，
            # 则确认加入candidate_k_reciprocal_index
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)

        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)

    original_dist = original_dist[:query_num,]    
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float16)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0) # V_qe: local query expansion of V (即paper中的Eq.11)
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    # 计算Jaccard distance
    jaccard_dist = np.zeros_like(original_dist,dtype = np.float16)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float16)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)

    # 组合original distance和Jaccard distance
    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value

    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist


