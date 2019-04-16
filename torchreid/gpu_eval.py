import numpy as np
import torch

def per_eval(qf,ql,qc,gf,gl,gc):
    query = qf.view(-1,1)
    score = torch.pow(gf, 2).sum(dim=1) - 2.0 * torch.mm(gf,query).squeeze(1)
    score = score.cpu()
    score = score.numpy()
    index = np.argsort(score)  #from small to large
    
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        ap = ap + d_recall*precision

    return ap, cmc    

def gpu_evaluate(qf, gf, q_pids, g_pids, q_camids, g_camids):  
    torch.cuda.empty_cache()
    CMC = torch.IntTensor(len(g_pids)).zero_()
    ap = 0.0

    for i in range(len(q_pids)):
        ap_tmp, CMC_tmp = per_eval(qf[i], q_pids[i], q_camids[i], gf, g_pids, g_camids)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(q_pids) 
    mAP = ap / len(q_pids)
    return CMC.numpy(), mAP
