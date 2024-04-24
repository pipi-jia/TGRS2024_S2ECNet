import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np


criterionCEre = nn.CrossEntropyLoss(reduce=False)


def ACE(featureidx, numsamples, classifier, feature, targetm, outdim, device):
    bs, zdim = feature.shape
    zdo = torch.randn(numsamples, bs, zdim).to(device)
    zdo[:,:,featureidx] = feature[:,featureidx]
    sample = classifier(zdo.view(numsamples*bs, zdim))
    ACEdo = sample.view(numsamples, bs, -1).mean(0)

    zrand=torch.randn(numsamples, bs, zdim).to(device)
    sample=classifier(zrand.view(numsamples*bs, zdim))
    ACEbaseline = sample.view(numsamples, bs, -1).mean(0)
    ace = ACEbaseline - ACEdo
        
    return(ace)


def contrastive_ace(numsamples, classifier, feature, targetm, outdim, anchorbs, device):
    numfeature = feature.shape[1]
    ace = []
    for i in range(numfeature):
        ace.append(ACE(i, numsamples, classifier, feature, targetm, outdim, device))
    
    acematrix = torch.stack(ace,dim=1) / (torch.stack(ace,dim=1).norm(dim=1).unsqueeze(1) +1e-8)  # [bs, num_feature]
    anchor = acematrix[:anchorbs] / acematrix[:anchorbs].norm(1)
    neighbor = acematrix[anchorbs:2*anchorbs] / acematrix[anchorbs:2*anchorbs].norm(1)
    distant = acematrix[2*anchorbs:] / acematrix[2*anchorbs:].norm(1)
    sigma = torch.stack(ace,dim=1).sum()

    margin = 0.02
    pos = (torch.abs(anchor - neighbor)).sum()
    neg = (torch.abs(anchor - distant)).sum()
    contrastive_loss = F.relu(pos - neg + margin)
        
    return sigma, contrastive_loss