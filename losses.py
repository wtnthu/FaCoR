import torch, pdb
import numpy as np
from utils import *



def contrastive_loss_att(x1,x2,beta=0.08):
    x1x2=torch.cat([x1,x2],dim=0)
    x2x1=torch.cat([x2,x1],dim=0)

    # a1 = torch.max(beta, 1)[0]
    # a2 =  torch.max(a1, 1)[0]

    # beta = torch.cat([a2, a2]).reshape(-1)
    # m=0.02
    m=0
    
    beta1, beta2, beta_e, beta_n, beta_m = beta

    beta1 = (beta1**2).sum([1,2])
    beta2 = (beta2**2).sum([1,2])
    beta_e = (beta_e**2).sum([1,2])
    beta_n = (beta_n**2).sum([1,2])
    beta_m = (beta_m**2).sum([1,2])
    # pdb.set_trace()
    beta = (beta_e+beta_n+beta_m)/3/12
    # beta = (beta1+beta2+beta_e+beta_n+beta_m)/5/1.3

    beta = torch.cat([beta, beta])


    cosine_mat=torch.cosine_similarity(torch.unsqueeze(x1x2,dim=1),
                                       torch.unsqueeze(x1x2,dim=0),dim=2)/(beta+m)
    mask=1.0-torch.eye(2*x1.size(0))
    numerators = torch.exp(torch.cosine_similarity(x1x2,x2x1,dim=1)/(beta+m))
    denominators=torch.sum(torch.exp(cosine_mat)*mask,dim=1)
    return -torch.mean(torch.log(numerators/denominators),dim=0)

def contrastive_loss(x1,x2,beta=0.08):
    x1x2=torch.cat([x1,x2],dim=0)
    x2x1=torch.cat([x2,x1],dim=0)

    # a1 = torch.max(beta, 1)[0]
    # a2 =  torch.max(a1, 1)[0]

    # beta = torch.cat([a2, a2]).reshape(-1)
    # m=0.02
    m=0.0
    # pdb.set_trace()
    beta = (beta**2).sum([1, 2])/500
    # beta = (beta**2).sum([1, 2])/20#18
    

    beta = torch.cat([beta, beta]).reshape(-1)
    cosine_mat=torch.cosine_similarity(torch.unsqueeze(x1x2,dim=1),
                                       torch.unsqueeze(x1x2,dim=0),dim=2)/(beta+m)
    mask=1.0-torch.eye(2*x1.size(0))
    numerators = torch.exp(torch.cosine_similarity(x1x2,x2x1,dim=1)/(beta+m))
    denominators=torch.sum(torch.exp(cosine_mat)*mask,dim=1)
    return -torch.mean(torch.log(numerators/denominators),dim=0)


def contrastive_loss_init(x1,x2,beta=0.08):
    x1x2=torch.cat([x1,x2],dim=0)
    x2x1=torch.cat([x2,x1],dim=0)

    # a1 = torch.max(beta, 1)[0]
    # a2 =  torch.max(a1, 1)[0]

    # beta = torch.cat([a2, a2]).reshape(-1)
    # m=0.02
    m=0
    # beta = (beta**2).sum([1, 2])/12
    # beta = torch.cat([beta, beta]).reshape(-1)
    cosine_mat=torch.cosine_similarity(torch.unsqueeze(x1x2,dim=1),
                                       torch.unsqueeze(x1x2,dim=0),dim=2)/(beta+m)
    mask=1.0-torch.eye(2*x1.size(0))
    numerators = torch.exp(torch.cosine_similarity(x1x2,x2x1,dim=1)/(beta+m))
    denominators=torch.sum(torch.exp(cosine_mat)*mask,dim=1)
    return -torch.mean(torch.log(numerators/denominators),dim=0)