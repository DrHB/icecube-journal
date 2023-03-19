import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_functions import VonMisesFisher3DLoss

def loss(pred,y):
    #print(pred.max())
    pred = F.normalize(pred.float(),dim=-1)
    
    sa2 = torch.sin(y['target'][:,0])
    ca2 = torch.cos(y['target'][:,0])
    sz2 = torch.sin(y['target'][:,1])
    cz2 = torch.cos(y['target'][:,1])
    
    scalar_prod = (pred[:,0]*sa2*sz2 + pred[:,1]*ca2*sz2 + pred[:,2]*cz2).clip(-1+1e-8,1-1e-8)
    return torch.acos(scalar_prod).abs().mean(-1)   

def loss_vms(pred,y):
    sa2 = torch.sin(y['target'][:,0])
    ca2 = torch.cos(y['target'][:,0])
    sz2 = torch.sin(y['target'][:,1])
    cz2 = torch.cos(y['target'][:,1])
    t = torch.stack([sa2*sz2,ca2*sz2,cz2],-1)
    
    p = pred.float()
    l = torch.norm(pred.float(),dim=-1).unsqueeze(-1)
    p = torch.cat([pred.float()/l,l],-1)
    
    loss = VonMisesFisher3DLoss()(p,t)
    return loss

def get_val(pred):
    pred = F.normalize(pred,dim=-1)
    zen = torch.acos(pred[:,2].clip(-1,1))
    f = F.normalize(pred[:,:2],dim=-1)
    az = torch.asin(f[:,0].clip(-1,1))
    az = torch.where(f[:,1] > 0, az, math.pi - az)
    az = torch.where(az > 0, az, az + 2.0*math.pi)
    return torch.stack([az,zen],-1)
