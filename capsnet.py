import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms

class PrimaryCaps(nn.Module):
    def __init__(self,in_channels=256,stride=2,kernel=9,vector_dim=8,caps_unit=32):
        super(PrimaryCaps, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=vector_dim*caps_unit,kernel_size=kernel,stride=stride)
        #He-Normalize
        nn.init.kaiming_normal_(self.conv.weight)
        
    def forward(self,x):
        x = self.conv(x)
        return x

class DigitCaps(nn.Module):
    def __init__(self,num_classes,num_capsuel,out_capsuel_dim,in_capsuel_dim,num_routing):
        super(DigitCaps,self).__init__()
        self.num_routing = num_routing
        self.num_classes = num_classes
        self.out_capsuel_dim = out_capsuel_dim
        self.W = nn.Parameter(torch.Tensor(num_capsuel,in_capsuel_dim,num_classes*out_capsuel_dim))
        nn.init.kaiming_normal_(self.W)
    def forward(self,x):
        batch_num,input_capsuel_num,_ = x.shape
        x = torch.stack([x]* self.num_classes*self.out_capsuel_dim,dim=3)
        
        u_hat = torch.sum(x*self.W,dim=2)
        u_hat = u_hat.view(batch_num,-1,self.num_classes,self.out_capsuel_dim)
        u_hat_detach = u_hat.detach()
        u_hat_trans = u_hat.permute(3, 0, 1, 2)
        u_hat_detach_trans = u_hat_trans.detach()
        
        routing_biases = Variable(0.1*torch.ones(batch_num,self.num_classes,self.out_capsuel_dim)).to("cuda")#[output_dim, output_atoms]
        b = Variable(torch.zeros(batch_num,input_capsuel_num,self.num_classes)).to("cuda")
        for i in range(self.num_routing):
            #[B,1152,10]→[B,1152,10]
            c = F.softmax(b, dim=2)
            if i==self.num_routing -1:
                s = c*u_hat_trans
                s = s.permute(1,2,3,0)
                s = torch.sum(s, axis=1) + routing_biases
                v = self._squash(s)
            else:
                s = c*u_hat_detach_trans
                s = s.permute(1,2,3,0)
                s = torch.sum(s, dim=1) + routing_biases
                v = self._squash(s)
                s = torch.stack([s]*input_capsuel_num,dim=1)
                distances = torch.sum(u_hat_detach*s,dim=3)
                b += distances
        return v
    
    def _squash(self,s,dim=2):
        norm = torch.norm(s, dim=2, keepdim=True)
        norm_sq = norm**2 
        tmp = s/ (1 + norm**2) / norm
        return tmp*norm_sq

class CapsNet(nn.Module):
    def __init__(self,in_capsuel_dim=8,out_capsuel_dim=16,num_classes=10,num_routing=3):
        super(CapsNet,self).__init__()
        self.in_capsuel_dim = in_capsuel_dim
        self.out_capsuel_dim = out_capsuel_dim
        self.num_classes = num_classes
        
        self.conv = nn.Conv2d(in_channels = 1,out_channels=256,kernel_size=9,stride=1)
        #He-Normalize
        nn.init.kaiming_normal_(self.conv.weight)
        self.prime = PrimaryCaps()
        self.digit = DigitCaps(num_classes,1152,out_capsuel_dim,in_capsuel_dim,num_routing)
    
    def forward(self,x):
        #[B,1,28,28]→[B,256,20,20]
        x = self.conv(x)
        #[B,256,20,20]→[B,256,6,6]
        x = self.prime(x)
        #[B,256,6,6]→[B,1152,8]
        x = x.view(x.shape[0],-1,self.in_capsuel_dim)
        #[B,1152,8]→[B,10,16]
        x = self.digit(x)
        return x.norm(dim=-1)

def margin_loss(labels, raw_logits, margin=0.4, downweight=0.5):
    logits = raw_logits - 0.5
    margin = margin*torch.ones(logits.shape).to("cuda")
    positive_cost = labels * torch.gt(-logits, -margin) * torch.pow(logits - margin, 2)#torch.gtらへんあやしい？
    negative_cost = (1 - labels) * torch.gt(logits, -margin) * torch.pow(logits + margin, 2)
    L_margin = 0.5 * positive_cost + downweight * 0.5 * negative_cost
    L_margin = L_margin.sum(dim=1).mean()
    return L_margin