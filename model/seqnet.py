#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class SeqNet(nn.Module):
    '''SeqNet: Learning Descriptors for Sequence-Based Hierarchical Place Recognition IEEE RA-L & ICRA 2021'''
    def __init__(self, inDims, outDims, w=1):

        super(SeqNet, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):
        
        if len(x.shape) < 3:
            x = x.unsqueeze(1) # convert [B,C] to [B,1,C]

        x = x.permute(0,2,1) # from [B,T,C] to [B,C,T]
        seqFt = self.conv(x)
        seqFt = torch.mean(seqFt,-1)

        return seqFt