#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2021/01/07 23:59:35
@Author  :   Zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid,  dropout=0.1):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_in, d_hid)
        self.w2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x.shape: (batch, seqlen, d_in)
        residual = x

        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        return x

batch_size = 64
seq_len = 20
d_model = 512
dff = 2048
data = torch.randn(batch_size, seq_len, d_model)

model = FeedForward(d_model, dff)

print(data.shape)
print(model(data).shape)