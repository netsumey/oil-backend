import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Lstm = nn.LSTM(configs.enc_in, configs.enc_in)
        self.fc = nn.Linear(configs.seq_len, configs.pred_len)

        # self.Linear2 = nn.LSTM(self.seq_len, self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x,x1=None,x2=None,x3=None,x4=None):
        # x: [Batch, Input length, Channel]
        x = self.Lstm(x.permute(1,0,2))[0].permute(1,0,2)
        x = self.fc(x.permute(0,2,1)).permute(0,2,1)
        # print('aa', x.shape)
        # x = self.Linear2(x.permute(0,2,1)).permute(0,2,1)

        return x[:, -self.pred_len:, :] # [Batch, Output length, Channel]

