import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureEncoderBlock(nn.Module):
    def __init__(self, k, C=320):
        super(FeatureEncoderBlock, self).__init__()
    
        tmp1 = np.power(2, (2 * k) % 5)
        tmp2 = np.power(2, (2 * k + 1) % 5)
        self.lev = k
        
        if k == 1:
            self.conv1 = nn.Conv1d(in_channels=270, out_channels=320, kernel_size=3, stride=1, dilation=tmp1, padding=tmp1)
        else:
            self.conv1 = nn.Conv1d(in_channels=320, out_channels=320, kernel_size=3, stride=1, dilation=tmp1, padding=tmp1)
        self.conv2 = nn.Conv1d(in_channels=320, out_channels=320, kernel_size=3, stride=1, dilation=tmp2, padding=tmp2)
        self.conv3 = nn.Conv1d(in_channels=320, out_channels=640, kernel_size=3, stride=1, padding=1)
       
        self.bn1 = nn.BatchNorm1d(num_features=C)
        self.bn2 = nn.BatchNorm1d(num_features=C)
       
    def forward(self, X):
        if self.lev == 1:
            Y = F.gelu(self.bn1(self.conv1(X)))
        else:
            Y = F.gelu(self.bn1(self.conv1(X) + X))
        Y = F.gelu(self.bn2(self.conv2(Y) + Y))
        Y = F.glu(self.conv3(Y), 1)
        return Y
    
class MEGModule(nn.Module):
    def __init__(self, channel_num=204, subject_num=13):
        super(MEGModule, self).__init__()
        self.spatial_attention = nn.Linear(channel_num, 270, bias=False)
        self.inter_conv = nn.Conv1d(in_channels=270, out_channels=270, kernel_size=1)
        self.subject_layer = nn.Embedding(num_embeddings=subject_num, embedding_dim=270)
        self.FeatureEncoder = nn.Sequential(
            FeatureEncoderBlock(1),
            FeatureEncoderBlock(2),
            FeatureEncoderBlock(3),
            FeatureEncoderBlock(4),
            FeatureEncoderBlock(5))
        self.finconv1 = nn.Conv1d(in_channels=320, out_channels=640, kernel_size=1)
        self.finconv2 = nn.Conv1d(in_channels=640, out_channels=1024, kernel_size=1)
        self.GELU = nn.GELU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, MEG, index):
        MEG = MEG.transpose(1, 2)
        MEG = self.spatial_attention(MEG)
        MEG = MEG.transpose(1, 2)
        MEG = self.inter_conv(MEG) 
        MEG = MEG.transpose(1, 2)
        MEG = MEG * self.subject_layer(index).unsqueeze(dim=1)
        MEG = MEG.transpose(1, 2)
        MEG = self.FeatureEncoder(MEG)
        MEG = self.finconv1(MEG)
        MEG = self.finconv2(MEG)
        MEG = self.GELU(MEG)
        return self.maxpool(MEG)