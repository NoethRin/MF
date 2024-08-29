import torch
import torch.nn as nn

class RandomModule(nn.Module):
    def __init__(self, batch_size=128, device="cuda:0"):
        super(RandomModule, self).__init__()
        self.batch_size = batch_size
        self.device = device

    def forward(self, x, sidx):
        return torch.randn([self.batch_size, 1024, 150]).to(self.device)