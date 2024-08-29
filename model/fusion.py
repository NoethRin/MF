import torch.nn as nn
from model.brainmagic import MEGModule
from model.mlp import fMRIModule

class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()
        self.mmodule = MEGModule()
        self.fmodule = fMRIModule()

    def forward(self, fmri, meg, subindex):
        return self.fmodule(fmri).unsqueeze(dim=-1) + self.mmodule(meg, subindex)