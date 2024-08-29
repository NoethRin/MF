import torch.nn as nn

class fMRIModule(nn.Module):
    def __init__(self):
        super(fMRIModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(24375, 1024, bias=True),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 1024, bias=True),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 1024, bias=True),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 1024)
        )

    def forward(self, fmri):
        return self.mlp(fmri)