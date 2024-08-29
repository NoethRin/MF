import torch
import torch.nn as nn

def CLIPLoss(x, y, device):
    batch_size = x.size(0)
    x = x.reshape(batch_size, -1)
    y = y.reshape(batch_size, -1)
    logits = torch.matmul(x, y.T)
    target = torch.eye(batch_size, dtype = torch.float64).to(device)
    loss = nn.CrossEntropyLoss()
    return loss(logits, target)