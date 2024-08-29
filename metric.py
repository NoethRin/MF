import torch
import numpy as np
from scipy.stats import pearsonr

def pearson(x, y, task, target):
    corr = []
    p_val = []
    if task == "fmri" or (task == "meg" and target == "emb"):
        for index in range(x.shape[1]):
            res1, res2 = pearsonr(x[:, index].detach().numpy(), y[:, index].detach().numpy())
            corr.append(res1)
            p_val.append(res2)
    else:
        for index in range(x.shape[0]):
            res1, res2 = pearsonr(x[index, :].detach().numpy(), y[index, :].detach().numpy())
            corr.append(res1)
            p_val.append(res2)
    return torch.tensor(corr).mean().item(), torch.tensor(p_val).mean().item()

def topacc(x, y, device):
    batch_size = x.size(0)
    x = x.reshape(batch_size, -1)
    y = y.reshape(batch_size, -1)
    logits = torch.matmul(x, y.T)
    targets = torch.arange(batch_size, requires_grad=False).long().to(device)
    top1accuracy = np.mean([label in row for row, label in zip(torch.topk(logits, 1, dim=1, largest=True)[1], targets)])
    top5accuracy = np.mean([label in row for row, label in zip(torch.topk(logits, 5, dim=1, largest=True)[1], targets)])
    top10accuracy = np.mean([label in row for row, label in zip(torch.topk(logits, 10, dim=1, largest=True)[1], targets)])
    return top1accuracy, top5accuracy, top10accuracy

def rankacc(x, y):
    batch_size = x.size(0)
    x = x.reshape(batch_size, -1)
    y = y.reshape(batch_size, -1)
    logits = torch.matmul(x, y.T)
    index = torch.argsort(-logits, dim=-1).cpu()
    pos = torch.arange(logits.shape[0]).unsqueeze(dim=1)
    rank = (index == pos).nonzero()[:, 1].to(torch.float32)
    return 1 - rank.mean() / (logits.shape[1] - 1)