import os
import torch
import torch.nn as nn
from metric import pearson

class LinearModule(nn.Module):
    def __init__(self):
        super(LinearModule, self).__init__()
        self.alphas = torch.logspace(0.5, 20, 20, dtype=torch.float32)
        self.path = os.path.join()

    def predict(self, *kwargs):
        pass

    def forward(self, train_x, train_y, test_x, test_y, task_name):
        xy = torch.matmul(train_x.T, train_y)
        del train_y
        xx = torch.matmul(train_x.T, train_x)
        del train_x
        ii = torch.eye(xx.shape[0])

        corr, p_val = [], []
        for alpha in self.alphas:
            weights = torch.matmul(torch.linalg.inv(xx + alpha * ii), xy)
            predictions = test_x @ weights
            res1, res2 = pearson(predictions, test_y)
            corr.append(res1)
            p_val.append(res2)
        corr = torch.tensor(corr)
        p_val = torch.tensor(p_val)

        results = {}
        results["corr_mean"] = corr.mean()
        results["corr_std"] = corr.std()
        results["p_val_mean"] = p_val.mean()
        results["p_val_std"] = p_val.std()
        results["best_result"] = (torch.max(corr), p_val[torch.argmax(corr)], self.alphas[torch.argmax(corr)])
        torch.save(results, os.path.join(self.path, f"{task_name}_linear.pt"))