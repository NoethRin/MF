import os
import torch
import data.config as config
from tqdm import tqdm
from utils import fixseed
from data.data_entry import select_eval_loader
from model.model_entry import select_model
from options import get_test_args
from metric import topacc, rankacc, pearson

class Evaluator:
    def __init__(self):
        self.args = get_test_args()
        fixseed(self.args.seed)

        self.model = select_model(self.args)
        if self.args.task != "random":
            self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", config.subs[self.args.subject], self.args.task, "model", "model.pt")))
        self.model = self.model.to(self.args.cuda)

        self.model.eval()

        self.eval_loader = select_eval_loader(self.args)
        self.save_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", config.subs[self.args.subject], "acc", self.args.task)
        os.makedirs(self.save_location, exist_ok=True)

    def eval(self):
        metrics = {}
        top1, top5, top10, rank = [], [], [], []
        for data in self.eval_loader:
            pred, label = self.step(data)
            top_1, top_5, top_10, rank_acc = self.compute_metrics(pred, label)
            top1.append(top_1)
            top5.append(top_5)
            top10.append(top_10)
            rank.append(rank_acc)
        metrics["top-1_mean"] = torch.tensor(top1).mean()
        metrics["top-1_std"] = torch.tensor(top1).std()
        metrics["top-5_mean"] = torch.tensor(top5).mean()
        metrics["top-5_std"] = torch.tensor(top5).std()
        metrics["top-10_mean"] = torch.tensor(top10).mean()
        metrics["top-10_std"] = torch.tensor(top10).std()
        metrics["rankacc_mean"] = torch.tensor(rank).mean()
        metrics["rankacc_std"] = torch.tensor(rank).std()
        torch.save(metrics, os.path.join(self.save_location, "res.pt"))

    def compute_metrics(self, pred, y):
        top1, top5, top10 = topacc(pred, y, self.args.cuda)
        rank = rankacc(pred, y)
        print("top-1/5/10: {}, {}, {}; rankacc: {}".format(top1, top5, top10, rank))
        return top1, top5, top10, rank
    
    def step(self, data):
        if self.args.task == "fusion":
            meg, fmri, wav, index = data
            meg = meg.to(self.args.cuda)
            fmri = fmri.to(self.args.cuda)
            wav = wav.to(self.args.cuda)
            index = torch.Tensor(index).to(self.args.cuda)
            pred = self.model(fmri, meg, index)
        elif self.args.task == "meg" or self.args.task == "random":
            x, wav, index = data
            x = x.to(self.args.cuda)
            wav = wav.to(self.args.cuda)
            index = torch.Tensor(index).to(self.args.cuda)
            pred = self.model(x, index)
        elif self.args.task == "fmri":
            x, wav = data
            x = x.to(self.args.cuda)
            wav = wav.to(self.args.cuda)
            pred = self.model(x)
        return pred, wav

def eval_main():
    evaluator = Evaluator()
    evaluator.eval()


if __name__ == '__main__':
    eval_main()
