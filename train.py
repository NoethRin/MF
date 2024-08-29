import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import data.config as config
from utils import fixseed
from data.data_entry import select_train_loader, select_val_loader
from model.model_entry import select_model
from options import get_train_args
from metric import pearson, topacc, rankacc
from loss import CLIPLoss

class Trainer:
    def __init__(self):
        self.args = get_train_args()
        fixseed(self.args.seed)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpus

        self.train_loader = select_train_loader(self.args)
        self.val_loader = select_val_loader(self.args)

        self.model = select_model(self.args).to(self.args.cuda)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr)
        
        self.save_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", config.subs[self.args.subject], self.args.task, "model")
        os.makedirs(self.save_location, exist_ok=True)

    def train(self):
        cnt = 10
        bestrank = -1e6
        for epoch in range(1, self.args.epochs + 1):
            if cnt <= 0:
                break
            self.train_per_epoch(epoch)
            rank = self.val_per_epoch()
            if rank > bestrank:
                torch.save(self.model.state_dict(), os.path.join(self.save_location, "model.pt"))
                bestrank = rank
                cnt = 10
            else:
                cnt -= 1

        print("training process is terminated!")

    def train_per_epoch(self, epoch):
        self.model.train()
        train_loss = []
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            pred, y = self.step(data)
            loss = self.compute_loss(pred, y)
            
            train_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()

            if (batch_idx + 1) % self.args.print_freq == 0:
                print("train loss: ", loss.item())
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, torch.tensor(train_loss).mean()))

    def val_per_epoch(self):
        self.model.eval()
        with torch.no_grad():
            rank = []
            for data in self.val_loader:
                pred, y = self.step(data)
                rank.append(self.compute_metrics(pred, y))
            return torch.tensor(rank).mean()

    
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
    
    def compute_metrics(self, pred, y):
        top1, top5, top10 = topacc(pred, y, self.args.cuda)
        rank = rankacc(pred, y)
        print("top-1/5/10: {}, {}, {}; rankacc: {}".format(top1, top5, top10, rank))
        return rank
    
    def compute_loss(self, pred, y):
        return 0.5 * CLIPLoss(pred, y, self.args.cuda) + 0.5 * F.mse_loss(pred, y)
    
def main():
    trainer = Trainer()
    trainer.train()

if __name__ == '__main__':
    main() 