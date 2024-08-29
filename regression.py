import os
import torch
import argparse
import data.config as config
from sklearn.model_selection import KFold
from metric import pearson

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=0)
    parser.add_argument("--xtype", type=str, default="fmri")
    parser.add_argument("--ytype", type=str, default="mel")
    args = parser.parse_args()

    save_dir = os.path.join(f"/mnt/Disk2/zak/ICASSP2025/outputs/{config.subs[args.subject]}/acc", "pa")
    os.makedirs(save_dir, exist_ok=True)

    alphas = torch.logspace(0.5, 20, 20, dtype=torch.float32)
    if args.xtype == "meg" and args.ytype == "mel":
        x = torch.load(f"/mnt/Disk2/zak/ICASSP2025/data/results/{config.subs[args.subject]}/pre{args.xtype}.pt").cpu().mean(axis=1)
    elif args.xtype == "meg" and args.ytype == "emb":
        x = torch.load(f"/mnt/Disk2/zak/ICASSP2025/data/results/{config.subs[args.subject]}/pre{args.xtype}.pt").cpu().mean(axis=-1)
    else:
        x = torch.load(f"/mnt/Disk2/zak/ICASSP2025/data/results/{config.subs[args.subject]}/pre{args.xtype}.pt").cpu()
    y = torch.load(f"/mnt/Disk2/zak/ICASSP2025/data/results/{args.ytype}/{args.ytype}.pt").cpu()
    kf = KFold(n_splits=5)
    final_res = {"corr": [], "pval": []}
    for i, (train_index, test_index) in enumerate(kf.split(y)):
        train_x = x[train_index, :]
        test_x = x[test_index, :]

        train_y = y[train_index, :]
        test_y = y[test_index, :]

        xy = torch.matmul(train_x.T, train_y)
        del train_y
        xx = torch.matmul(train_x.T, train_x)
        del train_x
        ii = torch.eye(xx.shape[0])

        corr, p_val = [], []
        for alpha in alphas:
            weights = torch.matmul(torch.linalg.inv(xx + alpha * ii), xy)
            predictions = test_x @ weights
            res1, res2 = pearson(predictions, test_y, args.xtype, args.ytype)
            print(res1, res2)
            corr.append(res1)
            p_val.append(res2)
        corr = torch.tensor(corr)
        p_val = torch.tensor(p_val)
        final_res["corr"].append(torch.max(corr))
        final_res["pval"].append(p_val[torch.argmax(corr)])

    final_res["corr"] = torch.tensor(final_res["corr"]).mean()
    final_res["pval"] = torch.tensor(final_res["pval"]).mean()
    torch.save(final_res, os.path.join(save_dir, f"res_{args.xtype}_{args.ytype}.pt"))