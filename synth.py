import os
import torch
import data.config as config

# for sub in config.subs:
#     final = []
#     for index in range(43 + 1):
#         final.append(torch.load(f"/mnt/Disk2/zak/ICASSP2025/outputs/{sub}/pa/fmri_{index}.pt"))
#     torch.save(torch.vstack(final), os.path.join(f"/mnt/Disk2/zak/ICASSP2025/data/results/{sub}", "prefmri.pt"))

#     final = []
#     for index in range(43 + 1):
#         final.append(torch.load(f"/mnt/Disk2/zak/ICASSP2025/outputs/{sub}/pa/meg_{index}.pt"))
#     torch.save(torch.vstack(final), os.path.join(f"/mnt/Disk2/zak/ICASSP2025/data/results/{sub}", "premeg.pt"))

#     os.system(f"rm -rf /mnt/Disk2/zak/ICASSP2025/outputs/{sub}/pa")

final = []
data = torch.load("/mnt/Disk2/zak/Sep9/data/results/emb/emb.pt")
for story in range(42):
    final.append(data[story])
for story in range(48, 60):
    final.append(data[story])
torch.save(torch.vstack(final), "/mnt/Disk2/zak/ICASSP2025/data/results/emb/emb.pt")

# final = []
# data = torch.load("/mnt/Disk2/zak/Sep9/data/results/mel/mel.pt")
# for story in range(42):
#     final.append(data[story])
# for story in range(48, 60):
#     final.append(data[story])
# torch.save(torch.vstack(final), "/mnt/Disk2/zak/ICASSP2025/data/results/mel/mel.pt")