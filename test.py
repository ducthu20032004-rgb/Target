import torch
ckpt = torch.load("checkpoint/checkpoint_task0.pth_0.pkl")
print(ckpt.keys())