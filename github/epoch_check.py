import torch

path = "/home/jykim/Project/updated_SRGAN/results/0721GAN_x4-DIV2K/g_best.pth.tar"
model = torch.load(path)
print(model["epoch"])