from nanodrz.modules import Attention
import torch

att = Attention(1024, 16, .2)
att(torch.rand(1, 64, 1024), None)