import torch
from torch.utils.data import IterableDataset, DataLoader
from itertools import chain
from nanodrz import data
from nanodrz.data import GeneratorIterableDataset
from nanodrz.model import DiarizeGPT


# model = DiarizeGPT()

# ds = GeneratorIterableDataset(data.artificial_drz_generator(model))

ds2 = data.voxconverse_dev()
for i in range(len(ds2)):
    x = ds2[i]
