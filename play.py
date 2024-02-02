import torch
from torch.utils.data import IterableDataset, DataLoader
from itertools import chain
from nanodrz import data
from nanodrz.data import GeneratorIterableDataset
from nanodrz.model import DiarizeGPT
from tqdm.contrib.concurrent import thread_map
from multiprocessing import Pool
from tqdm import tqdm
from nanodrz.data import Speaker, Utterance
from torch.nn import functional as F
import itertools
from os.path import basename, expanduser
import glob
from nanodrz.utils import find_nonsilence_chunks

# model = DiarizeGPT()

# ds = GeneratorIterableDataset(data.artificial_drz_generator(model))


# ds2 = data.voxconverse_dev()
# for i in range(len(ds2)):
#     x = ds2[i
