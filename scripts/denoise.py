from glob import glob
from nanodrz.augmentations import denoise
import torchaudio
from tqdm import tqdm

from denoiser import pretrained
import torchaudio
import torch
torch.cuda.set_device("cuda:1")

files = glob("/home/harry/.cache/nanodrz/chunks/*")

denoiser = pretrained.dns64().cuda().eval()

for f in tqdm(files[:10], desc="denoise", leave=False):
    try:
        audio, sr = torchaudio.load(f)
        audio = denoise(denoiser, audio, sr)
        torchaudio.save(f, audio.cpu(), sr)
    except Exception as e:
        print(e, f)
