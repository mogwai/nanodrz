from glob import glob
from nanodrz.augmentations import denoise
import torchaudio
from os.path import basename, exists
from tqdm import tqdm

from denoiser import pretrained
import torchaudio
from nanodrz.utils import resample
import torch

from torch.nn.utils.rnn import pad_sequence

torch.cuda.set_device("cuda:1")

B = 10
files = glob("/home/harry/.cache/nanodrz/chunks/*.flac")
files.reverse()

# Group files into batches of size B
file_batches = [files[i : i + B] for i in range(0, len(files), B)]

denoiser = pretrained.dns64().cuda().eval()

with torch.inference_mode():
    for fb in tqdm(file_batches, desc="denoise", leave=False):
        batch = []

        for f in fb:
            fsplit = f.split("/")
            fsplit[-1] = fsplit[-1].replace(".flac", "_d.wav")
            fn = "/".join(fsplit)

            if not exists(fn):
                batch.append([torchaudio.load(f)[0], fn, f])

        if len(batch) < 1:
            continue

        # Get audio lengths
        lens = [a[0].shape[-1] for a in batch]

        # Pad sequence
        audio = pad_sequence(
            [b[0].permute(1, 0) for b in batch], batch_first=True
        ).permute(0, 2, 1)

        # audio = torch.nn.functional.pad(audio, (0, 112000 - audio.shape[-1]), value=0)
        out = denoiser(audio.cuda()).cpu()
        print(out.shape)
        for i in range(out.shape[0]):
            torchaudio.save(batch[i][1], out[i, :, :lens[i]], 16000)
