import torch
from nanodrz.model import DiarizeGPT, Config
from nanodrz.data import libritts_test, artificial_diarisation_sample

speakers = libritts_test()

ckpt = torch.load("/home/harry/0000100.pt")
config = Config(**ckpt["config"])
model:DiarizeGPT = DiarizeGPT.from_pretrained(ckpt)
audio, labels = artificial_diarisation_sample(speakers, **config.data.model_dump())
out = model.generate(audio)
x = 1