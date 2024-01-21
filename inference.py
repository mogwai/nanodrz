import torch
from nanodrz.model import DiarizeGPT, Config
from nanodrz.data import libritts_test, artificial_diarisation_sample

speakers = libritts_test()

ckpt = torch.load("/home/harry/0004600.pt")
config = Config(**ckpt["config"])
model:DiarizeGPT = DiarizeGPT.from_pretrained(ckpt).cuda()
# Use the same parameters that the model was trained on to generate a sample
audio, labels = artificial_diarisation_sample(speakers, **config.data.model_dump())
print(model.generate(audio.cuda()))