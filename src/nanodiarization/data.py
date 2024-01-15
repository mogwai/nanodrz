"""
In this file there are various strategies to collate utterances from single speakers
to create diarisation targets.
"""
import torch
import glob
import torchaudio
import os
from os.path import expanduser
import numpy as np 
import random
import itertools

from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import IterableDataset


class Speaker:
    # Audio samples
    name: str
    # Change this to a list of files
    samples: list[str]
    
class GeneratorIterableDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return iter(self.generator)
    
def collate_fn(batch):
    audios = [b[0] for b in batch]
    audio_lengths = [a.shape[-1] for a in audios]
    labels = [b[1][0] for b in batch]
    label_lengths = [l.shape[-1] for l in labels]
    audios = pad_sequence([a.permute(1,0) for a in audios], batch_first=True).permute(0,2,1)
    labels = pad_sequence(labels, batch_first=True)
    return {
        "audio": audios, 
        "labels": labels, 
        "audio_lengths": torch.tensor(audio_lengths), 
        "label_lengths": torch.tensor(label_lengths),
    }

def artificial_drz_generator(speakers:list[Speaker], model:torch.nn.Module, max_secs=30, sr=16000):   
    while True:
        audio, label = artificial_diarisation_sample(speakers, max_secs=max_secs, sr=sr)
        audio = model.dac.preprocess(audio, model.dac.sample_rate) 
        label = "\n".join([",".join([str(x) for x in l]) for l in label])
        label = model.text_tokenizer.encode(
                label,  
                return_tensors="pt",
                padding="longest",
        )
        yield audio, label

def artificial_diarisation_sample(speakers:list[Speaker], max_secs=30, interrupt_sec_mean=.2, interrupt_var=.1, num_speakers=4, sr=16000):
    audio = torch.zeros(1, 0)
    names, labels = [], []
    cur_speakers = random.sample(speakers, k=num_speakers)

    last_speaker = None
    # While we're still less than the target secs
    while audio.shape[-1]//sr < max_secs:
        # Pick a random speaker
        speaker = random.choice(cur_speakers)
        if speaker.name == last_speaker:
            continue
        last_speaker = speaker.name
        
        # Pick a random sample
        random_sample_file = random.choice(speaker.samples)
        random_sample, ssr = torchaudio.load(random_sample_file)
        random_sample = random_sample.sum(dim=0)[None]

        # Resample to 44100
        random_sample = torchaudio.transforms.Resample(ssr, sr)(random_sample)
        
        # Beginning
        if audio.shape[-1] == 0:
            audio = random_sample
            start_label = 0 
        else:
            # Choose interrupt section with mean interrupt_sec_mean and var interrupt_var
            interrupt_dur = np.random.normal(interrupt_sec_mean, interrupt_var)
            start_label = audio.shape[-1]/sr - interrupt_dur 

            audio = torch.cat((audio, torch.zeros(1, random_sample.shape[-1]-int(interrupt_dur*sr))), dim=-1)
            audio[:, -random_sample.shape[-1]:] = random_sample
        
        if speaker.name not in names:
            i =  len(names)
            names.append(speaker.name)
        else:
            i = names.index(speaker.name)
        
        name_label = chr(ord('A') + i)
    
        labels.append((start_label, random_sample.shape[-1]/sr, name_label))
    
    return audio, labels

def find_files_in_subfolders(folder):
    files = []
    for root, dirs, filenames in os.walk(folder):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def gather_speakers_from_folder(folder:str, retrieve_speaker:callable, exts:list[str] = ["wav", "opus", "mp3"]):
    """
    Retrieves all the audio files from a specific directory recursively. 
    """
    
    folder = expanduser(folder)
    wav_files = itertools.chain(*[glob.glob(folder + f'/**/*.{ext}', recursive=True) for ext in exts])
    speakers = []


    for file in wav_files:
        # Extract the speaker name from the file path
        speaker_name = retrieve_speaker(file)
        
        # Check if the speaker object already exists
        speaker_exists = False
        for speaker in speakers:
            if speaker.name == speaker_name:
                speaker_exists = True
                speaker.samples.append(file)
                break
        
        # If the speaker object doesn't exist, create a new one
        if not speaker_exists:  
            new_speaker = Speaker()
            new_speaker.name = speaker_name
            new_speaker.samples = [file]
            speakers.append(new_speaker)

    return speakers 



if __name__ == "__main__":
    speakers = gather_speakers_from_folder("/home/harry/storj/data/LibriTTS/test-clean/", lambda x: x.split("/")[-3])
    