"""
In this file there are various strategies to collate utterances from single speakers
to create diarisation targets.
"""
import torch
import glob
import torchaudio
from torch import Tensor
import numpy as np 
import random

class Speaker:
    # Audio samples
    name: str
    # Change this to a list of files
    samples: list[str]

def artificial_diarisation_sample(speakers:list[Speaker], max_secs=30, interrupt_sec_mean=.2, interrupt_var=.1, num_speakers=4):
    audio = torch.zeros(1, 0)
    names, labels = [], []
    sr = 44100

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
        random_sample = torchaudio.transforms.Resample(ssr, 44100)(random_sample)
        
        if audio.shape[-1] == 0:
            audio = torch.cat((audio, random_sample), dim=-1)
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

def gather_speakers_from_folder(folder:str, retrieve_speaker:callable, file_ext:str = ".wav"):
    wav_files = glob.glob(folder + f"**/*{file_ext}", recursive=True)
    # Gather all the files here. And create Speaker objects for each subfolder

    speakers = []

    for file in wav_files:
        # Extract the speaker name from the file path
        speaker_name = retrieve_speaker(file)
        speaker_name = file.split("/")[-3]
        
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
    