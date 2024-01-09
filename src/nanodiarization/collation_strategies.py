"""
In this file there are various strategies to collate utterances from single speakers
to create diarisation targets.
"""
from torch import Tensor

# Basic interleave speech without interruptions and short pauses

class Speaker:
    # Audio samples
    samples: list[Tensor]

def no_pauses_speech(speakers:list[Speaker], max_secs=30):
    # pick a random sample from the speaker
    # build the list of start_time, duration, speaker label (A, B, C)
    # cut the audio so that t

    