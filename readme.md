# Nano Diarization

Trying as much as possible to keep it simple in this repo

Partially inspired by [Pix2Seq](https://ai.googleblog.com/2022/04/pix2seq-new-language-interface-for.html)

# Instructions

### Install

```
pip install -e .

```

### Train

```
train configs/simple_no_ints.yaml
```

### Inference

```
TODO
```

# Experiment Ideas

### Labelling
- More efficient labelling A:2.3,4|6.5|2\nB:0.1,0.8\n
- Just label the speaker.
- Labelling boundaries instead of start and duration?

### Augmentation
- Add noise generated from audiogen
- Volume augmentation (Someone coming in closer to the mic as they speak)
- Pitch Augmentation
- Add Noise to signal
- 

### Core
- Fine tune pretrained decoder only byt5
- DAC 44.1khz, 24khz models


# Credits

(James Parsloe)[https://github.com/jamesparsloe] for project structure and lots of the utils found here.
(Herve Bredin)[https://github.com/hbredin] for his amazing research on pyannote.audio
(Andrej Karpathy - NanoGPT)[https://github.com/karpathy/nanoGPT/] For lots of great explaination on transformers 