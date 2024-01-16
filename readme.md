# Nano Diarization - WIP!

Trying as much as possible to keep it simple in this repo

Partially inspired by [Pix2Seq](https://ai.googleblog.com/2022/04/pix2seq-new-language-interface-for.html)


# Core Questions

- Can we get transformers doing diarisation
- Can we create synthetic combinations of tts datasets to experiment with diarisation difficulty.
    - Will this help generalise to different domains.
- What happens as we increase the difficult of the data with interrupts, noise, number of speakers, loudness of speakers.

# Instructions

### Install

```sh
pip install -e .
```

### Train

```sh
train configs/simple_no_ints.yaml
```

### Inference

```
TODO
```

# Experiments

https://wandb.ai/harrycblum/nano-diarization?workspace=user-harrycblum

# Experiment Notes

https://fluxions.notion.site/nanodrz-Experiment-Log-acea3d5f436949b68e1f5a520c8cfdbc


# Credits

[James Parsloe](https://github.com/jamesparsloe) for project structure and lots of the utils found here.
[Herve Bredin](https://github.com/hbredin) for his amazing research on pyannote.audio
[Andrej Karpathy - NanoGPT](https://github.com/karpathy/nanoGPT/) For lots of great explanation on transformers 
