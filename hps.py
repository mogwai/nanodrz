"""
Little script to do some HPS
"""

from nanodrz.train import main, Config
from nanodrz.config import diffstr, DataConfig

default = Config()
default.model.dmodel = 1536
default.model.layers = 12
default.train.total_steps = 20_000
default.train.continue_from_checkpoint = False
default.train.max_lr = 1e-4

# Enable this is you can!
default.train.flash.enable_flash = True
default.train.batch_size = 30

# turn this into the format above
data = DataConfig()
data.silence_max = 2
data.interrupt_max = 1
data.max_secs = 60
data.num_speakers = 5
data.synth_datasets = ["librilight_medium"]
default.data = data

variations = {
    "train."
    # Scramble Speakers
}

for k, v in variations.values():
    # TODO
    pass
