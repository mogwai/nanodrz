from nanodiarization.data import gather_speakers_from_folder, artificial_diarisation_sample, libritts_clean
from nanodiarization.utils import play, visualise_annotation

speakers = libritts_clean()

while True:
    audio, labels = artificial_diarisation_sample(speakers, max_secs=30)
    # visualise_annotation(labels)
    # play(audio)