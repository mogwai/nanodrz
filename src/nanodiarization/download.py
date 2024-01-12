import requests
import tarfile

from nanodiarization.constants import CACHE_DIR
import os


def download_speech():
    link = "https://openslr.elda.org/resources/60/test-clean.tar.gz"

    # Download the file into memory
    response = requests.get(link)

    # Save the file to the CACHE_DIR
    file_path = os.path.join(CACHE_DIR, "test-clean.tar.gz")
    with open(file_path, "wb") as file:
		file.write(response.content)

    tar = tarfile.open(file_path, mode="r:gz")
    tar.extractall(CACHE_DIR)
    tar.close()
