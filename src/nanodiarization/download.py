import requests
import tarfile

from nanodiarization.constants import CACHE_DIR
from os import path
from tqdm import tqdm


def dl_libritts_clean():
    """
    LibriTTS https://openslr.elda.org/60/
    """

    link = "https://openslr.elda.org/resources/60/test-clean.tar.gz"
    output_path = path.join(CACHE_DIR, "LibriTTS")
    if not path.exists(output_path):
        response = requests.get(link, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1 MB
        progress_bar = tqdm(
            total=total_size,
            desc=path.basename(link),
            unit="B",
            unit_scale=True,
            leave=False,
        )

        # Save the file to the CACHE_DIR
        file_path = path.join(CACHE_DIR, "test-clean.tar.gz")
        if not path.exists(file_path):
            with open(file_path, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
        progress_bar.close()

        tar = tarfile.open(file_path, mode="r:gz")
        tar.extractall(CACHE_DIR)
        tar.close()

    return output_path


if __name__ == "__main__":
    print(dl_libritts_clean())
