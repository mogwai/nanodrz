import tarfile

from nanodrz.constants import CACHE_DIR

from os import path, makedirs
from tqdm import tqdm
import os
import subprocess
from urllib import request


def dl_scp_file(link: str):
    """
    Downloads a file using rsync and scp subprocess

    hostname:runs/nanodrz/nanodrz/1705840799/0013500.pt

    Use rsync --partial --progress --human-readable -e ssh to download the file.
    """
    if ":" not in link:
        raise "Invalid scp path"

    file_path = path.join(CACHE_DIR, link.split(":")[-1])

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Use rsync --partial --progress --human-readable -e ssh
    cmd = [
        "rsync",
        "--partial",
        "--progress",
        "--human-readable",
        "-e",
        "ssh",
        link,
        file_path,
    ]
    print(" ".join(cmd))
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        text=True,
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    return file_path


def dl_http_file(link: str):
    file_path = path.join(CACHE_DIR, path.basename(link))

    if not os.path.exists(file_path):
        with request.urlopen(link) as response, open(file_path, "wb") as file:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 1024 * 5 # 5MB
            progress_bar = tqdm(
                total=total_size,
                desc=link,
                unit="B",
                unit_scale=True,
                leave=False,
            )

            while True:
                data = response.read(block_size)
                if not data:
                    break
                progress_bar.update(len(data))
                file.write(data)

            progress_bar.close()

    return file_path


def dl_libritts_test():
    """
    LibriTTS https://openslr.elda.org/60/
    """

    link = "https://openslr.elda.org/resources/60/test-clean.tar.gz"

    file_path = dl_http_file(link)
    extract_folder = path.join(CACHE_DIR, path.basename(file_path).split(".")[0])

    if not path.exists(extract_folder):
        makedirs(extract_folder, exist_ok=True)
        mode = "r"
        if "gz" in file_path:
            mode += ":gz"

        tar = tarfile.open(file_path, mode=mode)
        tar.extractall(extract_folder)
        tar.close()

    return path.join(extract_folder, "LibriTTS")


def dl_libritts_dev():
    """
    LibriTTS https://openslr.elda.org/60/
    """

    link = "https://openslr.elda.org/resources/60/dev-clean.tar.gz"

    file_path = dl_http_file(link)
    extract_folder = path.join(CACHE_DIR, path.basename(file_path).split(".")[0])

    if not path.exists(extract_folder):
        makedirs(extract_folder, exist_ok=True)
        tar = tarfile.open(file_path, mode="r:gz")
        tar.extractall(extract_folder)
        tar.close()

    return path.join(extract_folder, "LibriTTS")


def dl_libri_light_small():
    """
    https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md

    https://dl.fbaipublicfiles.com/librilight/data/small.tar
    """

    link = "https://dl.fbaipublicfiles.com/librilight/data/small.tar"

    file_path = dl_http_file(link)
    extract_folder = path.join(CACHE_DIR, path.basename(file_path).split(".")[0])

    if not path.exists(extract_folder):
        makedirs(extract_folder, exist_ok=True)
        mode = "r"
        if "gz" in file_path:
            mode += ":gz"

        tar = tarfile.open(file_path, mode=mode)
        tar.extractall(extract_folder)
        tar.close()

    return path.join(extract_folder, "small")


def dl_libri_light_medium():
    """
    https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md

    https://dl.fbaipublicfiles.com/librilight/data/medium.tar
    """

    link = "https://dl.fbaipublicfiles.com/librilight/data/medium.tar"

    file_path = dl_http_file(link)
    extract_folder = path.join(CACHE_DIR, path.basename(file_path).split(".")[0])

    if not path.exists(extract_folder):
        makedirs(extract_folder, exist_ok=True)
        mode = "r"
        if "gz" in file_path:
            mode += ":gz"

        tar = tarfile.open(file_path, mode=mode)
        tar.extractall(extract_folder)
        tar.close()

    return path.join(extract_folder, "medium")


def dl_libri_light_large():
    """
    https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md

    https://dl.fbaipublicfiles.com/librilight/data/large.tar
    """

    link = "https://dl.fbaipublicfiles.com/librilight/data/large.tar"

    file_path = dl_http_file(link)
    extract_folder = path.join(CACHE_DIR, path.basename(file_path).split(".")[0])

    if not path.exists(extract_folder):
        makedirs(extract_folder, exist_ok=True)
        mode = "r"
        if "gz" in file_path:
            mode += ":gz"

        tar = tarfile.open(file_path, mode=mode)
        tar.extractall(extract_folder)
        tar.close()

    return path.join(extract_folder, "large")


if __name__ == "__main__":
    directory = dl_libri_light_small()
    files = os.listdir(directory)

    for file in files:
        print(file)
    pass
