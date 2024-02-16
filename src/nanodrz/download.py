import tarfile

from nanodrz.constants import CACHE_DIR

from os import path, makedirs
from tqdm import tqdm
import os
import subprocess
from urllib import request
import zipfile
import shutil

def dl(link: str):
    if ":" in link:
        return dl_scp_file(link)
    else:
        return dl_http_file(link)

    
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

    current_size = 0
    expected_size = int(request.urlopen(link).headers.get("content-length", 0))
    
    if os.path.exists(file_path):
        current_size = os.path.getsize(file_path)

        if current_size == expected_size:
            return file_path

    headers = {"Range": f"bytes={current_size}-"}
    req = request.Request(link, headers=headers)

    with request.urlopen(req) as response, open(file_path, "ab") as file:
        block_size = 1024 * 1024
        progress_bar = tqdm(
            total=expected_size - current_size,  # Adjust total size
            initial=current_size,
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

    return path.join(extract_folder)


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

    return path.join(extract_folder)


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

    return path.join(extract_folder)


def dl_voxconverse_dev():
    link = "http://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_dev_wav.zip"

    file_path = dl_http_file(link)

    extract_folder = path.join(CACHE_DIR, path.basename(file_path).split(".")[0])

    if not path.exists(extract_folder):
        makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

        extract_folder = path.join(extract_folder, "audio")

        files = os.listdir(extract_folder)

        for i in files:
            i = i.replace(".wav", ".rttm")
            link = f"https://raw.githubusercontent.com/joonson/voxconverse/master/dev/{i}"
            file = dl_http_file(link)
            # Check if file is empty before copying
            with open(file) as f:
                rttm = f.read()

            if rttm.strip() != "":
                shutil.copy(file, extract_folder)
                os.remove(file)

    return path.join(extract_folder)


if __name__ == "__main__":
    directory = dl_libri_light_small()
    files = os.listdir(directory)
    print(files)