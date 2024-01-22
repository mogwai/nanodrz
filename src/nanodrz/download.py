import requests
import tarfile

from nanodrz.constants import CACHE_DIR

from os import path, makedirs
from tqdm import tqdm
import os
import subprocess


def dl_scp_file(link:str):
    """
    Downloads a file using ssh scp via subprocess

    hostname:runs/nanodrz/nanodrz/1705840799/0013500.pt
    """
    if ":" not in link:
        raise "Invalid scp path"
    
    file_path = path.join(CACHE_DIR, link.split(":"))
    
    if not path.exists(file_path):
        subprocess.run(["scp", link, file_path])
    
    return file_path

def dl_http_file(link:str):
    file_path = path.join(CACHE_DIR, path.basename(link))

    if not path.exists(file_path):
        response = requests.get(link, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1 MB
        progress_bar = tqdm(
            total=total_size,
            desc=link,
            unit="B",
            unit_scale=True,
            leave=False,
        )
        
        with open(file_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
    
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
        tar = tarfile.open(file_path, mode="r:gz")
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
