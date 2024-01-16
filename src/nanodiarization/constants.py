import os
from os.path import expanduser as eu
import logging

def path_from_env(env: str, default_value: str):
    value = eu(os.getenv(env, default_value))
    os.makedirs(value, exist_ok=True)
    return value

CACHE_DIR = path_from_env("CACHE_DIR", "~/.cache/nanodrz/")
RUN_DIR = path_from_env("RUN_DIR", "~/runs/nanodrz/")
LOG_LEVEL = os.getenv("NANODRZ_LOG_LEVEL", logging.DEBUG)
