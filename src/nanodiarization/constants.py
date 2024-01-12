import os

CACHE_DIR = os.path.expanduser(os.getenv("CACHE_DIR", "~/.cache/"))
RUN_DUR = os.path.expanduser(os.getenv("CACHE_DIR", "~/runs/"))