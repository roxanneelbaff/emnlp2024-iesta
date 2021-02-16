


import pathlib
import shutil
import re
from pathlib import Path


def create_folder(folder_path):
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)