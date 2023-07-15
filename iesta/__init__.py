"""iesta package that uses existing libraries"""

# __version__ = "0.3.0"

from pkg_resources import resource_filename
import logging
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv

filepath = resource_filename(__name__, ".env")

logs_folder = os.path.join(os.path.dirname(__file__), "iesta-logs")
Path(logs_folder).mkdir(parents=True, exist_ok=True)

# Create a custom logger
global logger
logger = logging.getLogger("iesta")

# Create handlers
console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("%(name)-12s %(levelname)-8s %(message)s")
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)


f_handler = logging.FileHandler("{}/logs.txt".format(logs_folder))
# f_handler.setLevel(logging.INFO)
file_format = logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%d.%m.%Y %H:%M",
)
f_handler.setFormatter(file_format)
logger.addHandler(f_handler)

logger.setLevel(logging.DEBUG)
found = load_dotenv(find_dotenv())
print(f"dotenv was {found}")
