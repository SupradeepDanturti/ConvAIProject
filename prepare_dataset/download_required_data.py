"""
This script will download datasets for SpeakerCounter and extracts the required data into the specified folder.
[Librispeech Clean-100,dev-clean,test-clean,RIRS-noises]

usage: python download_required_data.py --output_folder <destination_folder_path>
"""

import argparse
import os
import tarfile

from speechbrain.utils.data_utils import download_file


def extract_tar_gz(file_path, output_path):
    """Extracts a .tar.gz file to the specified output path."""
    if file_path.endswith(".tar.gz"):
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=output_path)
        os.remove(file_path)  # remove the .tar.gz file after extraction


LIBRISPEECH_URLS = [
    "http://www.openslr.org/resources/12/test-clean.tar.gz",
    "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
]

OPENRIR_URL = "http://www.openslr.org/resources/28/rirs_noises.zip"

parser = argparse.ArgumentParser(
    "Python script to download required recipe data"
)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--stage", type=int, default=0)
args = parser.parse_args()

# output folder will be:
# LibriSpeech/
# rirs_noises/

os.makedirs(args.output_folder, exist_ok=True)

if args.stage <= 0:
    print("Stage 0: Downloading LibriSpeech")
    for url in LIBRISPEECH_URLS:
        name = url.split("/")[-1]
        download_file(url, os.path.join(args.output_folder, name), unpack=False)
        extract_tar_gz(os.path.join(args.output_folder, name), args.output_folder)

if args.stage <= 1:
    print("Stage 1: Downloading RIRs and Noises")
    name = OPENRIR_URL.split("/")[-1]
    download_file(
        OPENRIR_URL, os.path.join(args.output_folder, name), unpack=True
    )
