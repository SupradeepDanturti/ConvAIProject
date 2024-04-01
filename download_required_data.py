"""
Source datasets downloading script for LibriParty.

Author
------
Samuele Cornell, 2020
"""

import argparse
import os
import speechbrain
from speechbrain.utils.data_utils import download_file
from local.resample_folder import resample_folder

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
        download_file(url, os.path.join(args.output_folder, name), unpack=True)

if args.stage <= 1:
    print("Stage 1: Downloading RIRs and Noises")
    name = OPENRIR_URL.split("/")[-1]
    download_file(
        OPENRIR_URL, os.path.join(args.output_folder, name), unpack=True
    )
