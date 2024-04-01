"""
Custom LibriParty creation script with user specified parameters.

Author
------
Samuele Cornell, 2020
"""

import os
import sys
import json
import random
import math
import numpy as np
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.data_utils import get_all_files
from local.create_mixtures_metadata import create_metadata
from local.create_mixtures_from_metadata import create_mixture
from pathlib import Path
import torch
import torchaudio
from speechbrain.augment.preparation import write_csv
from speechbrain.augment.time_domain import AddNoise, AddReverb
from tqdm import tqdm

# Load hyperparameters file with command-line overrides
params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = load_hyperpyyaml(fin, overrides)

# setting seeds for reproducible code.
np.random.seed(params["seed"])
random.seed(params["seed"])


# we parse the yaml, and create mixtures for every train, dev and eval split.


def split_list(array, split_factors):
    assert round(sum(split_factors), 6) == 1, "split_factors should sum to one"
    np.random.shuffle(array)
    pivots = [int(len(array) * x) for x in split_factors]
    out = []
    indx = 0
    for i in pivots:
        out.append(array[indx: i + indx])
        indx = i
    return out


def parse_libri_folder(libri_folders):
    # parsing librispeech
    utterances = []
    txt_files = []
    for libri_dir in libri_folders:
        utterances.extend(get_all_files(libri_dir, match_and=[".flac"]))
        txt_files.extend(get_all_files(libri_dir, match_and=["trans.txt"]))

    words_dict = {}
    for trans in txt_files:
        with open(trans, "r") as f:
            for line in f:
                splitted = line.split(" ")
                utt_id = splitted[0]
                words = " ".join(splitted[1:])
                words_dict[utt_id] = words.strip("\n")

    speakers = {}
    for u in utterances:
        spk_id = Path(u).parent.parent.stem
        speakers[spk_id] = speakers.get(spk_id, []) + [u]

    return speakers, words_dict

os.makedirs(os.path.join(params["out_folder"], "metadata"), exist_ok=True)

# we generate metadata for each split
for indx, split in enumerate(["train", "dev", "eval"]):
    print("Generating metadata for {} set".format(split))
    # we parse librispeech utterances for current split
    c_libri_folder = params["librispeech_folders"][split]
    c_utterances, c_words = parse_libri_folder(c_libri_folder)

    create_metadata(
        os.path.join(params["out_folder"], "metadata", split),
        params["n_sessions"][split],
        params,
        c_utterances,
        c_words,
    )

# from metadata we generate the actual mixtures

for indx, split in enumerate(["train", "dev", "eval"]):
    print(f"Creating {split} set")
    # load metadata
    metadata_path = os.path.join(params["out_folder"], "metadata", f"{split}.json")
    with open(metadata_path) as f:
        c_meta = json.load(f)

    c_folder = os.path.join(params["out_folder"], split)
    os.makedirs(c_folder, exist_ok=True)

    # Here we loop through sessions with progress tracking
    for sess in tqdm(list(c_meta.keys()), desc=f"Creating {split} set"):
        # print(c_meta[sess])
        create_mixture(sess, c_folder, params, c_meta)

print("Creating segments....")


def create_segments(x="train"):
    segment_length = 2  # seconds
    sample_rate = 16000
    file_list = get_all_files((os.path.join(params["out_folder"], f"{x}")), match_and=[".wav"])

    for file in tqdm(file_list, desc=f"Processing {x} segments"):
        wav_path = file.replace("\\", "/")

        waveform, _ = torchaudio.load(wav_path)
        num_samples_per_segment = sample_rate * segment_length
        total_segments = math.ceil(waveform.size(1) / num_samples_per_segment)

        for segment_id in range(total_segments):
            start_sample = segment_id * num_samples_per_segment
            end_sample = start_sample + num_samples_per_segment

            if end_sample <= waveform.size(1):
                segment_waveform = waveform[:, start_sample:end_sample]

                segment_file_name = f"{os.path.splitext(wav_path)[0]}_{segment_id:03d}_segment.wav"
                torchaudio.save(segment_file_name, segment_waveform, sample_rate)


create_segments("train")
create_segments("dev")
create_segments("eval")
print("Adding Noise....")

rir_audios = get_all_files(os.path.join(params["rirs_noises_root"], "simulated_rirs"), match_and=['.wav'])
rir_audios.extend(
    get_all_files(os.path.join(params["rirs_noises_root"], "real_rirs_isotropic_noises"), match_and=['.wav']))
noise_audios = get_all_files(os.path.join(params["rirs_noises_root"], "pointsource_noises"), match_and=['.wav'])

write_csv(rir_audios, os.path.join(params["out_folder"], "simulated_rirs.csv"))
write_csv(noise_audios, os.path.join(params["out_folder"], "noises.csv"))

noisifier = AddNoise(os.path.join(params["out_folder"], "noises.csv"), snr_low=5, snr_high=20)
reverber = AddReverb(os.path.join(params["out_folder"], "simulated_rirs.csv"), reverb_sample_rate=16000,
                     clean_sample_rate=16000)

batch_size = 10
probability_noise = 0.5
probability_reverb = 0.5


def load_and_addnoise(x="train"):
    # with open(f'{x}_data.json', 'r') as f:
    #     data = json.load(f)
    json_data = get_all_files((os.path.join(params["out_folder"], f"{x}")), match_and=[".wav"])
    for data in tqdm(json_data, desc=f'Loading {x}'):
        audio_path = data.replace("\\", "/")
        signal = read_audio(audio_path)
        clean = signal.unsqueeze(0)

        if random.random() < probability_noise:
            noisy = noisifier(clean, torch.ones(clean.size(0)))
        else:
            noisy = clean

        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(-1)

        if random.random() < probability_reverb:
            reverbed = reverber(noisy)
        else:
            reverbed = noisy

        processed_audio = reverbed.squeeze(0).transpose(0, 1)
        output_path = audio_path

        torchaudio.save(output_path, processed_audio, 16000)


load_and_addnoise("train")
load_and_addnoise("dev")
load_and_addnoise("eval")
print("AddedNoise")

print("Created Custom Dataset")
