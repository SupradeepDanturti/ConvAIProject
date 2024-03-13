"""
This file contains functions to create json metadata used to create
mixtures which simulate a multi-party conversation in a noisy scenario.

Author
------
Samuele Cornell, 2020
"""


import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import torchaudio


def _read_metadata(file_path, configs):
    meta = torchaudio.info(file_path)
    if meta.num_channels > 1:
        channel = np.random.randint(0, meta.num_channels - 1)
    else:
        channel = 0
    assert (
        meta.sample_rate == configs["samplerate"]
    ), "file samplerate is different from the one specified"

    return meta, channel


def create_metadata(output_filename, n_sessions, configs, utterances_dict, words_dict, rir_list):
    dataset_metadata = {}
    for n_sess in tqdm(range(n_sessions), desc="Generating sessions"):
        for num_speakers in range(1, configs["n_speakers"] + 1):
            session_name = f"session_{n_sess}_spk_{num_speakers}"
            c_speakers = np.random.choice(list(utterances_dict.keys()), num_speakers, replace=False)

            activity = {spk: [] for spk in c_speakers}
            tot_length = configs["max_length"]
            segment_length = tot_length / num_speakers  # Divide total length by number of speakers for equal distribution

            for i, spk_id in enumerate(c_speakers):
                spk_utts = utterances_dict[spk_id]
                np.random.shuffle(spk_utts)
                utt_idx = 0
                start_time = 0  # Each speaker starts at the beginning

                while start_time < tot_length and utt_idx < len(spk_utts):
                    utt = spk_utts[utt_idx]
                    meta, channel = _read_metadata(utt, configs)
                    c_rir = np.random.choice(rir_list, 1)[0]
                    meta_rir, rir_channel = _read_metadata(c_rir, configs)
                    utt_length = meta.num_frames / meta.sample_rate

                    if start_time + utt_length > start_time + segment_length:  # Ensure not to exceed each speaker's segment
                        break

                    lvl = np.clip(np.random.normal(configs["speech_lvl_mean"], configs["speech_lvl_var"]),
                                  configs["speech_lvl_min"], configs["speech_lvl_max"])
                    activity[spk_id].append({
                        "start": start_time,
                        "stop": start_time + utt_length,
                        "words": words_dict[Path(utt).stem],
                        "rir": str(Path(c_rir).relative_to(configs["rirs_noises_root"])),
                        "file": str(Path(utt).relative_to(configs["librispeech_root"])),
                        "lvl": lvl,
                        "channel": channel,
                        "rir_channel": rir_channel,
                    })

                    start_time += utt_length  # Move to next utterance start time
                    utt_idx += 1

            dataset_metadata[session_name] = activity

    with open(output_filename + ".json", "w") as f:
        json.dump(dataset_metadata, f, indent=4)