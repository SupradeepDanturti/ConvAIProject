"""
Create mixtures into a json file which contains overlapped utterances.
"""

import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import torchaudio
import torch


def _read_metadata(file_path, configs):
    """
    Reads metadata from an audio file and selects a random channel if the file has multiple channels.
    Parameters:
    - file_path (str): The path to the audio file from which to read metadata.
    - configs (dict): Configuration options, which must include 'samplerate' specifying the expected sample rate.

    Returns:
    - tuple:
        - meta (torchaudio.backend.common.AudioMetaData): Metadata of the audio file, including sample rate,
          number of channels, etc.
        - channel (int): The index of the selected channel. If the audio file is mono, returns 0.
    """
    meta = torchaudio.info(file_path)
    if meta.num_channels > 1:
        channel = np.random.randint(0, meta.num_channels - 1)
    else:
        channel = 0
    assert (
            meta.sample_rate == configs["samplerate"]
    ), "file samplerate is different from the one specified"

    return meta, channel


def generate_silence(duration, sample_rate):
    """
    Generate a tensor representing silence of the given duration.

    Parameters:
    - duration: Duration of silence in seconds.
    - sample_rate: Sampling rate of the audio.

    Returns:
    - A PyTorch tensor representing the silent audio.
    """
    num_samples = int(duration * sample_rate)
    silence = torch.zeros(num_samples)
    return silence


def create_metadata(output_filename, n_sessions, configs, utterances_dict, words_dict):

    """
    Generates metadata for a dataset containing multiple audio sessions with varying numbers of speakers.

    Parameters:
    - output_filename (str): The base name of the output file where the session metadata will be saved.
    - n_sessions (int): The number of audio sessions to generate metadata for.
    - configs (dict): Configuration parameters which must include 'max_length', 'n_speakers', and 'librispeech_root'.
                      'max_length' is the maximum duration of each session, 'n_speakers' is the maximum number
                      of speakers per session, and 'librispeech_root' is the directory containing audio files.
    - utterances_dict (dict): A dictionary where each key is a speaker ID and the value is a list of utterance
                            file paths.
    - words_dict (dict): A dictionary where each key is the stem of an utterance file, and the value is a list of words
                        spoken in the utterance.
    Returns:
        None. This function generates a JSON file named `{output_filename}.json` which contains metadata for each
    session.
    """

    dataset_metadata = {}
    for n_sess in tqdm(range(n_sessions), desc="Generating sessions"):

        """Creating 0 Speaker Utterance"""
        session_name = f"session_{n_sess}_spk_0"
        dataset_metadata[session_name] = {
            "0": [{  # Using "0" to indicate no speaker
                "start": 0,
                "stop": configs["max_length"],
                "words": [],  # No words since it's silence
                "file": "generate_silence"  # Indicate that silence should be generated
            }]
        }

        """Creating 1 - n-Speaker Utterances as specified in configs"""
        for num_speakers in range(1, configs["n_speakers"] + 1):
            session_name = f"session_{n_sess}_spk_{num_speakers}"
            c_speakers = np.random.choice(list(utterances_dict.keys()), num_speakers, replace=False)

            activity = {spk: [] for spk in c_speakers}
            tot_length = configs["max_length"]
            segment_length = tot_length / num_speakers

            for i, spk_id in enumerate(c_speakers):
                spk_utts = utterances_dict[spk_id]
                np.random.shuffle(spk_utts)
                utt_idx = 0
                start_time = 0  # Each speaker starts at the beginning

                while start_time < tot_length and utt_idx < len(spk_utts):
                    utt = spk_utts[utt_idx]
                    meta, channel = _read_metadata(utt, configs)
                    utt_length = meta.num_frames / meta.sample_rate

                    if start_time + utt_length > start_time + segment_length:
                        # Ensure not to exceed each speaker's segment
                        break

                    activity[spk_id].append({
                        "start": start_time,
                        "stop": start_time + utt_length,
                        "words": words_dict[Path(utt).stem],
                        "file": str(Path(utt).relative_to(configs["librispeech_root"]))
                    })

                    start_time += utt_length  # Move to next utterance start time
                    utt_idx += 1

            dataset_metadata[session_name] = activity

    with open(output_filename + ".json", "w") as f:
        json.dump(dataset_metadata, f, indent=4)
