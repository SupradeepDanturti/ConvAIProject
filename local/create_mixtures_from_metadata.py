import os
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
import json
from speechbrain.augment.time_domain import AddNoise, AddReverb


def create_mixture(session_n, output_dir, params, metadata):
    os.makedirs(os.path.join(output_dir, session_n), exist_ok=True)
    session_meta = {}

    # Initialize mixture length based on max_length parameter
    max_length_sec = params['max_length']
    tot_length_samples = int(np.ceil(max_length_sec * params["samplerate"]))
    mixture = torch.zeros(tot_length_samples)

    # Placeholder for tracking utterance start and stop times for overlap management
    active_utterances = []

    for speaker_id, utterances in metadata[session_n].items():
        # Initialize a list for each speaker ID if not already present
        if speaker_id not in session_meta:
            session_meta[speaker_id] = []

        for utterance in utterances:
            audio_path = os.path.join(params["librispeech_root"], utterance["file"])
            audio, _ = torchaudio.load(audio_path)

            # Mono conversion and channel selection if needed
            if audio.shape[0] > 1:
                audio = audio[utterance["channel"]]

            # print("Shape before effects:", audio.shape)
            # clean = audio.unsqueeze(0)
            # print("Shape after unsqueeze(0) :", clean.shape)
            #
            # if torch.rand(1) < 0.5:
            #     noisy = noisifier(clean, torch.ones(clean.size(0)))
            #     print("Shape After adding Noisy effects:", noisy.shape)
            # else:
            #     noisy = clean
            #     print("Did not add noise due to probability")
            #
            # if noisy.dim() == 2:
            #     noisy = noisy.unsqueeze(-1)
            #     print("Shape After noisy.unsqueeze(-1) as noisy.dim() == 2 :", noisy.shape)
            # if torch.rand(1) < 0.5:
            #     reverbed = reverber(noisy)
            #     print("Shape After adding reberb effects:", reverbed.shape)
            # else:
            #     reverbed = noisy
            #     print("Did not add reberb  due to probability")
            #
            # audio = reverbed.squeeze(0).transpose(0, 1)
            # print("Shape After reverbed.squeeze(0).transpose(0, 1):", audio.shape)

            start_sample = int(utterance["start"] * params["samplerate"])
            stop_sample = start_sample + audio.shape[1]

            # Ensure mixture can accommodate current utterance
            if stop_sample > mixture.shape[0]:
                additional_length = stop_sample - mixture.shape[0]
                mixture = torch.cat((mixture, torch.zeros(additional_length)), 0)

            # Make sure audio is 1D before adding it to the mixture
            if audio.dim() > 1:
                audio = audio.squeeze()  # Squeeze audio to ensure it's 1D

            # Add current utterance to the mixture
            mixture[start_sample:stop_sample] += audio

            # Update active utterances for overlap management
            active_utterances.append((start_sample, stop_sample))

            # Append utterance details to the corresponding speaker ID in session_meta
            session_meta[speaker_id].append({
                "file": utterance["file"],
                "start": utterance["start"],
                "stop": utterance["stop"],
                "words": utterance["words"]
            })

    # Clamp mixture to avoid clipping
    mixture = torch.clamp(mixture, min=-1.0, max=1.0)

    # Save the mixture
    mixture_path = os.path.join(output_dir, session_n, f"{session_n}_mixture.wav")
    torchaudio.save(mixture_path, mixture.unsqueeze(0), params["samplerate"])

    # Save session metadata to JSON
    metadata_path = os.path.join(output_dir, session_n, f"{session_n}_metadata.json")
    with open(metadata_path, "w") as jsonfile:
        json.dump(session_meta, jsonfile, indent=4)
