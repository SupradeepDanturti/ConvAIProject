import os
import torch
import torchaudio
from speechbrain.inference.VAD import VAD

# Assuming you have a structure of directories and files for your dataset
# Define your dataset directory and VAD model
dataset_dir = "../data/LibriSpeech/train-clean-100/"  # Update this path to your dataset directory
vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="./pretrained_models/vad-crdnn-libriparty/")


def process_audio_files_with_vad(dataset_directory):
    for subdir, dirs, files in os.walk(dataset_directory):
        for file in files:
            if file.endswith(".flac"):
                input_audio_path = os.path.join(subdir, file)
                output_text_path = "VAD_" + os.path.splitext(input_audio_path)[0] + "_.txt"

                boundaries = vad.get_speech_segments(input_audio_path)

                vad.save_boundaries(boundaries, save_path=output_text_path)

                merge_speech_segments(input_audio_path, output_text_path,
                                      input_audio_path)


def merge_speech_segments(input_audio_path, segments_text_path, output_audio_path):
    speech_segments = []
    with open(segments_text_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[-1] == 'SPEECH':
                speech_segments.append({'start': float(parts[1]), 'end': float(parts[2])})

    waveform, original_sr = torchaudio.load(input_audio_path)
    speech_segments_list = []

    for segment in speech_segments:
        start_sample = int(segment['start'] * original_sr)
        end_sample = int(segment['end'] * original_sr)
        speech_segment = waveform[:, start_sample:end_sample]
        speech_segment = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=16000)(speech_segment)
        speech_segments_list.append(speech_segment)

    if speech_segments_list:
        merged_segments = torch.cat(speech_segments_list, dim=1)
    else:
        merged_segments = torch.empty((waveform.shape[0], 0), dtype=torch.float)

    torchaudio.save(output_audio_path, merged_segments, 16000)


# Run the process on your dataset directory
process_audio_files_with_vad(dataset_dir)
