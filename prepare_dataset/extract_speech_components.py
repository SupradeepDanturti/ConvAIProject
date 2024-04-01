import json
import torch
import torchaudio
from speechbrain.inference.VAD import VAD
import torchaudio
print(torchaudio.get_audio_backend())


# Step 1: Extract Speech Segments Using VAD
def extract_speech_segments(input_audio_path, model_path, output_text_path):
    vad = VAD.from_hparams(source=model_path, savedir="./pretrained_models/vad-crdnn-libriparty")
    boundaries = vad.get_speech_segments(input_audio_path)
    vad.save_boundaries(boundaries, save_path=output_text_path)


# Step 2: Save Detected Speech Segments to JSON File
def save_speech_segments_to_json(input_text_path, output_json_path):
    speech_segments = []
    with open(input_text_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[-1] == 'SPEECH':
                speech_segments.append({'start': float(parts[1]), 'end': float(parts[2])})
    with open(output_json_path, 'w') as f:
        json.dump(speech_segments, f)


# Step 3: Load and Merge Speech Segments from JSON and Create a New WAV File
def merge_speech_segments(input_audio_path, segments_json_path, output_audio_path):
    with open(segments_json_path, 'r') as f:
        speech_segments = json.load(f)

    # Load the original audio file
    waveform, original_sr = torchaudio.load(input_audio_path)

    # Placeholder for merged speech segments
    merged_segments = torch.FloatTensor().empty()

    for segment in speech_segments:
        start_sample = int(segment['start'] * original_sr)
        end_sample = int(segment['end'] * original_sr)
        # Extract and resample the segment
        speech_segment = waveform[:, start_sample:end_sample]
        speech_segment = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=16000)(speech_segment)
        merged_segments = torch.cat((merged_segments, speech_segment), dim=1)

    # Save the merged segments into a new WAV file
    torchaudio.save(output_audio_path, merged_segments, 16000)


# Example usage
input_audio_path = "../data/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"
model_path = "./vad-crdnn-libriparty"
output_text_path = "VAD_file.txt"
output_json_path = "speech_segments.json"
output_audio_path = "merged_speech.wav"

# Execute the steps
extract_speech_segments(input_audio_path, model_path, output_text_path)
save_speech_segments_to_json(output_text_path, output_json_path)
merge_speech_segments(input_audio_path, output_json_path, output_audio_path)
