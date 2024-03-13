import os
import sys
import random
import torch
import torchaudio
from tqdm import tqdm
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import get_all_files
from speechbrain.augment.time_domain import AddNoise, AddReverb
from speechbrain.augment.preparation import write_csv
from speechbrain.dataio.dataio import read_audio


def add_noise_reverb(params):
    # Use paths from the loaded configuration
    train_audios = get_all_files(os.path.join(params['out_folder'], "train"), match_and=['.wav'])
    dev_audios = get_all_files(os.path.join(params['out_folder'], "dev"), match_and=['.wav'])
    eval_audios = get_all_files(os.path.join(params['out_folder'], "eval"), match_and=['.wav'])

    rir_audios = get_all_files(os.path.join(params['rirs_noises_root'], "simulated_rirs"), match_and=['.wav'])
    rir_audios.extend(get_all_files(os.path.join(params['rirs_noises_root'], "real_rirs_isotropic_noises"), match_and=['.wav']))

    noise_audios = get_all_files(os.path.join(params['rirs_noises_root'], "pointsource_noises"), match_and=['.wav'])

    # Assuming you have lists of RIRs and noises prepared similar to the train/dev/eval audios

    write_csv(rir_audios, os.path.join(params['out_folder'], "simulated_rirs.csv"))
    write_csv(noise_audios, os.path.join(params['out_folder'], "noises.csv"))

    noisifier = AddNoise(os.path.join(params['out_folder'], "noises.csv"), num_workers=8)
    reverb = AddReverb(os.path.join(params['out_folder'], "simulated_rirs.csv"), num_workers=8)

    batch_size = 10
    audio_list = [train_audios, dev_audios, eval_audios]

    # Define probabilities
    probability_noise = 0.5
    probability_reverb = 0.5

    for audios in audio_list:
        total_batches = len(audios) // batch_size + (1 if len(audios) % batch_size > 0 else 0)

        for i in tqdm(range(0, len(audios), batch_size), desc=f'Processing Batches', total=total_batches):
            batch_paths = audios[i:i + batch_size]

            processed_audios = []
            for audio_path in batch_paths:
                audio_path = audio_path.replace("\\", "/")
                signal = read_audio(audio_path)
                clean = signal.unsqueeze(0)

                if random.random() < probability_noise:
                    noisy = noisifier(clean, torch.ones(clean.size(0)))
                else:
                    noisy = clean

                if noisy.dim() == 2:
                    noisy = noisy.unsqueeze(-1)
                if random.random() < probability_reverb:
                    reverbed = reverb(noisy)
                else:
                    reverbed = noisy

                processed_audio = reverbed.squeeze(0).transpose(0, 1)
                processed_audios.append(processed_audio)

            for j, processed_audio in enumerate(processed_audios):
                output_path = batch_paths[j]
                torchaudio.save(output_path, processed_audio, 16000)

if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)
    add_noise_reverb(params)
    print("Added noise reverb")