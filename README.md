# **Overlap Detector + Speaker Counter** 

This project implements and compares speaker counting techniques for challenging overlapping speech scenarios, including self-supervised approaches.

**Data Preparation**: Custom dataset creation with flexible parameters (number of speakers, segment lengths)
**Models**:
- X-Vector
- ECAPA-TDNN
- Self-supervised Wav2Vec 2.0 with MLP classifier
- Self-supervised Wav2Vec 2.0 with X-Vector
**Training**: Hyperparameters provided for model training
**Inference**: Simple-to-use interface scripts for running trained models on new audio

## **Prerequisites**
- Python (version 3.7 or above)
 - speechbrain toolkit (pip install speechbrain)

## **Installation**
```
!git clone https://github.com/SupradeepDanturti/ConvAIProject
%cd ConvAIProject
```
## **Dataset Preparation**
```
!python prepare_dataset/download_required_data.py --output_folder <destination_folder_path>
!python prepare_dataset/create_custom_dataset.py prepare_dataset/dataset.yaml
```
Edit dataset.yaml to customize the number of sessions, speakers, and segment lengths.
Sample of dataset.yaml:
```
n_sessions:
  train: 1000 # Creates 1000 sessions per class
  dev: 200 # Creates 200 sessions per class
  eval: 200 # Creates 200 sessions per class
n_speakers: 4 # max number of speakers. In this case the total classes will be 5 (0-4 speakers)
max_length: 120 # max length in seconds for each session/utterance.
```
## **Model Training**
To train the XVector model run the following command.
```
# Train X-Vector model:
!cd xvector
!python train_xvector_augmentation.py hparams_xvector_augmentation.yaml 

# Train other models (follow similar pattern):
!cd ecapa_tdnn
!python train_ecapa_tdnn.py hparams_ecapa_tdnn_augmentation.yaml 

!cd selfsupervised
!python selfsupervised_mlp.py hparams_selfsupervised_mlp.yaml

!cd selfsupervised
!python selfsupervised_xvector.py hparams_selfsupervised_xvector.yaml
```

## **Inference Interface**
To run interface for XVector & ECAPA-TDNN
```
""" This is used for Both XVector and ECAPA-TDNN """

from interface.SpeakerCounter import SpeakerCounter

wav_path = "interface/sample_audio1.wav"  # Path to your audio file
save_dir = "interface/sample_inference_run2/" # Where to save results
model_path = "interface/xvector" # /ecapa_tdnn  # Path to your trained model

# Create classifier object
audio_classifier = SpeakerCounter.from_hparams(source=model_path, savedir=save_dir)

# Run inference on the audio file
audio_classifier.classify_file(wav_path)
```
For Selfsupervised model
```

from interface.SpeakerCounterSelfsupervisedMLP import SpeakerCounter

wav_path = "interface/sample_audio1.wav" # Path to your audio file
save_dir = "interface/sample_inference_run" # Where to save results
model_path = "interface/selfsupervised_mlp" # Path to your trained model

# Create classifier object
audio_classifier = SpeakerCounter.from_hparams(source=model_path, savedir=save_dir)

# Run inference on the audio file
audio_classifier.classify_file(wav_path)
```

## **Project Structure**
- prepare_dataset/ : Scripts for dataset downloading and creation
- xvector/ : X-Vector model implementation and training
- ecapa_tdnn/ : ECAPA-TDNN implementation and training
- selfsupervised/ : Self-supervised models implementation and training
- interface/ : Inference scripts
