# **ConvAIProject Project 7: Overlap Detector + Speaker Counter** 
- The objective of this project  is to build a speaker counter for meeting recordings using speech technology. Meetings often have overlapping speech and speech separation technologies such as SepFormer (implemented in SpeechBrain) can separate the speech into individual tracks for each speaker.
- However, before speech separation can be applied, the segments of the recording that contain overlapping speech must be identified. This can be done with a neural network that inputs a short speech segment and outputs the number of speakers present in the segment.
- The output could be 0, 1, 2, or 3, representing no speakers, one speaker, two speakers, or three or more speakers, respectively.  The student is tasked with the following steps to achieve this goal:  Reviewing the literature on speaker counting.
- Implementing a data simulator that creates overlapping speech signals by sampling clean data from a large dataset (e.g. librispeech-clean-100) and adding noise and reverberation from the open-rir dataset with a specified probability (e.g. 0.5).
- Implementing and testing at least two models for speaker counting, such as x-vectors or ECAPA-TDNN (or any other model proposed by the students), with input being a 1-2 second speech segment.
- Implementing the inference stage where the model can process long recordings by chunking them into 1-2 second segments and making a decision on the number of speakers present in each segment. The output should be in the form of a text file indicating the start and end times of each segment and the decision made by the model.
- For instance: 0.00 1.00 0 (mo speech) 1.00 5.50 1 (1 speaker) 5.50 6.55 2 (two speakers) 6.55 10.34  1 (1 speaker)  Where each line contains begin_second, end_second, classifier decision.  Integrating the code into the main SpeechBrain project and making the best model available on the SpeechBrain repository for use by others. Develop and interface for inference purposes, similar to the one available for the VAD.



## **Steps Download, Create and Train all models**

Pull codebase from github
```
!git clone --filter=blob:none --no-checkout https://github.com/SupradeepDanturti/ConvAIProject
%cd ConvAIProject
```

Prepare the dataset

1.   Download Dataset
```
!python prepare_dataset/download_required_data.py --output_folder <destination_folder_path>
```
2.   Create Damples
```
!python prepare_dataset/create_custom_dataset.py dataset.yaml
```
dataset.yaml
```
n_sessions:
  train: 1000 # Creates 1000 sessions per class
  dev: 200 # Creates 200 sessions per class
  eval: 200 # Creates 200 sessions per class
n_speakers: 4 # max number of speakers. In this case the total classes will be 5 (0-4 speakers)
max_length: 120 # max length in seconds for each session/utterance.
```
3. Train XVector Model
```
!python xvector/train_xvector_augmentation.py xvector/hparams_xvector_augmentation.yaml
```
4. Train ECAPA-TDNN Model
```
!python ecapa_tdnn/train_ecapa_tdnn.py ecapa_tdnn/hparams_ecapa_tdnn_augmentation.yaml
```
5. Train Selfsupervised MLP(Linear Classifier)
```
!python selfsupervised/train_selfsupervised_mlp.py selfsupervised/hparams_selfsupervised_mlp.yaml
```
6. Train Selfsupervised XVector
```
!python selfsupervised/train_selfsupervised.py selfsupervised/hparams_selfsupervised_xvector.yaml
```






