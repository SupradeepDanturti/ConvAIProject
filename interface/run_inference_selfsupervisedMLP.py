from SpeakerCounterSelfsupervisedMLP import SpeakerCounter
wav_path = "./sample_audio1.wav"
save_dir = "./sample_inference_run/"
model_path = "./selfsupervised_mlp"

audio_classifier = SpeakerCounter.from_hparams(source=model_path, savedir=save_dir)

audio_classifier.classify_file(wav_path)
