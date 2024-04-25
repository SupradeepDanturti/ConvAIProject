import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils import hpopt as hp

class ECAPABrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""
    
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, lens = self.hparams.wav_augment(wavs, lens)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        predictions, lens = predictions
        spkenc, _ = batch.num_speakers_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            spkenc = self.hparams.wav_augment.replicate_labels(spkenc)

        loss = self.hparams.compute_cost(predictions, spkenc, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, spkenc, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if self.hparams.ckpt_enable:
              self.checkpointer.save_and_keep_only(
                  meta={"ErrorRate": stage_stats["ErrorRate"]},
                  min_keys=["ErrorRate"],
              )
            hp.report_result(stage_stats)
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

def dataio_prep(hparams):
    """Prepares the data IO (loading datasets, defining processing pipelines)"""

    # Initialize the label encoder
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    print(label_encoder)

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav_path):
        sig, fs = torchaudio.load(wav_path)

        sig = torchaudio.functional.resample(sig, fs, 16000).squeeze(0)
        return sig

    # Define label pipeline
    @sb.utils.data_pipeline.takes("num_speakers")
    @sb.utils.data_pipeline.provides("num_speakers_encoded")
    def label_pipeline(num_speakers):
        num_speakers_encoded = label_encoder.encode_label_torch(num_speakers)
        yield num_speakers_encoded

    # Create datasets
    datasets = {}
    for dataset_name in ["train", "valid", "test", "test_0_spk", "test_1_spk", "test_2_spk", "test_3_spk", "test_4_spk"]:
        datasets[dataset_name] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset_name}_annotation"],
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "num_speakers_encoded"],
        )
    print(datasets["train"])
    # Load or compute label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="num_speakers",
    )

    return datasets

if __name__ == "__main__":

    with hp.hyperparameter_optimization(objective_key="ErrorRate") as hp_ctx:
      # Loading the hyperparameters file and command line arguments
      hparams_file, run_opts, overrides = hp_ctx.parse_arguments(
        sys.argv[1:], pass_trial_id=False)

      # Load hyperparameter configuration file
      with open(hparams_file) as fin:
          hparams = load_hyperpyyaml(fin, overrides)

      # Create experiment directory
      sb.core.create_experiment_directory(
          experiment_directory=hparams["output_folder"],
          hyperparams_to_save=hparams_file,
          overrides=overrides,
      )

      # Prepare data IO
      datasets = dataio_prep(hparams)

      # Initialize the Brain object for training the ECAPA-TDNN model
      ecapa_brain = ECAPABrain(
          modules=hparams["modules"],
          opt_class=hparams["opt_class"],
          hparams=hparams,
          run_opts=run_opts,
          checkpointer=hparams["checkpointer"],
      )

      # Train the model
      ecapa_brain.fit(
          epoch_counter=ecapa_brain.hparams.epoch_counter,
          train_set=datasets["train"],
          valid_set=datasets["valid"],
          train_loader_kwargs=hparams["dataloader_options"],
          valid_loader_kwargs=hparams["dataloader_options"],
      )

      # Evaluate the model
      if not hp_ctx.enabled:
        ecapa_brain.evaluate(
          test_set=datasets["test"],
          min_key="error",
          test_loader_kwargs=hparams["dataloader_options"],
        )
