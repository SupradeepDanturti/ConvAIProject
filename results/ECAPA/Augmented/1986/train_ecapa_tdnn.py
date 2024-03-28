        
#!/usr/bin/env python3

import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

class ECAPABrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""
    
    def compute_forward(self, batch, stage):
        """Forward pass to compute embeddings and class predictions."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Data augmentation (if in training stage)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, lens = self.hparams.wav_augment(wavs, lens)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embedding and classification
        embeddings = self.modules.embedding_model(feats)
        predictions = self.modules.classifier(embeddings)

        return predictions, lens

    def compute_objectives(self, predictions, batch, stage):
        """Compute the loss (NLL) between predicted and true labels."""
        predictions, lens = predictions
        uttid = batch.id
        targets, _ = batch.num_speakers_encoded

        # Concatenate labels (if data augmentation is used)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            targets = self.hparams.wav_augment.replicate_labels(targets)

        loss = sb.nnet.losses.nll_loss(predictions, targets, lens)

        # Metric tracking
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, targets, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Called at the beginning of each epoch, sets up metrics"""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Called at the end of an epoch, summarizes metrics, adjusts learning rate"""
        if stage == sb.Stage.TRAIN:
            self.train_stats = {"loss": stage_loss}
        else:
            stats = {"loss": stage_loss, "error": self.error_metrics.summarize("average")}

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

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
    def label_pipeline(spk_id):
        num_speakers_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield num_speakers_encoded

    # Create datasets
    datasets = {}
    for dataset_name in ["train", "valid", "test"]:
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
    # Loading the hyperparameters file and command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

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
    if not sb.utils.distributed.if_multiple_gpus(run_opts):
        ecapa_brain.evaluate(
            test_set=datasets["test"],
            min_key="error",
            test_loader_kwargs=hparams["dataloader_options"],
        )

