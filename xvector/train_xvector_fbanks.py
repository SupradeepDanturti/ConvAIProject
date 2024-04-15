
import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils import hpopt as hp
import torchaudio


class XVectorSpkCounter(sb.Brain):
    """
    A custom Brain class for training and evaluating a speaker counting model.
    This class is designed to handle the forward pass, loss computation, and
    the training and validation cycles, leveraging SpeechBrain's workflow.
    """

    def compute_forward(self, batch, stage):
        """
        Processes the input batch to produce model predictions.

        Parameters:
        - batch (PaddedBatch): Contains all tensors needed for computation.
        - stage (sb.Stage): The stage of the pipeline (TRAIN, VALID, or TEST).

        Returns:
        - Tensor: Posterior probabilities over the number of classes.
        """

        batch = batch.to(self.device)
        feats, lens = self.prepare_features(batch.sig, stage)
        embeddings = self.modules.embedding_model(feats, lens)
        predictions = self.modules.classifier(embeddings)

        return predictions

    def prepare_features(self, wavs, stage):
        """
        Prepares the signal features for model computation, applying
        waveform augmentation and feature extraction.

        Parameters:
        - wavs (tuple): Tuple of signals and their lengths.
        - stage (sb.Stage): Current training stage.

        Returns:
        - Tuple[Tensor, Tensor]: Features and their lengths.
        """

        wavs, lens = wavs

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        return feats, lens

    def compute_objectives(self, predictions, batch, stage):
        """
        Computes the loss given predictions and targets.

        Parameters:
        - predictions (Tensor): Model predictions.
        - batch (PaddedBatch): Batch providing the targets.
        - stage (sb.Stage): The training stage.

        Returns:
        - Tensor: The loss tensor.
        """

        _, lens = batch.sig
        spks, _ = batch.num_speakers_encoded
        
        # Compute the cost function
        loss = sb.nnet.losses.nll_loss(predictions, spks, lens)

        self.loss_metric.append(
            batch.id, predictions, spks, lens, reduction="batch"
        )

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, spks, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """
        Initializes trackers at the beginning of each stage.

        Parameters:
        - stage (sb.Stage): Current stage.
        - epoch (int, optional): Current epoch number.
        """

        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """
        Handles logging and learning rate adjustments at the end of each stage.

        Parameters:
        - stage (sb.Stage): Current stage.
        - stage_loss (float): Average loss of the stage.
        - epoch (int, optional): Current epoch number.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            if self.hparams.ckpt_enable:
                self.checkpointer.save_and_keep_only(
                    meta=stats, min_keys=["error"]
                )
            hp.report_result(stats)

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


def dataio_prep(hparams):
    """
    Prepares and returns datasets for training, validation, and testing.
    Parameters:
    - hparams (dict): A dictionary of hyperparameters for data preparation.
    Returns:
    - datasets (dict): A dictionary containing 'train', 'valid', and 'test' datasets.
    """

    # Initialization of the label encoder.
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav_path):
        """
        Audio processing pipeline that loads and returns an audio signal.
        Parameters:
            - wav_path (str): Path to the audio file.
        Returns:
            - sig (Tensor): Loaded audio signal tensor.
        """
        sig, fs = torchaudio.load(wav_path)

        # Resampling
        sig = torchaudio.functional.resample(sig, fs, 16000).squeeze(0)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("num_speakers")
    @sb.utils.data_pipeline.provides("num_speakers", "num_speakers_encoded")
    def label_pipeline(num_speakers):
        """
        Processes and encodes the number of speakers.

        Parameters:
        - num_speakers (int): The number of speakers in the audio.

        Yields:
        - num_speakers (int): The original number of speakers.
        - num_speakers_encoded (Tensor): Encoded tensor of the number of speakers.
        """
        yield num_speakers
        num_speakers_encoded = label_encoder.encode_label_torch(num_speakers)
        yield num_speakers_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    hparams["dataloader_options"]["shuffle"] = True
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "num_speakers_encoded"],
        )

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="num_speakers",
    )

    return datasets


# RECIPE BEGINS!
if __name__ == "__main__":

    with hp.hyperparameter_optimization(objective_key="error") as hp_ctx:

        # Reading command line arguments
        hparams_file, run_opts, overrides = hp_ctx.parse_arguments(
            sys.argv[1:], pass_trial_id=False
        )

        # Load hyperparameters file with command-line overrides.
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )

        # Create dataset objects "train", "valid", and "test".
        datasets = dataio_prep(hparams)

        # Initialize the Brain object to prepare for mask training.
        spk_counter = XVectorSpkCounter(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )

        spk_counter.fit(
            epoch_counter=spk_counter.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
        if not hp_ctx.enabled:
            # Load the best checkpoint for evaluation
            test_stats = spk_counter.evaluate(
                test_set=datasets["test"],
                min_key="error",
                test_loader_kwargs=hparams["dataloader_options"],
            )

