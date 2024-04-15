import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


class SelfSupervisedSpeakerCounter(sb.Brain):
    def compute_forward(self, batch, stage):
        """
        Forward pass for generating predictions from input batches.

        Parameters:
        - batch (dict): The batch of data to process.
        - stage (sb.Stage): The stage of the process (TRAIN, VALID, or TEST).

        Returns:
        - outputs (Tensor): The output predictions from the classifier.
        """

        batch = batch.to(self.device)
        wavs, lens = batch.sig

        outputs = self.modules.ssl_model(wavs, lens)
        feats = self.modules.mean_var_norm(outputs, lens)
        embeddings = self.modules.embedding_model(feats, lens)
        outputs = self.modules.classifier(embeddings)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """
        Computes the loss for the current batch and stage.

        Parameters:
        - predictions (Tensor): The predictions made by the model.
        - batch (dict): The batch of data including labels.
        - stage (sb.Stage): The current stage (TRAIN, VALID, or TEST).

        Returns:
        - loss (Tensor): The computed loss value.
        """

        spkid, _ = batch.num_speakers_encoded
        predictions = predictions.squeeze(1)
        spkid = spkid.squeeze(1)

        loss = self.hparams.compute_cost(predictions, spkid)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, spkid)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """
        Called at the beginning of each stage to setup metrics and state.

        Parameters:
        - stage (sb.Stage): The current stage (TRAIN, VALID, or TEST).
        - epoch (int, optional): The current epoch number.
        """

        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """
        Called at the end of each stage to summarize and log the stage results.

        Parameters:
        - stage (sb.Stage): The current stage (TRAIN, VALID, or TEST).
        - stage_loss (float): The average loss of the stage.
        - epoch (int, optional): The current epoch number, if applicable.
        """

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        if stage == sb.Stage.VALID:
            # Learning rate adjustments and logging
            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            (
                old_lr_ssl,
                new_lr_ssl,
            ) = self.hparams.lr_annealing_ssl(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(
                self.ssl_optimizer, new_lr_ssl
            )

            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "ssl_lr": old_lr_ssl},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def init_optimizers(self):
        """
        Initializes optimizers for the SSL model and the main model.
        """
        self.ssl_optimizer = self.hparams.ssl_opt_class(
            self.modules.ssl_model.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "ssl_opt", self.ssl_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "model_optimizer": self.optimizer,
            "ssl_optimizer": self.ssl_optimizer,
        }


def dataio_prep(hparams):
    """
    Prepares and returns datasets for training, validation, and testing.
    Parameters:
    - hparams (dict): A dictionary of hyperparameters for data preparation.
    Returns:
    - datasets (dict): A dictionary containing 'train', 'valid', and 'test' datasets.
    """

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
        sig = sb.dataio.dataio.read_audio(wav_path)
        return sig
    
    # Label Encoder
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

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

    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
        "test_annotation_0_spk": hparams["test_annotation_0_spk"],
        "test_annotation_1_spk": hparams["test_annotation_1_spk"],
        "test_annotation_2_spk": hparams["test_annotation_2_spk"],
        "test_annotation_3_spk": hparams["test_annotation_3_spk"],
        "test_annotation_4_spk": hparams["test_annotation_4_spk"],
    }
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

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

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

    hparams["ssl_model"] = hparams["ssl_model"].to(device=run_opts["device"])
    # freeze the feature extractor part when unfreezing
    if not hparams["freeze_ssl"] and hparams["freeze_ssl_conv"]:
        hparams["ssl_model"].model.feature_extractor._freeze_parameters()

    # Initialize the Brain object to prepare for mask training.
    spkcounter = SelfSupervisedSpeakerCounter(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    spkcounter.fit(
        epoch_counter=spkcounter.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    test_stats = spkcounter.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["dataloader_options"],
    )
    """
    To get test accuracy on each class uncomment the code below.
    """
    # print("Evaluating on no spk class")
    # test_stats = spkcounter.evaluate(
    #     test_set=datasets["test_annotation_0_spk"],
    #     min_key="error_rate",
    #     test_loader_kwargs=hparams["dataloader_options"],
    # )
    #
    # print("Evaluating on 1 spk class")
    # test_stats = spkcounter.evaluate(
    #     test_set=datasets["test_annotation_1_spk"],
    #     min_key="error_rate",
    #     test_loader_kwargs=hparams["dataloader_options"],
    # )
    #
    # print("Evaluating on 2 spk class")
    # test_stats = spkcounter.evaluate(
    #     test_set=datasets["test_annotation_2_spk"],
    #     min_key="error_rate",
    #     test_loader_kwargs=hparams["dataloader_options"],
    # )
    #
    # print("Evaluating on 3 spk class")
    # test_stats = spkcounter.evaluate(
    #     test_set=datasets["test_annotation_3_spk"],
    #     min_key="error_rate",
    #     test_loader_kwargs=hparams["dataloader_options"],
    # )
    #
    # print("Evaluating on 4 spk class")
    # test_stats = spkcounter.evaluate(
    #     test_set=datasets["test_annotation_4_spk"],
    #     min_key="error_rate",
    #     test_loader_kwargs=hparams["dataloader_options"],
    # )

