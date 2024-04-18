import torch
from speechbrain.inference.interfaces import Pretrained
import torchaudio


def merge_overlapping_segments(segments):
    """
    Merges segments that overlap or are contiguous, ensuring each speaker segment is represented once.

    Args:
        segments (list of tuples): List of tuples representing (start, end, label) of segments.

    Returns:
        list of tuples: Merged list of segments.
    """
    if not segments:
        return []
    merged = [segments[0]]
    for current in segments[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            if current[2] == prev[2]:
                merged[-1] = (prev[0], max(prev[1], current[1]), prev[2])
            else:
                merged.append(current)
        else:
            merged.append(current)
    return merged


def refine_transitions(aggregated_predictions):
    """
    Refines transitions between speaker segments to enhance accuracy.

    Args:
        aggregated_predictions (list of tuples): The aggregated predictions with potential overlaps.

    Returns:
        list of tuples: Predictions with adjusted transitions.
    """
    refined_predictions = []
    for i in range(len(aggregated_predictions)):
        if i == 0:
            refined_predictions.append(aggregated_predictions[i])
            continue

        current_start, current_end, current_label = aggregated_predictions[i]
        prev_start, prev_end, prev_label = aggregated_predictions[i - 1]

        if current_start - prev_end <= 1.0:
            new_start = prev_end
        else:
            new_start = current_start

        refined_predictions.append((new_start, current_end, current_label))

    return refined_predictions


def refine_transitions_with_confidence(aggregated_predictions, segment_confidences):
    """
    Refines transitions between segments based on confidence levels.

    Args:
        aggregated_predictions (list of tuples): Initial aggregated predictions.
        segment_confidences (list of float): Confidence scores corresponding to each segment.

    Returns:
        list of tuples: Refined segment predictions.
    """
    refined_predictions = []
    for i in range(len(aggregated_predictions)):
        if i == 0:
            refined_predictions.append(aggregated_predictions[i])
            continue

        current_start, current_end, current_label = aggregated_predictions[i]
        prev_start, prev_end, prev_label, prev_confidence = refined_predictions[-1] + (segment_confidences[i - 1],)

        current_confidence = segment_confidences[i]

        if current_label != prev_label:
            if prev_confidence < current_confidence:
                transition_point = current_start
            else:
                transition_point = prev_end
            refined_predictions[-1] = (prev_start, transition_point, prev_label)
            refined_predictions.append((transition_point, current_end, current_label))
        else:
            if prev_confidence < current_confidence:
                refined_predictions[-1] = (prev_start, current_end, current_label)
            else:
                refined_predictions.append((current_start, current_end, current_label))

    return refined_predictions


def aggregate_segments_with_overlap(segment_predictions):
    """
    Aggregates overlapping segments into single segments based on speaker labels.

    Args:
        segment_predictions (list of tuples): List of tuples representing (start, end, label) of segments.

    Returns:
        list of tuples: Aggregated segments.
    """
    aggregated_predictions = []
    last_start, last_end, last_label = segment_predictions[0]

    for start, end, label in segment_predictions[1:]:
        if label == last_label and start <= last_end:
            last_end = max(last_end, end)
        else:
            aggregated_predictions.append((last_start, last_end, last_label))
            last_start, last_end, last_label = start, end, label

    aggregated_predictions.append((last_start, last_end, last_label))

    merged = merge_overlapping_segments(aggregated_predictions)
    return merged


class SpeakerCounter(Pretrained):
    """
    A class for counting speakers in an audio file, built upon the SpeechBrain Pretrained class.
    This class integrates several preprocessing and prediction modules to handle speaker diarization tasks.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the SpeakerCounter with standard and custom parameters.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.sample_rate = self.hparams.sample_rate

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model",
        "classifier",
    ]

    def resample_waveform(self, waveform, orig_sample_rate):
        """
        Resamples the input waveform to the target sample rate specified in the object.

        Args:
            waveform (Tensor): The input waveform tensor.
            orig_sample_rate (int): The original sample rate of the waveform.

        Returns:
            Tensor: The resampled waveform.
        """
        if orig_sample_rate != self.sample_rate:
            resample_transform = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=self.sample_rate)
            waveform = resample_transform(waveform)
        return waveform

    def encode_batch(self, wavs, wav_lens=None):
        """
        Encodes a batch of waveforms into embeddings using the loaded models.

        Args:
            wavs (Tensor): Batch of waveforms.
            wav_lens (Tensor, optional): Lengths of the waveforms for normalization.

        Returns:
            Tensor: Batch of embeddings.
        """
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        return embeddings

    def create_segments(self, waveform, segment_length, overlap):
        """
        Creates segments from a single waveform for batch processing.

        Args:
            waveform (Tensor): Input waveform tensor.
            segment_length (float): Length of each segment in seconds.
            overlap (float): Overlap between segments in seconds.

        Returns:
            tuple: (segments, segment_times) where segments is a list of tensors, and segment_times
            is a list of (start, end) times.
        """
        num_samples = waveform.shape[1]
        segment_samples = int(segment_length * self.sample_rate)
        overlap_samples = int(overlap * self.sample_rate)
        step_samples = segment_samples - overlap_samples
        segments = []
        segment_times = []

        for start in range(0, num_samples - segment_samples + 1, step_samples):
            end = start + segment_samples
            segments.append(waveform[:, start:end])
            start_time = start / self.sample_rate
            end_time = end / self.sample_rate
            segment_times.append((start_time, end_time))

        return segments, segment_times

    def classify_file(self, path, segment_length=2.0, overlap=1.47):
        """
        Processes an audio file to classify and count speakers within segments.
        Utilizes multiple stages of processing to handle overlapping speech and transitions.

        Args:
            path (str): Path to the audio file.
            segment_length (float): Length of each segment in seconds.
            overlap (float): Overlap between segments in seconds.

        Outputs:
            Writes the number of speakers in each segment to a text file.
        """
        waveform, osr = torchaudio.load(path)
        waveform = self.resample_waveform(waveform, osr)

        segments, segment_times = self.create_segments(waveform, segment_length, overlap)
        segment_predictions = []

        for segment, (start_time, end_time) in zip(segments, segment_times):
            rel_length = torch.tensor([1.0])
            emb = self.encode_batch(segment, rel_length)
            out_prob = self.mods.classifier(emb).squeeze(1)
            score, index = torch.max(out_prob, dim=-1)
            text_lab = index.item()
            segment_predictions.append((start_time, end_time, text_lab))

        aggregated_predictions = aggregate_segments_with_overlap(segment_predictions)
        refined_predictions = refine_transitions(aggregated_predictions)
        preds = refine_transitions_with_confidence(aggregated_predictions, refined_predictions)

        with open("sample_segment_predictions.txt", "w") as file:
            for start_time, end_time, prediction in preds:
                speaker_text = "no speech" if str(prediction) == "0" else (
                    "1 speaker" if str(prediction) == "1" else f"{prediction} speakers")
                print(f"{start_time:.2f}-{end_time:.2f} has {speaker_text}")
                file.write(f"{start_time:.2f}-{end_time:.2f} has {speaker_text}\n")

    def forward(self, wavs, wav_lens=None):
        """
        Forward pass for classifying audio using preloaded modules.

        Args:
            wavs (Tensor): Input waveforms.
            wav_lens (Tensor, optional): Lengths of the input waveforms.

        Returns:
            Output from classify_file method.
        """
        return self.classify_file(wavs, wav_lens)
