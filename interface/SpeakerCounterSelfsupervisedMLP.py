import torch
from speechbrain.inference.interfaces import Pretrained
import torchaudio


def merge_overlapping_segments(segments):
    """
    Merges overlapping or adjacent segments that share the same label.

    Args:
        segments (list of tuples): Segments formatted as (start, end, label).

    Returns:
        list of tuples: Merged segments.
    """
    if not segments:
        return []
    merged = [segments[0]]
    for current in segments[1:]:
        prev = merged[-1]
        if current[0] <= prev[1] and current[2] == prev[2]:
            merged[-1] = (prev[0], max(prev[1], current[1]), prev[2])
        else:
            merged.append(current)
    return merged


def refine_transitions(aggregated_predictions):
    """
    Adjusts the start times of segments to ensure smooth transitions.

    Args:
        aggregated_predictions (list of tuples): Segments formatted as (start, end, label).

    Returns:
        list of tuples: Segments with refined transitions.
    """
    refined_predictions = []
    for i, (current_start, current_end, current_label) in enumerate(aggregated_predictions):
        if i == 0:
            refined_predictions.append((current_start, current_end, current_label))
            continue
        prev_start, prev_end, prev_label = refined_predictions[-1]
        new_start = prev_end if current_start - prev_end <= 1.0 else current_start
        refined_predictions.append((new_start, current_end, current_label))
    return refined_predictions


def refine_transitions_with_confidence(aggregated_predictions, segment_confidences):
    """
    Further refines segment transitions based on confidence scores.

    Args:
        aggregated_predictions (list of tuples): Initial segment predictions.
        segment_confidences (list): Confidence scores for each segment.

    Returns:
        list of tuples: Refined segment predictions with adjusted transitions.
    """
    refined_predictions = [aggregated_predictions[0]]
    for i in range(1, len(aggregated_predictions)):
        current_start, current_end, current_label = aggregated_predictions[i]
        prev_start, prev_end, prev_label, prev_confidence = refined_predictions[-1] + (segment_confidences[i - 1],)
        current_confidence = segment_confidences[i]

        transition_point = current_start if prev_confidence < current_confidence else prev_end
        refined_predictions[-1] = (prev_start, transition_point, prev_label)
        refined_predictions.append((transition_point, current_end, current_label))
    return refined_predictions


def aggregate_segments_with_overlap(segment_predictions):
    """
    Aggregates contiguous segments with the same label to minimize redundancy.

    Args:
        segment_predictions (list of tuples): Predictions containing overlapping segments.

    Returns:
        list of tuples: Aggregated segment predictions.
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
    return merge_overlapping_segments(aggregated_predictions)


class SpeakerCounter(Pretrained):
    """
    A class derived from Pretrained to count speakers in audio files.
    This class processes audio inputs to detect and count the number of speakers.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the SpeakerCounter with necessary parameters and modules.
        Inherits settings and methods from Pretrained class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.sample_rate = self.hparams.sample_rate  # Desired sample rate for processing.

    MODULES_NEEDED = [
        "output_mlp",
        "log_softmax",
        "avg_pool",
        "ssl_model",
    ]

    def resample_waveform(self, waveform, orig_sample_rate):
        """
        Resamples the provided waveform to the target sample rate if necessary.

        Args:
            waveform (Tensor): The audio waveform tensor.
            orig_sample_rate (int): The original sampling rate of the waveform.

        Returns:
            Tensor: The resampled waveform.
        """
        if orig_sample_rate != self.sample_rate:
            resample_transform = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=self.sample_rate)
            waveform = resample_transform(waveform)
        return waveform

    def encode_batch(self, wavs, wav_lens=None):
        """
        Encodes a batch of waveforms into features for classification.

        Args:
            wavs (Tensor): Batch of waveform tensors.
            wav_lens (Tensor, optional): Lengths of each waveform in the batch.

        Returns:
            Tensor: Encoded outputs ready for classification.
        """
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()
        feats = self.mods.ssl_model(wavs, wav_lens)
        outputs = self.mods.avg_pool(feats, wav_lens)
        outputs = outputs.view(outputs.shape[0], -1)
        return outputs

    def create_segments(self, waveform, segment_length, overlap):
        """
        Creates segments from an audio waveform based on specified segment length and overlap.

        Args:
            waveform (Tensor): The waveform tensor.
            segment_length (float): Duration of each segment in seconds.
            overlap (float): Overlap duration between segments in seconds.

        Returns:
            tuple: (segments, segment_times) where segments is a list of tensors and segment_times
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
        Processes an audio file, classifies segments, and writes predictions to a file.

        Args:
            path (str): Path to the audio file.
            segment_length (float): Length of each audio segment in seconds.
            overlap (float): Overlap between segments in seconds.
        """
        waveform, osr = torchaudio.load(path)
        waveform = self.resample_waveform(waveform, osr)
        segments, segment_times = self.create_segments(waveform, segment_length, overlap)
        segment_predictions = []

        for segment, (start_time, end_time) in zip(segments, segment_times):
            rel_length = torch.tensor([1.0])
            outputs = self.encode_batch(segment, rel_length)
            outputs = self.mods.output_mlp(outputs)
            out_prob = self.mods.log_softmax(outputs)
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
        Primary forward pass to classify audio input.

        Args:
            wavs (Tensor): Input waveforms.
            wav_lens (Tensor, optional): Lengths of each waveform input.

        Returns:
            Outputs from classify_file method.
        """
        return self.classify_file(wavs, wav_lens)
