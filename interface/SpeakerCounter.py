import torch
from speechbrain.inference.interfaces import Pretrained
import torchaudio
import math
from speechbrain.utils.data_utils import split_path
from speechbrain.utils.fetching import fetch

class SpeakerCounter(Pretrained):
    def __init__(self, *args, **kwargs):
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
        Resample the waveform to a new sample rate.
        """
        if orig_sample_rate != self.sample_rate:
            resample_transform = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=self.sample_rate)
            waveform = resample_transform(waveform)
        return waveform

    def merge_overlapping_segments(self, segments):
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

    def refine_transitions(self, aggregated_predictions):
        """
        Refines transition times by potentially adjusting them to be at the start
        or end of segments, aiming to make the transitions smoother and more accurate.
        """
        refined_predictions = []
        for i in range(len(aggregated_predictions)):
            if i == 0:
                refined_predictions.append(aggregated_predictions[i])
                continue

            current_start, current_end, current_label = aggregated_predictions[i]
            prev_start, prev_end, prev_label = aggregated_predictions[i-1]

            if current_start - prev_end <= 1.0:
                new_start = prev_end
            else:
                new_start = current_start

            refined_predictions.append((new_start, current_end, current_label))

        return refined_predictions

    def refine_transitions_with_confidence(self, aggregated_predictions, segment_confidences):
        refined_predictions = []
        for i in range(len(aggregated_predictions)):
            if i == 0:
                refined_predictions.append(aggregated_predictions[i])
                continue

            current_start, current_end, current_label = aggregated_predictions[i]
            prev_start, prev_end, prev_label, prev_confidence = refined_predictions[-1] + (segment_confidences[i-1],)

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



    def aggregate_segments_with_overlap(self, segment_predictions):
        aggregated_predictions = []
        last_start, last_end, last_label = segment_predictions[0]

        for start, end, label in segment_predictions[1:]:
            if label == last_label and start <= last_end:
                last_end = max(last_end, end)
            else:
                aggregated_predictions.append((last_start, last_end, last_label))
                last_start, last_end, last_label = start, end, label

        aggregated_predictions.append((last_start, last_end, last_label))

        merged = self.merge_overlapping_segments(aggregated_predictions)
        return merged

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
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

    def classify_file(self, path, segment_length=2.0, overlap=1.47, **kwargs):
        """Adjusted to handle overlapped segment predictions and refining transitions"""
        waveform, osr = torchaudio.load(path)
        waveform = self.resample_waveform(waveform, osr)


        """ Attempt - Overlap Segments """
        segments, segment_times = self.create_segments(waveform, segment_length, overlap)
        segment_predictions = []

        for segment, (start_time, end_time) in zip(segments, segment_times):
            rel_length = torch.tensor([1.0])
            emb = self.encode_batch(segment, rel_length)
            out_prob = self.mods.classifier(emb).squeeze(1)
            score, index = torch.max(out_prob, dim=-1)
            text_lab = self.hparams.label_encoder.decode_torch(index)
            segment_predictions.append((start_time, end_time, text_lab[0]))

        aggregated_predictions = self.aggregate_segments_with_overlap(segment_predictions)
        refined_predictions = self.refine_transitions(aggregated_predictions)
        preds = self.refine_transitions_with_confidence(aggregated_predictions , refined_predictions)


        with open("sample_segment_predictions.txt", "w") as file:
            for start_time, end_time, prediction in preds:
                speaker_text = "no speech" if str(prediction) == "0" else ("1 speaker" if str(prediction) == "1" else f"{prediction} speakers")
                print(f"{start_time:.2f}-{end_time:.2f} has {speaker_text}")
                file.write(f"{start_time:.2f}-{end_time:.2f} has {speaker_text}\n")

        """ End of Attempt - Overlap Segments """

    def forward(self, wavs, wav_lens=None):
        """Runs the classification"""
        return self.classify_file(wavs, wav_lens)
