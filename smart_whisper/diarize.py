from typing import List, Optional, Union

import numpy as np
import requests
import torch
from pyannote.audio import Pipeline
from torchaudio import functional as F
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read


class ASRDiarizationPipeline:
    def __init__(
        self,
        asr_pipeline,
        diarization_pipeline,
    ):
        self.asr_pipeline = asr_pipeline
        self.sampling_rate = asr_pipeline.feature_extractor.sampling_rate

        self.diarization_pipeline = diarization_pipeline

    @classmethod
    def from_pretrained(
        cls,
        asr_model: Optional[str] = "openai/whisper-medium",
        *,
        diarizer_model: Optional[str] = "pyannote/speaker-diarization",
        chunk_length_s: Optional[int] = 30,
        use_auth_token: Optional[Union[str, bool]] = True,
        **kwargs,
    ):
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            chunk_length_s=chunk_length_s,
            #token=use_auth_token,
            **kwargs,
        )
        diarization_pipeline = Pipeline.from_pretrained(diarizer_model, use_auth_token=use_auth_token)
        return cls(asr_pipeline, diarization_pipeline)

    def __call__(
        self,
        inputs: Union[np.ndarray, List[np.ndarray]],
        group_by_speaker: bool = True,
        **kwargs,
    ):
        kwargs_asr = {
            argument[len("asr_") :]: value for argument, value in kwargs.items() if argument.startswith("asr_")
        }

        kwargs_diarization = {
            argument[len("diarization_") :]: value for argument, value in kwargs.items() if argument.startswith("diarization_")
        }
        
        inputs, diarizer_inputs = self.preprocess(inputs)

        diarization = self.diarization_pipeline(
            {"waveform": diarizer_inputs, "sample_rate": self.sampling_rate},
            **kwargs_diarization,
        )

        segments = []
        for segment, track, label in diarization.itertracks(yield_label=True):
            segments.append({'segment': {'start': segment.start, 'end': segment.end},
                             'track': track,
                             'label': label})

        new_segments = []
        prev_segment = cur_segment = segments[0]

        for i in range(1, len(segments)):
            cur_segment = segments[i]

            if cur_segment["label"] != prev_segment["label"] and i < len(segments):
                new_segments.append(
                    {
                        "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["start"]},
                        "speaker": prev_segment["label"],
                    }
                )
                prev_segment = segments[i]

        new_segments.append(
            {
                "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["end"]},
                "speaker": prev_segment["label"],
            }
        )

        asr_out = self.asr_pipeline(
            {"array": inputs, "sampling_rate": self.sampling_rate},
            return_timestamps=True,
            **kwargs_asr,
        )
        transcript = asr_out["chunks"]

        # get the end timestamps for each chunk from the ASR output
        end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])
        segmented_preds = []

        # align the diarizer timestamps and the ASR timestamps
        for segment in new_segments:
            # get the diarizer end timestamp
            end_time = segment["segment"]["end"]
            # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
            if end_timestamps.any():
                upto_idx = np.argmin(np.abs(end_timestamps - end_time))

                if group_by_speaker:
                    segmented_preds.append(
                        {
                            "speaker": segment["speaker"],
                            "text": "".join([chunk["text"] for chunk in transcript[: upto_idx + 1]]),
                            "timestamp": (transcript[0]["timestamp"][0], transcript[upto_idx]["timestamp"][1]),
                        }
                    )
                else:
                    for i in range(upto_idx + 1):
                        segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

                transcript = transcript[upto_idx + 1 :]
                end_timestamps = end_timestamps[upto_idx + 1 :]

        return segmented_preds

    def preprocess(self, inputs):
        if isinstance(inputs, str):
            if inputs.startswith("http://") or inputs.startswith("https://"):
                inputs = requests.get(inputs).content
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.sampling_rate)

        if isinstance(inputs, dict):
            if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                raise ValueError(
                    "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                    '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array"
                )

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                # Remove path which will not be used from `datasets`.
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sampling_rate")
            inputs = _inputs
            if in_sampling_rate != self.sampling_rate:
                inputs = F.resample(torch.from_numpy(inputs), in_sampling_rate, self.sampling_rate).numpy()

        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for ASRDiarizePipeline")

        diarizer_inputs = torch.from_numpy(inputs).float()
        diarizer_inputs = diarizer_inputs.unsqueeze(0)

        return inputs, diarizer_inputs