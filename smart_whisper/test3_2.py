from transformers import pipeline

from pyannote.audio import Pipeline
import os
import torch

from util import format_as_transcription
from speechbox import ASRDiarizationPipeline
HF_TOKEN = os.environ.get("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")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", cache_dir="./speaker-diarization"
)


asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
)

#pipeline = ASRDiarizationPipeline.from_pretrained("openai/whisper-base")

pipeline = ASRDiarizationPipeline(
    asr_pipeline=asr_pipeline, diarization_pipeline=diarization_pipeline
)
# pipeline = ASRDiarizationPipeline.from_pretrained(
#     #asr_model="openai/whisper-base",
#     diarizer_model="pyannote/speaker-diarization"
#     ,use_auth_token="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"
#    # , device=device
# )
response = pipeline("/Users/mac/Downloads/source/audio.wav")

#print(response)
print(format_as_transcription(response))