import json
from io import StringIO
from threading import Lock
import logging
import torchaudio
# pip install pyannote.audio
from pyannote.audio import Pipeline
import requests

from typing import List, Optional, Union

import numpy as np
import requests
import torch
from pyannote.audio import Pipeline
from torchaudio import functional as F
from diarize_process import DiarizationPipeline
from transformers.pipelines.audio_utils import ffmpeg_read


diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", cache_dir="./speaker-diarization"
)

pipeline = DiarizationPipeline(
 diarization_pipeline=diarization_pipeline
)
#result = diarization_pipeline("/Users/mac/Downloads/source/005-Greetings.mp3")
segments, inputs = pipeline("/Users/mac/Downloads/source/005-Greetings.mp3")

#print(response)
print(segments)
print(inputs)

