
import json
from io import StringIO
from threading import Lock
import logging
import torch
from transformers import pipeline
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

MODEL_PATH_W = "/Users/mac/Downloads/whisper-large"
whisper = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH_W,
    device=device,
    torch_dtype=torch_dtype
)
model_lock = Lock()


def transcriber(
        audio,
        task,
        lang,
        output
):
    logging.info("trnascriber started")
    with model_lock:
        kwargs = {"language": f"{lang}", "task": f"{task}"}
        logger.info(f"generate_kwargs :{kwargs}")
        
        logger.info(f"audio :{audio}")
        # whisper.tokenizer.get_decoder_prompt_ids(
        #     language=lang,
        #     task=task,
        # )
        result = whisper(audio,
                         generate_kwargs=kwargs,
                         #  chunk_length_s=30,
                         #  stride_length_s=5,
                         #  batch_size=16
                         )
    output_file = StringIO()
    if "json" in output:
        json.dump(result["text"], output_file)
    else:
        print(result["text"], file=output_file, flush=True)
    output_file.seek(0)
    return output_file

