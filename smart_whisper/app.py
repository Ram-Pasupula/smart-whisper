from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
from process import transcriber
from fastapi.responses import StreamingResponse
import glob
import time
import os
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
LANGUAGE_CODES = sorted(("en", "es"))


app = FastAPI(
    title="Whisper API",
    debug=True,
    #description="Automatic Speech Recognition API",
    #version="1.0.0",
   # openapi_url="/api/openapi.json",
    docs_url="/",
    redoc_url="/api/redoc",
    # Set generate_schema to False to disable automatic schema generation
    generate_schema=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcode")
async def asr(file: UploadFile = File(...),
              task: Union[str, None] = Query(default="transcribe", enum=[
                  "transcribe"]),
              lang: Union[str, None] = Query(
                  default="en", enum=LANGUAGE_CODES),
              output: Union[str, None] = Query(
    default="txt", enum=["txt",  "json"])
):

    try:
        start_time = time.time()
        file_location = f"/tmp/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        file_path = os.path.dirname(file_location)
        logger.info(f"task : {task}")
        logger.info(f"lang : {lang}")
        logger.info(f"{file_path}/{file.filename}")
        result = transcriber(
            f"{file_path}/{file.filename}", task, lang, output)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time} seconds")
        logger.info(f"Execution time: {elapsed_time} seconds")
        # logger.info(result)
    except Exception:
        raise Exception(status_code=500, detail='File not able to load')
    else:
        return StreamingResponse(
            result,
            media_type="application/octet-stream",
            headers={
                'Content-Disposition': f'attachment; filename="{file.filename}.{output}"'
            })
    finally:
        try:
            files = glob.glob(f"/tmp/{file.filename}")
            for f in files:
                os.remove(f)
        except Exception:
            pass
        else:
            logger.info("Successfully deleted temp files")
