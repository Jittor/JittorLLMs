import argparse
import datetime
import logging
import typing as t

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import models

logger = logging.getLogger(__name__)

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

app = FastAPI()

model: models.LLMModel


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    status: t.Optional[int] = 200
    response: str
    history: t.Optional[list] = []
    time: str


@app.post("/", response_model=ChatResponse)
async def chat_completions(request: ChatRequest) -> ChatResponse:
    prompt = request.prompt
    output = model.run(prompt)
    if isinstance(output, tuple):
        response, history = output
    else:
        response = output
        history = []
    logger.info(f"prompt: {prompt}, response: {response}")
    return ChatResponse(response=response, history=history, time=datetime.datetime.now().strftime(DATETIME_FORMAT))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=models.availabel_models)
    args = parser.parse_args()
    model = models.get_model(args)
    logger.info(f"model<{args}> load success")

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
