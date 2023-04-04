from fastapi import FastAPI, Request
import argparse
import models
import uvicorn, json, datetime
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

app = FastAPI()

@app.post("/")
async def create_item(request: Request):
    global model
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    # history = json_post_list.get('history')
    # max_length = json_post_list.get('max_length')
    # top_p = json_post_list.get('top_p')
    # temperature = json_post_list.get('temperature')
    output = model.run(prompt)
    if isinstance(output, tuple):
        response, history = output
    else:
        response = output
        history = []
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=models.availabel_models)
    args = parser.parse_args()
    model = models.get_model(args)
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)