from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from Environment import Environment
from Inference import Inference
import json
from asyncio import sleep

app = FastAPI()
env = Environment()
inference = Inference()


async def generator():
    for i in range(10):
        data = json.dumps({'level_num': i})
        yield f"event: level\ndata: {data}\n\n"
        await sleep(1)


class TreeParams(BaseModel):
    prompt: str


@app.get("/status")
def status():
    return {"status": "ok"}


@app.post("/v1/sample")
def sample(params: TreeParams):
    return StreamingResponse(inference.candidates_generator(params.prompt), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
