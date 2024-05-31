from fastapi import FastAPI
from pydantic import BaseModel
from Environment import Environment
from Inference import Inference

app = FastAPI()
env = Environment()
inference = Inference(env)


class FuzzParams(BaseModel):
    prompt: str


@app.get("/status")
def status():
    return {"status": "ok"}


@app.post("/v1/fuzz")
def fuzz(params: FuzzParams):
    return {"result": inference.embed([params.prompt])}
