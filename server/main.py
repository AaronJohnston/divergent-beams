from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from Environment import Environment
from Inference import Inference
from Clustering import Clustering

app = FastAPI()
env = Environment()


class TreeParams(BaseModel):
    prompt: str


@app.get("/status")
def status():
    return {"status": "ok"}


@app.post("/v1/tree")
def fuzz(params: TreeParams):
