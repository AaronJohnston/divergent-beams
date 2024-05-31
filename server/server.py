from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class FuzzParams(BaseModel):
    prompt: str


@app.get("/status")
def status():
    return {"status": "ok"}


@app.post("/v1/fuzz")
def fuzz(params: FuzzParams):
    return {}
