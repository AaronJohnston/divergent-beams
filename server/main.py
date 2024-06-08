from asyncio import sleep
import json
from Inference import Inference
from Environment import Environment
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from InferenceTensor import InferenceTensor

app = FastAPI()
env = Environment()
inference = InferenceTensor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def status():
    return {"status": "ok"}


@app.get("/api/v1/tree")
def tree(prompt: str):
    return StreamingResponse(inference.candidates_generator(prompt), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
