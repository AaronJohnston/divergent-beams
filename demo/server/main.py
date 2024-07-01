print('Starting up server...')

from asyncio import sleep
import json
from typing import Union
from Inference import Inference
from Environment import Environment
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from InferenceTensor import InferenceTensor

print('Resolved imports, initializing modules...')

app = FastAPI()
env = Environment()
inference = InferenceTensor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://aaronjohnston.me", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def status():
    return {"status": "ok"}


@app.get("/api/v1/tree")
def tree(topP: float, topK: int, maxBeams: int, maxNewTokens: int, gatherAlgo: str, prompt: str, topPDecay: float = 1.0):
    return StreamingResponse(inference.candidates_generator(topP, topPDecay, topK, maxBeams, maxNewTokens, gatherAlgo, prompt), media_type="text/event-stream")


# Allows running the demo server with `python main.py`.
# In production, recommended to use a production server like gunicorn with uvicorn service workers.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")