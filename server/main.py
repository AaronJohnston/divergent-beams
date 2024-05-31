from fastapi import FastAPI
from pydantic import BaseModel
from Environment import Environment
from Inference import Inference
from Clustering import Clustering

app = FastAPI()
env = Environment()
inference = Inference(env)
clustering = Clustering()


class FuzzParams(BaseModel):
    prompt: str


@app.get("/status")
def status():
    return {"status": "ok"}


@app.post("/v1/fuzz")
def fuzz(params: FuzzParams):
    texts = []
    for i in range(10):
        response = inference.generate(params.prompt, seed=i)
        texts.append(response["text"])
    embeddings = inference.embed(texts)["embeddings"]
    clusters = clustering.cluster(embeddings)

    return {"results": list(zip(texts, clusters))}
