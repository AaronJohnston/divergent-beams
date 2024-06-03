import time
from transformers.utils import is_flash_attn_2_available
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
from datetime import timedelta
from fastapi import FastAPI
from pydantic import BaseModel
from Environment import Environment
from Inference import Inference
from Clustering import Clustering

app = FastAPI()


class FuzzParams(BaseModel):
    prompt: str


@app.get("/status")
def status():
    return {"status": "ok"}


@app.post("/v1/fuzz")
def fuzz(params: FuzzParams):
    texts = []
    for i in range(4):
        response = inference.generate(params.prompt, seed=i)
        texts.append(response.text)
    embeddings = inference.embed(texts).embeddings
    clusters = clustering.cluster(embeddings)

    print(list(zip(texts, clusters, embeddings)))

    return {"results": "none"}


torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    # attn_implementation="flash_attention_2",
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
streamer = TextStreamer(tokenizer, skip_prompt=True)

print("flash_attn_2 available:", is_flash_attn_2_available())


def gen(text, preview=True):
    duration_start = time.perf_counter()
    prompt = "<|user|>\n{} <|end|>\n<|assistant|>".format(text)
    tokens = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        tokens,
        max_new_tokens=1024,
        return_dict_in_generate=True,
        streamer=streamer if preview else None,
    )
    output_tokens = outputs.sequences[0]
    output_gen_tokens = output_tokens[
        len(tokens[0]): -1
    ]  # From just after prompt to just before <|end|> token
    output_string = tokenizer.decode(output_gen_tokens)
    duration_seconds = time.perf_counter() - duration_start
    if preview:
        print(
            "== took {} ({} toks: {}/tok; {} tps) ==".format(
                timedelta(seconds=duration_seconds),
                len(output_gen_tokens),
                timedelta(seconds=duration_seconds / len(output_gen_tokens)),
                len(output_gen_tokens) / duration_seconds,
            )
        )
        print()
    del tokens, outputs, output_tokens, output_gen_tokens
    return output_string


def forward(text, mean_layers=False, mean_tokens=False, prompt_prefix=""):
    duration_start = time.perf_counter()
    if prompt_prefix:
        prompt = "<|user|>\n{}\n```\n{}\n``` <|end|>\n<|assistant|>".format(
            prompt_prefix, text
        )
    else:
        prompt = "<|user|>\n{} <|end|>\n<|assistant|>".format(text)
    tokens = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model(tokens, output_hidden_states=True)
    embedding = outputs.hidden_states
    # print(len(embedding), embedding[0].shape)
    if mean_layers:
        # print(torch.stack(embedding).shape)
        embedding = torch.stack(embedding).mean(dim=0)  # Mean layers
    else:
        embedding = embedding[-1]  # Take last layer

    if mean_tokens:
        embedding = embedding.mean(dim=1)  # Mean tokens
    else:
        embedding = embedding[:, -1, :]  # Take last token

    embedding = embedding[0]  # Take first and only element of batch

    embedding_cpu = embedding.to("cpu").detach()
    del tokens, outputs, embedding
    return embedding_cpu
