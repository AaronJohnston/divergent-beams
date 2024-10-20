print('Starting up server...')

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from divergent_beams import DivergentBeams

# quantization_config = BitsAndBytesConfig(load_in_4bit=True)

print('Resolved imports, initializing modules...')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aaronjohnston.me", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model-specific config
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
    use_cache=True,
    # quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct"
)
eos_token_id = 32007  # Corresponds to <|end|> for Phi-3
def format_prompt(text):
    return "<|user|>\n{} <|end|>\n<|assistant|>".format(text)


print('Finished initializing modules')
print('Memory footprint', model.get_memory_footprint())

divergentBeams = DivergentBeams(model, tokenizer, eos_token_id)


@app.get("/api/status")
def status():
    return {"status": "ok"}


@app.get("/api/v1/tree")
def tree(topP: float, topK: int, maxBeams: int, maxNewTokens: int, gatherAlgo: str, prompt: str, topPDecay: float = 1.0):
    return StreamingResponse(divergentBeams.generator(topP, topPDecay, topK, maxBeams, maxNewTokens, gatherAlgo, format_prompt(prompt)), media_type="text/event-stream")



# Allows running the demo server with `python main.py`.
# In production, recommended to use a production server like gunicorn with uvicorn service workers.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")