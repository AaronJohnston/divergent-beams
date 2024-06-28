# Divergent Beams

Aaron Johnston, 2024

A library for sampling diverse outputs from an LLM's probability space.

## Demo

https://aaronjohnston.me/projects/divergent-beams

## Divergent Beams Algorithm

While most language model sampling strategies aim to produce the most
_likely_ output(s), Divergent Beams aims to produce the most
_diverse_ outputs. It does this by sending a set of beams
through the model's probability space, selecting completions that
are probable (controlled by Top-P and Top-K) but as different from
one another as possible (controlled by the gather algorithm). Each
beam keeps track of the joint probability of its entire sequence.
For each token, the algorithm runs at most 2 steps:

### Sample Step

For every active beam, some number of likely next tokens are sampled
according to Top-P and Top-K. These tokens become new beams. The
underlying algorithm takes advantage of GPU concurrency by doing
this sampling in a batch size of 8.

### Gather Step

When the Sample step produces more beams than Max Beams allows, the
beams are gathered into the "most different" representatives, where
"different" is defined as distance in the model's latent space. The
underlying algorithm has implementations of k-Farthest Neighbors or
K-Means for selecting representatives, and the probability mass of
each beam is consolidated into its closest representative.

Each beam terminates when it hits the EOS token or the Max New
Tokens limit.

## Using Library

### Build

```
cd divergent-beams
poetry build
```

### Install

You can use pip to install the locally-built version of the library with `pip install local/path/to/divergent-beams-{version}.tar.gz`

### Use

Here's a complete usage example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from divergent_beams import DivergentBeams

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
    use_cache=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct"
)
eos_token_id = 32007  # Corresponds to <|end|> for Phi-3

# Initialize
divergentBeams = DivergentBeams(model, tokenizer, eos_token_id=eos_token_id, batch_size=8)

# Parameters for generation
top_p = 0.9
top_p_decay = 0.99
top_k = 3
max_beams = 5
max_new_tokens = 50
gather_algo = 'farthest_neighbors'
prompt = 'Generate a 3-utterance conversation between two participants.'

# Generate
sequences, sequence_logprobs = divergentBeams.generator(top_p=top_p, top_p_decay=top_p_decay, top_k=top_k, max_beams=max_beams, max_new_tokens=max_new_tokens, gather_algo=gather_algo, prompt=prompt)

# Can also be used as a generator to get information about the algorithm state at each step, useful for debugging.
for step_info in divergentBeams.generator(top_p=top_p, top_p_decay=top_p_decay, top_k=top_k, max_beams=max_beams, max_new_tokens=max_new_tokens, gather_algo=gather_algo, prompt=prompt):
    print(step_info)

```

## Running Demo Locally

The demo consists of a UI and an inference server.

### Server

Note: the server is designed for use with a single NVidia GPU. Although Huggingface's Optimum library is
used to try and automatically distribute tensors to devices as appropriate, the code is not tested using only
a CPU, with AMD GPUs, or with multiple NVidia GPUs.

```
cd demo/server
conda activate {env} # Or whichever python virtual environment manager you use
pip install -r requirements.txt
python main.py
```

### UI

First, change the server URL (in `demo/ui/src/App.tsx`) to point to your running server instance.

```
cd demo/ui
nvm use
npm install
npm run dev
```

## Models

This library operates using a model and tokenizer from the Huggingface Transformers library. Although any model/tokenizer pair can be passed in, it is currently only tested on models similar to Microsoft's Phi-3-mini-4k-instruct.
