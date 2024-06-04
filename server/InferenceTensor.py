from transformers.utils import is_flash_attn_2_available
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import numpy as np
import torch.nn.functional as F
import torch
from datetime import timedelta
import time
from collections import namedtuple
import json

torch.random.manual_seed(0)


class Candidate(namedtuple('Candidate', ['parent_idx', 'content', 'sequence', 'prob'])):
    def __repr__(self):
        return f'Candidate [{self.prob}]: {self.content}'


class Inference:
    def __init__(self):
        print('Flash Attn 2 available', is_flash_attn_2_available())
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
            use_cache=True,
            # attn_implementation='flash_attention_2',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_candidates = 20
        self.max_new_tokens = 100
        self.p_falloff = 0.5 # UNIMPLEMENTED
        self.prune_similar_sequences = True # UNIMPLEMENTED
        self.prune_similar_branches = True # UNIMPLEMENTED
        self.prune_similar_embeddings = True # UNIMPLEMENTED

    def candidates_generator(self, text: str):
        print(text)

        prompt = "<|user|>\n{} <|end|>\n<|assistant|>".format(text)
        inputs = self.tokenizer(prompt, return_tensors='pt')

        max_total_tokens = inputs.input_ids.shape[1] + self.max_new_tokens
        candidates = torch.zeros((self.max_candidates, max_total_tokens), dtype=torch.long, device=self.device)

        candidates[0, :inputs.input_ids.shape[1]] = inputs.input_ids
