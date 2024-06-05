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
        candidates, candidate_masks, candidate_parents, candidate_logprobs = self._init_candidates(text)

        

    def _init_candidates(self, text: str):
        prompt = "<|user|>\n{} <|end|>\n<|assistant|>".format(text)
        inputs = self.tokenizer(prompt, return_tensors='pt')

        max_total_tokens = inputs.input_ids.shape[1] + self.max_new_tokens

        # (max_candidates, max_total_tokens)
        candidates = torch.zeros((self.max_candidates, max_total_tokens), dtype=torch.long, device=self.device)
        # (max_candidates, max_total_tokens)
        candidate_masks = torch.zeros((self.max_candidates, max_total_tokens), dtype=torch.bool, device=self.device)
        # (max_candidates)
        candidate_parents = torch.zeros((self.max_candidates), dtype=torch.long, device=self.device)
        # (max_candidates)
        candidate_logprobs = torch.zeros((self.max_candidates), dtype=torch.float32, device=self.device)

        candidates[0, :inputs.input_ids.shape[1]] = inputs.input_ids
        candidate_masks[0, :inputs.input_ids.shape[1]] = inputs.attention_mask
        candidate_parents[0] = 0
        candidate_logprobs[0] = 0.0

        return candidates, candidate_masks, candidate_parents, candidate_logprobs
    
    def _infer(self, candidates, candidate_masks, candidate_parents, candidate_logprobs):
        outputs = self.model(input_ids=candidates, attention_mask=candidate_masks)