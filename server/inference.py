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
            # low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct")
        self.max_candidates = 10

    def top_p_tokens(self, logits, top_p=0.9):
        """Does not support batches yet. logits must be of shape (VOCAB_SIZE)."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        # Create tensor of bools indicating which indices are cumulatively less than top_p
        sorted_keep_indices = cum_probs < 0.9
        # Keep the last element that went over top_p
        sorted_keep_indices[1:] = sorted_keep_indices[:-1].clone()
        sorted_keep_indices[0] = 1  # Always keep the first element
        keep_toks = sorted_indices[sorted_keep_indices]
        keep_probs = probs[sorted_keep_indices]
        return keep_toks, keep_probs

    def get_candidate_dicts(self, candidates):
        candidate_dicts = []
        for candidate in candidates:
            candidate_dict = {
                'content': candidate.content,
                'prob': candidate.prob,
            }
            if candidate.parent_idx is not None:
                candidate_dict['parent'] = candidate.parent_idx
            candidate_dicts.append(candidate_dict)
        return candidate_dicts

    async def candidates_generator(self, text: str):
        prompt = "<|user|>\n{} <|end|>\n<|assistant|>".format(text)
        inputs = self.tokenizer(prompt, return_tensors='pt')

        candidates = [
            Candidate(None, prompt, inputs.input_ids, 1.0)
        ]

        p = 0.99

        finished = []
        # OPTIM: Batch inference, which means rewrite top_p_tokens to use batch
        # OPTIM: Keep previous_key values
        # OPTIM: Use Tensors to keep track of candidates (with masked values)
        # OPTIM: Log probs
        for i in list(range(10)):
            new_candidates = []
            for candidate_idx, candidate in enumerate(candidates):
                outputs = self.model(input_ids=candidate.sequence)
                new_toks, new_probs = self.top_p_tokens(
                    outputs.logits[0, -1, :], p)
                for new_tok, new_prob in zip(new_toks, new_probs):
                    new_candidate = Candidate(
                        candidate_idx,
                        self.tokenizer.decode(new_tok),
                        torch.cat(
                            [candidate.sequence, new_tok.unsqueeze(0).unsqueeze(0)], dim=1),
                        candidate.prob * new_prob.item()
                    )
                    if new_tok == 32000 or new_tok == self.tokenizer.eos_token_id:
                        finished.append(new_candidate)
                    else:
                        new_candidates.append(new_candidate)
            candidates = new_candidates[:self.max_candidates]

            candidate_dicts = self.get_candidate_dicts(candidates)
            print(candidate_dicts)

            data = json.dumps(candidate_dicts)
            yield f"event: level\ndata: {data}\n\n"

            # print(i, p, len(candidates))
            # for candidate in candidates:
            #     print(candidate)
            # print()

            p *= 0.5
