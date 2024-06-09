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

# Not directly comparable to legacy Inference yet --:
# - Remove p falloff from original
# - Are max candidates and max new tokens taken into account the same way?
class InferenceTensor:
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
        self.batch_size = 8
        self.p_falloff = 0.5 # UNIMPLEMENTED
        self.prune_similar_sequences = True # UNIMPLEMENTED
        self.prune_similar_branches = True # UNIMPLEMENTED
        self.prune_similar_embeddings = True # UNIMPLEMENTED

    def candidates_generator(self, text: str):
        print(text)
        candidates, candidate_logprobs = self._init_candidates(text)
        for level_idx in range(self.max_new_tokens):
            candidates, candidate_parents, candidate_logprobs = self._infer(candidates[:self.max_candidates, ...], candidate_logprobs[:self.max_candidates, ...])
            candidate_texts = self.tokenizer.batch_decode(candidates[:, -1])
            candidate_dicts = []
            for i in range(len(candidate_texts)):
                candidate_dicts.append({'content': candidate_texts[i], 'parent': candidate_parents[i].item(), 'prob': candidate_logprobs[i].item()})
            data = json.dumps(candidate_dicts)
            yield f"event: level\nid: {level_idx}\ndata: {data}\n\n"

        yield f"event: level\nid: END\ndata: []\n\n"

    def _init_candidates(self, text: str):
        prompt = "<|user|>\n{} <|end|>\n<|assistant|>".format(text)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        print(self.tokenizer.batch_decode(inputs.input_ids))

        candidates = inputs.input_ids.to(self.device)
        candidate_logprobs = torch.zeros((1), dtype=torch.float32, device=self.device)

        return candidates, candidate_logprobs

    def _top_p_single_batch(self, logits, candidates, candidate_logprobs):
        last_tok_logits = logits[:, -1, :]
        
        sorted_logits, sorted_indices = torch.sort(last_tok_logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create tensor of bools indicating which indices are cumulatively less than top_p
        keep_indices = cum_probs < 0.96

        # Keep the last element that went over top_p
        keep_indices[:, 1:] = keep_indices[:, :-1].clone() # Is this inefficient?
        keep_indices[:, 0] = 1  # Always keep the first element
        
        new_candidate_parents = keep_indices.nonzero()[:, 0]
        
        # OPTIM: Potential optimization -- have a fixed tensor of size (max_candidates, max_tokens) and copy this into that (batch-aware).
        # OPTIM: consider which of these operations can be done in-place to prevent new allocations?
        carryover_candidates = candidates.index_select(0, new_candidate_parents)
        carryover_candidate_logprobs = candidate_logprobs.index_select(0, new_candidate_parents)  # Not strictly necessary since 1d
        
        new_candidate_toks = sorted_indices[keep_indices].unsqueeze(1)
        new_candidate_tok_logprobs = sorted_probs[keep_indices].log()
        
        new_candidates = torch.cat([carryover_candidates, new_candidate_toks], dim=1)
        new_candidate_logprobs = carryover_candidate_logprobs.add_(new_candidate_tok_logprobs)
        
        return new_candidates, new_candidate_parents, new_candidate_logprobs
        

    def _infer(self, candidates, candidate_logprobs):
        with torch.inference_mode():
            num_batches = (candidates.shape[0] + self.batch_size - 1) // self.batch_size  # Round up to nearest whole number of batches
            print('\nnum_batches', num_batches)
            new_candidates_list = []
            new_candidate_parents_list = []
            new_candidate_logprobs_list = []

            for i in range(0, num_batches, 1):
                batch_candidates = candidates[i * self.batch_size:(i + 1) * self.batch_size]
                batch_candidate_logprobs = candidate_logprobs[i * self.batch_size:(i + 1) * self.batch_size]

                batch_outputs = self.model(input_ids=batch_candidates)
                
                # TODO: Pruning step based on K-Means Clustering of embeddings here
                
                new_batch_candidates, new_batch_candidate_parents, new_batch_candidate_logprobs = self._top_p_single_batch(batch_outputs.logits, batch_candidates, batch_candidate_logprobs)
                new_candidates_list.append(new_batch_candidates)
                new_candidate_parents_list.append(new_batch_candidate_parents)
                new_candidate_logprobs_list.append(new_batch_candidate_logprobs)
                
            return torch.cat(new_candidates_list), torch.cat(new_candidate_parents_list), torch.cat(new_candidate_logprobs_list)