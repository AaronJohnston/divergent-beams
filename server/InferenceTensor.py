from transformers.utils import is_flash_attn_2_available
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import numpy as np
import torch.nn.functional as F
import torch
from datetime import timedelta
import time
from collections import namedtuple
import json
from sklearn.cluster import KMeans

torch.random.manual_seed(0)

# Debugging functions used in interactive mode in Jupyter notebook
def D(*args):
    pass

def DS(*args):
    pass

def check_gpu(*args):
    pass

def display(*args):
    pass

class InferenceTensor:
    def __init__(self):
        print('Initializing model...')
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
            use_cache=True,
            # attn_implementation='flash_attention_2',
        )
        print('Initializing tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_new_tokens = 10
        self.batch_size = 8
        
    def candidates_generator(self, top_p: float, max_beams: int, prompt: str):
        print(prompt)
        candidates, candidate_logprobs = self._init_candidates(prompt)
        for level_idx in range(self.max_new_tokens):
            logits, embeddings = self._infer(candidates, candidate_logprobs)
        
            if candidates.shape[0] > max_beams:
                candidates, candidate_parents, candidate_aunts, candidate_logprobs, logits = self._k_means(logits, embeddings, candidates, candidate_logprobs, max_beams)
                yield self._format_k_means(level_idx, candidates, candidate_parents, candidate_aunts, candidate_logprobs)

            candidates, candidate_parents, candidate_logprobs = self._top_p(logits, candidates, candidate_logprobs, top_p)
            yield self._format_top_p(level_idx, candidates, candidate_parents, candidate_logprobs)

        yield f"event: message\nid: END\ndata: []\n\n"

    def _format_k_means(self, level_idx, candidates, candidate_parents, candidate_aunts, candidate_logprobs):
        candidate_texts = self.tokenizer.batch_decode(candidates[:, -1])
        candidate_probs = candidate_logprobs.exp()
        candidate_dicts = []
        idx = f"{level_idx}-k"
        for i in range(len(candidate_texts)):
            candidate_dicts.append({'content': candidate_texts[i], 'parent': candidate_parents[i], 'aunts': candidate_aunts[i], 'prob': candidate_probs[i].item()})
        data = json.dumps({'id': idx, 'level_type': 'k_means', 'nodes': candidate_dicts})
        return f"event: message\nid: {idx}\"\ndata: {data}\n\n"

    def _format_top_p(self, level_idx, candidates, candidate_parents, candidate_logprobs):
        candidate_texts = self.tokenizer.batch_decode(candidates[:, -1])
        candidate_probs = candidate_logprobs.exp()
        candidate_dicts = []
        idx = f"{level_idx}-p"
        for i in range(len(candidate_texts)):
            candidate_dicts.append({'content': candidate_texts[i], 'parent': candidate_parents[i], 'prob': candidate_probs[i].item()})
        data = json.dumps({'id': idx, 'level_type': 'top_p', 'nodes': candidate_dicts})
        return f"event: message\nid: {idx}\ndata: {data}\n\n"


    def _init_candidates(self, text: str):
        prompt = "<|user|>\n{} <|end|>\n<|assistant|>".format(text)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        D(inputs.input_ids, 'input_ids')
        print(self.tokenizer.batch_decode(inputs.input_ids))

        candidates = inputs.input_ids.to(self.device)
        candidate_logprobs = torch.zeros((1), dtype=torch.float32, device=self.device)

        return candidates, candidate_logprobs

    def _k_means(self, logits, embeddings, candidates, candidate_logprobs, max_beams):
        D(candidates, 'candidates')
        D(candidate_logprobs, 'candidate_logprobs')
        # === CPU ===
        embeddings_np = embeddings.float().numpy(force=True)
        D(embeddings_np, 'embeddings_np')
        k_means = KMeans(n_clusters=min(2, embeddings_np.shape[0]), random_state=0, n_init="auto")
        k_mean_space = k_means.fit_transform(embeddings_np)
        D(k_mean_space, 'k_mean_space')
        k_mean_clusters = k_means.predict(embeddings_np)
        D(k_mean_clusters, 'k_mean_clusters')
        k_mean_logprob_mass = np.bincount(k_mean_clusters, weights=candidate_logprobs.cpu())
        D(k_mean_logprob_mass, 'k_mean_logprob_mass')
        closest = np.argmin(k_mean_space, axis=0)
        D(closest, 'closest')
        # === END CPU ===
        
        closest_indices = torch.from_numpy(closest).to(self.device)
        new_candidates = candidates.index_select(0, closest_indices)
        D(new_candidates, 'new_candidates')
        new_candidate_parents = closest_indices.tolist()
        D(new_candidate_parents, 'new_candidate_parents')
        new_candidate_aunts = [torch.nonzero(torch.from_numpy(k_mean_clusters).to(self.device) == i).squeeze(-1).tolist() for i in range(new_candidates.shape[0])]
        D(new_candidate_aunts, 'new_candidate_aunts')
        new_candidate_logprobs = torch.from_numpy(k_mean_logprob_mass).to(self.device)
        D(new_candidate_logprobs, 'new_candidate_logprobs')
        new_candidate_logits = logits.index_select(0, closest_indices)
        
        return new_candidates, new_candidate_parents, new_candidate_aunts, new_candidate_logprobs, new_candidate_logits
        
    def _top_p(self, logits, candidates, candidate_logprobs, top_p):
        D(candidates, 'candidates')
        D(candidate_logprobs, 'candidate_logprobs')
        
        last_tok_logits = logits[:, -1, :]
        D(last_tok_logits, 'last_tok_logits')

        sorted_logits, sorted_indices = torch.sort(last_tok_logits, descending=True, dim=-1)
        DS(sorted_logits, 'sorted_logits')
        DS(sorted_indices, 'sorted_indices')
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        D(sorted_probs, 'sorted_probs')
        display(sorted_probs.sum(dim=1))
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        D(cum_probs, 'cum_probs')

        # Create tensor of bools indicating which indices are cumulatively less than top_p
        keep_indices = cum_probs < top_p

        # Keep the last element that went over top_p
        keep_indices[:, 1:] = keep_indices[:, :-1].clone() # Is this inefficient?
        keep_indices[:, 0] = 1  # Always keep the first element
        D(keep_indices, 'keep_indices')

        new_candidate_parents = keep_indices.nonzero()[:, 0]
        D(new_candidate_parents, 'new_candidate_parents')

        # OPTIM: Potential optimization -- have a fixed tensor of size (max_candidates, max_tokens) and copy this into that (batch-aware).
        # OPTIM: consider which of these operations can be done in-place to prevent new allocations?
        carryover_candidates = candidates.index_select(0, new_candidate_parents)
        D(carryover_candidates, 'carryover_candidates')

        # Similar code could be used to trace entire origin of sequence. For now since server just traces parent of the preceding generation, not needed
        # carryover_candidate_parents = candidate_parents.index_select(0, carryover_candidate_indices)  # Not strictly necessary since 1d
        # D(carryover_candidate_parents, 'carryover_candidate_parents')

        carryover_candidate_logprobs = candidate_logprobs.index_select(0, new_candidate_parents)  # Not strictly necessary since 1d
        D(carryover_candidate_logprobs, 'carryover_candidate_logprobs')

        new_candidate_toks = sorted_indices[keep_indices].unsqueeze(1)
        D(new_candidate_toks, 'new_candidate_toks')
        new_candidate_tok_logprobs = sorted_probs[keep_indices].log()
        D(new_candidate_tok_logprobs, 'new_candidate_tok_logprobs')

        new_candidates = torch.cat([carryover_candidates, new_candidate_toks], dim=1)
        D(new_candidates, 'new_candidates')
        new_candidate_logprobs = carryover_candidate_logprobs.add_(new_candidate_tok_logprobs)
        D(new_candidate_logprobs, 'new_candidate_logprobs')

        return new_candidates, new_candidate_parents, new_candidate_logprobs


    def _infer(self, candidates, candidate_logprobs):
        with torch.inference_mode():
            num_batches = (candidates.shape[0] + self.batch_size - 1) // self.batch_size  # Round up to nearest whole number of batches
            D(num_batches, 'num_batches')

            check_gpu('infer start')
            output_logits_list = []
            output_embeddings_list = []
            for i in range(0, num_batches, 1):
                batch_candidates = candidates[i * self.batch_size:(i + 1) * self.batch_size]
                DS(batch_candidates, 'batch_candidates')
                batch_candidate_logprobs = candidate_logprobs[i * self.batch_size:(i + 1) * self.batch_size]
                DS(batch_candidate_logprobs, 'batch_candidate_logprobs')

                batch_outputs = self.model(input_ids=batch_candidates, output_hidden_states=True)
                DS(batch_outputs.logits, 'batch_logits')
                DS(batch_outputs.hidden_states[-1], 'hidden_states[-1]')

                output_logits_list.append(batch_outputs.logits)
                output_embeddings_list.append(batch_outputs.hidden_states[-1][:,-1,:])
                check_gpu('infer - after batch run')

            output_logits = torch.cat(output_logits_list, dim=0)
            output_embeddings = torch.cat(output_embeddings_list, dim=0)
            
            return output_logits, output_embeddings