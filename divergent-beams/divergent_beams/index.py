import numpy as np
import torch.nn.functional as F
import torch
import time
import json
from sklearn.cluster import KMeans
from torchmetrics.functional import pairwise_cosine_similarity

class DivergentBeams:
    def __init__(self, model, tokenizer, eos_token_id=None, batch_size=8):
        """Initialize a DivergentBeams object with a model and tokenizer.

        Args:
            model (transformers.PreTrainedModel): The model to use for generation.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for generation.
            eos_token_id (int, optional): The ID of the EOS token. If None, the EOS token ID from the tokenizer will be used.
            batch_size (int, optional): The batch size to use for generation. Defaults to 8.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.batch_size = batch_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        
    def generator(self, top_p: float, top_p_decay: float, top_k: float, max_beams: int, max_new_tokens: int, gather_algo: str, prompt: str):
        """
        Generate candidates using the divergent beams algorithm. See http://aaronjohnston.me/divergent-beams/ for more information.
        
        Args:
            top_p (float): The nucleus sampling probability. A value of 0.9 means that the tokens representing at least the top 90% of probability mass will be sampled. With top_k, the most restrictive limit is used (i.e. the smallest number of tokens will be sampled.)
            top_p_decay (float): The decay factor for the nucleus sampling probability. A value of 1.0 means no decay. A value of 0.99 means the top_p value will decay by 1% after each token step.
            top_k (int): The number of top tokens to consider at each step. A value of 3 means at most 3 tokens will be sampled. With top_p, the most restrictive limit is used (i.e. the smallest number of tokens will be sampled.)
            max_beams (int): The maximum number of beams to keep after each gather step.
            max_new_tokens (int): The maximum number of new tokens to generate.
            gather_algo (str): The algorithm to use for gathering beams. Must be 'k_means' or 'farthest_neighbors'.
            prompt (str): The prompt to generate candidates from.
            
        Yields:
            str: A JSON string representing the candidates at each gather and sample step. The JSON string will be formatted as follows:
                {
                    "id": str,  # The ID of the step, with a suffix denoting the type: -f for farthest_neighbors (gather),  -k for k_means (gather), or -p for top_p (sample)
                    "level_type": str,  # The type of the step, either 'gather' or 'sample'
                    "duration": float,  # The duration of the step in seconds. Because gather and sample steps share a single forward pass of the model for efficiency, that forward pass duration is included in the first step to run for that token (gather if there was one, otherwise sample).
                    "nodes": [  # The list of active beams at this step.
                        {
                            "content": str,  # The last token that was generated for the beam.
                            "parent": int,  # The index in the nodes list of the previous step's output representing the parent of this beam.
                            "aunts": [int],  # A list of indices in the nodes list of the previous step's output representing beams whose probability mass was consolidated into this one (only applies during a gather step -- will be empty for a sample step).
                            "prob": float  # The joint log probability of this beam.
                        }
                    ]
                }

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the finished candidate sequences (Tensor of size NUM_CANDIDATES, SEQUENCE_LENGTH) and their log probabilities.
        """
        candidates, candidate_logprobs, prompt_len = self._init_candidates(prompt)
        all_finished = []
        all_finished_logprobs = []
        
        for level_idx in range(max_new_tokens):
            start = time.perf_counter()
            logits, embeddings = self._infer(candidates)
            
            if candidates.shape[0] > max_beams:
                if gather_algo == 'k_means':
                    candidates, candidate_parents, candidate_aunts, candidate_logprobs, logits = self._k_means(logits, embeddings, candidates, candidate_logprobs, max_beams)
                    inference_duration = time.perf_counter() - start
                    start = time.perf_counter() # Reset timing for top_p
                    yield self._format_gather(level_idx, 'k', candidates, candidate_parents, candidate_aunts, candidate_logprobs, inference_duration)

                elif gather_algo == 'farthest_neighbors':
                    candidates, candidate_parents, candidate_aunts, candidate_logprobs, logits = self._farthest_neighbors(logits, embeddings, candidates, candidate_logprobs, max_beams)
                    inference_duration = time.perf_counter() - start
                    start = time.perf_counter() # Reset timing for top_p
                    yield self._format_gather(level_idx, 'f', candidates, candidate_parents, candidate_aunts, candidate_logprobs, inference_duration)
            
            candidates, candidate_parents, candidate_logprobs = self._top_p(logits, candidates, candidate_logprobs, top_p, top_k)
            inference_duration = time.perf_counter() - start
            candidates, candidate_parents, candidate_logprobs, max_beams, finished, finished_parents, finished_logprobs = self._select_finished(candidates, candidate_parents, candidate_logprobs, max_beams)
            if finished.shape[0] > 0:
                all_finished.extend(finished)
                all_finished_logprobs.extend(finished_logprobs)
            yield self._format_top_p(level_idx, candidates, candidate_parents, candidate_logprobs, inference_duration, finished, finished_parents, finished_logprobs, prompt_len)
            top_p *= top_p_decay
            
            if candidates.shape[0] == 0:
                break

        yield f"event: message\nid: END\ndata: []\n\n"
        return all_finished, all_finished_logprobs
    

    def _format_gather(self, suffix, level_idx, candidates, candidate_parents, candidate_aunts, candidate_logprobs, duration):
        candidate_texts = self.tokenizer.convert_ids_to_tokens(candidates[:, -1], skip_special_tokens=True)
        candidate_dicts = []
        idx = f"{level_idx}-{suffix}"
        for i in range(len(candidate_texts)):
            candidate_dicts.append({'content': candidate_texts[i], 'parent': candidate_parents[i], 'aunts': candidate_aunts[i], 'prob': candidate_logprobs[i].item()})
        data = json.dumps({'id': idx, 'level_type': 'gather', 'duration': duration, 'nodes': candidate_dicts})
        return f"event: message\nid: {idx}\"\ndata: {data}\n\n"

    def _format_top_p(self, level_idx, candidates, candidate_parents, candidate_logprobs, duration, finished, finished_parents, finished_logprobs, prompt_len):
        candidate_texts = self.tokenizer.convert_ids_to_tokens(candidates[:, -1])
        candidate_dicts = []
        idx = f"{level_idx}-p"
        for i in range(len(candidate_texts)):
            candidate_dicts.append({'content': candidate_texts[i], 'parent': candidate_parents[i], 'prob': candidate_logprobs[i].item()})
        
        finished_texts = self.tokenizer.batch_decode(finished[:,prompt_len:], skip_special_tokens=False)
        finished_parents = finished_parents.tolist()
        finished_dicts = []
        for i in range(len(finished_texts)):
            finished_dicts.append({'content': finished_texts[i], 'parent': finished_parents[i], 'prob': finished_logprobs[i].item()})
        
        data = json.dumps({'id': idx, 'level_type': 'sample', 'duration': duration, 'nodes': candidate_dicts, 'finished': finished_dicts})
        return f"event: message\nid: {idx}\ndata: {data}\n\n"


    def _init_candidates(self, text: str):
        prompt = "<|user|>\n{} <|end|>\n<|assistant|>".format(text)
        inputs = self.tokenizer(prompt, return_tensors='pt')

        candidates = inputs.input_ids.to(self.device)
        candidate_logprobs = torch.zeros((1), dtype=torch.float32, device=self.device)

        return candidates, candidate_logprobs, inputs.input_ids.shape[1]

    def _k_means(self, logits, embeddings, candidates, candidate_logprobs, max_beams):
        # === CPU ===
        embeddings_np = embeddings.float().numpy(force=True)
        k_means = KMeans(n_clusters=min(max_beams, embeddings_np.shape[0]), random_state=0, n_init="auto")
        k_mean_space = k_means.fit_transform(embeddings_np)
        k_mean_clusters = k_means.predict(embeddings_np)
        k_mean_logprob_mass = np.log(np.bincount(k_mean_clusters, weights=candidate_logprobs.cpu().exp()))
        closest = np.argmin(k_mean_space, axis=0)
        # === END CPU ===
        
        closest_indices = torch.from_numpy(closest).to(self.device)
        new_candidates = candidates.index_select(0, closest_indices)
        new_candidate_parents = closest_indices.tolist()
        new_candidate_aunts = [torch.nonzero(torch.from_numpy(k_mean_clusters).to(self.device) == i).squeeze(-1).tolist() for i in range(new_candidates.shape[0])]
        new_candidate_logprobs = torch.from_numpy(k_mean_logprob_mass).to(self.device)
        new_candidate_logits = logits.index_select(0, closest_indices)
        
        return new_candidates, new_candidate_parents, new_candidate_aunts, new_candidate_logprobs, new_candidate_logits
        
    def _farthest_neighbors(self, logits, embeddings, candidates, candidate_logprobs, max_beams):        
        selected = torch.zeros((candidates.shape[0],), dtype=torch.bool).to(self.device)
        max_prob_idx = candidate_logprobs.argmax()
        selected[max_prob_idx] = 1
        
        
        for _ in range(min(max_beams - 1, candidates.shape[0])):
            selected_embeddings = embeddings[selected]
            # Add 2 because bfloat16 on cuda can have imprecision and we need 0 to be lower than every
            # cosine distance
            distances = torch.add(2, pairwise_cosine_similarity(embeddings, selected_embeddings), alpha=-1)
            min_distances = torch.min(distances, dim=1).values
            min_remaining_distances = min_distances * ~selected
            next_selected = min_remaining_distances.argmax(dim=0)
            selected[next_selected] = 1
            
        # We have all the candidates that are selected to move forward. Figure out which probability mass
        # to assign where.
        selected_embeddings = embeddings[selected]
        # Add 2 because bfloat16 on cuda can have imprecision and we need 0 to be lower than every
        # cosine distance
        distances = torch.add(2, pairwise_cosine_similarity(embeddings, selected_embeddings), alpha=-1)
        
        closest_per_candidate = distances.argmin(dim=1)
        
        new_candidates = candidates[selected]
        new_candidate_parents = torch.arange(candidates.shape[0]).to(self.device)[selected].tolist()
        new_candidate_aunts = [list(filter(lambda x: x != i, torch.nonzero(closest_per_candidate == i).squeeze(-1).tolist())) \
                       for i in range(new_candidates.shape[0])]
        new_candidate_logprobs = torch.zeros((new_candidates.shape[0],)).to(self.device)
        new_candidate_logprobs.index_add_(0, closest_per_candidate, candidate_logprobs)
        new_candidate_logits = logits[selected]
        
        return new_candidates, new_candidate_parents, new_candidate_aunts, new_candidate_logprobs, new_candidate_logits

    
    def _select_finished(self, candidates, candidate_parents, candidate_logprobs, max_beams):
        finished_mask = candidates[:,-1] == self.eos_token_id
        unfinished_mask = ~finished_mask
        
        new_candidates = candidates[unfinished_mask]
        if finished_mask.sum() == 0:
            new_candidate_parents = candidate_parents
        else:
            new_candidate_parents = []
            for i in range(len(candidate_parents)):
                if unfinished_mask[i]:
                    new_candidate_parents.append(candidate_parents[i])
        new_candidate_logprobs = candidate_logprobs[unfinished_mask]
        new_max_beams = max_beams - finished_mask.sum()
        
        finished = candidates[finished_mask][:,:-1] # Remove the EOS token
        finished_parents = torch.arange(candidates.shape[0], device=self.device)[finished_mask]
        finished_logprobs = candidate_logprobs[finished_mask]
        
        return new_candidates, new_candidate_parents, new_candidate_logprobs, new_max_beams, finished, finished_parents, finished_logprobs
    

    def _top_p(self, logits, candidates, candidate_logprobs, top_p, top_k):
        last_tok_logits = logits[:, -1, :]

        sorted_logits, sorted_indices = torch.sort(last_tok_logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create tensor of bools indicating which indices are cumulatively less than top_p
        keep_indices = cum_probs < top_p

        # Keep the last element that went over top_p
        keep_indices[:, 1:] = keep_indices[:, :-1].clone() # Is this inefficient?
        keep_indices[:, 0] = 1  # Always keep the first element

        # Don't keep any indices that are greater than top_k
        keep_indices[:, top_k:] = 0

        new_candidate_parents = keep_indices.nonzero()[:, 0]

        carryover_candidates = candidates.index_select(0, new_candidate_parents)
        carryover_candidate_logprobs = candidate_logprobs.index_select(0, new_candidate_parents)  # Not strictly necessary since 1d

        new_candidate_toks = sorted_indices[keep_indices].unsqueeze(1)
        new_candidate_tok_logprobs = sorted_probs[keep_indices].log()

        new_candidates = torch.cat([carryover_candidates, new_candidate_toks], dim=1)
        new_candidate_logprobs = carryover_candidate_logprobs.add_(new_candidate_tok_logprobs)

        return new_candidates, new_candidate_parents.tolist(), new_candidate_logprobs


    def _infer(self, candidates):
        with torch.inference_mode():
            num_batches = (candidates.shape[0] + self.batch_size - 1) // self.batch_size  # Round up to nearest whole number of batches

            output_logits_list = []
            output_embeddings_list = []
            for i in range(0, num_batches, 1):
                batch_candidates = candidates[i * self.batch_size:(i + 1) * self.batch_size]

                batch_outputs = self.model(input_ids=batch_candidates, output_hidden_states=True)

                output_logits_list.append(batch_outputs.logits)
                output_embeddings_list.append(batch_outputs.hidden_states[-1][:,-1,:])

            output_logits = torch.cat(output_logits_list, dim=0)
            output_embeddings = torch.cat(output_embeddings_list, dim=0)
            
            return output_logits, output_embeddings