# Divergent Beams

Aaron Johnston, 2024

A library for sampling diverse outputs from an LLM's probability space.

## Demo

See demo at https://aaronjohnston.me/projects/divergent-beams

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

## Library

TODO

## Running Demo Locally

## Models

This library operates using a model and tokenizer from the Huggingface Transformers library. Although any model/tokenizer pair can be passed in, it is currently only tested on models similar to Microsoft's Phi-3-mini-4k-instruct.
