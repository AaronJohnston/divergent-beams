import cohere
from Environment import Environment


class Inference():
    def __init__(self, env: Environment):
        self.cohere = cohere.Client(env.cohere_api_key)

    def generate(self, prompt: str, seed: int = 0, max_tokens=256, temperature: float = 0.8):
        response = self.cohere.chat(
            message=prompt, seed=seed, max_tokens=max_tokens, temperature=temperature,
        )
        return response

    def embed(self, texts: list[str]):
        response = self.cohere.embed(
            texts=texts, model="embed-english-v3.0", input_type="classification"
        )
        return response
