import cohere
from Environment import Environment


class Inference():
    def __init__(self, env: Environment):
        self.cohere = cohere.Client(env.cohere_api_key)

    def embed(self, texts: list[str]):
        response = self.cohere.embed(
            texts=texts, model="embed-english-v3.0", input_type="classification"
        )
        return response
