from typing import List
from sklearn.cluster import AgglomerativeClustering
import numpy as np


class Clustering():
    def __init__(self):
        pass

    def cluster(self, embeddings: List[List[float]]):
        return AgglomerativeClustering().fit_predict(embeddings)
