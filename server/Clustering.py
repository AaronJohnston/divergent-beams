from typing import List
from sklearn.cluster import AgglomerativeClustering
import numpy as np


class Clustering():
    def __init__(self):
        pass

    def cluster(embeddings: List[List[float]]):
        return AgglomerativeClustering().fit_predict(embeddings)
