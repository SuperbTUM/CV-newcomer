import dask.dataframe as dd
import numpy as np
import jax.numpy as jnp  # The installation does not support on Windows, only on Linux & Mac
import jax.random as jrand
import jaxlib
from collections import Counter


class KNearestNeighbor:
    def __init__(self, k):
        # self.locations = np.random.random()
        key = jrand.PRNGKey(0)
        self.locations = jrand.uniform(key)
        self.labels = ""
        self.k = k

    def load(self, dataset):
        # assuming the dataset is composed of point locations
        # as well as labels
        file = dd.read_csv(dataset, delimiter=',', dtype={"location":jnp.float32, "label":str})
        self.locations = file.loc[1:,0]
        self.labels = file.loc[1:,1]

    @staticmethod
    def euclidean_distance(vector_a, vector_b):
        if len(vector_a) != len(vector_b):
            return
        return jnp.sqrt(jnp.sum((vector_a - vector_b) ** 2))

    def knn(self, instance):
        distances = [self.euclidean_distance(instance, x) for x in self.locations]
        # the above could be down in parallel
        k_neighbors = jnp.argsort(distances[:self.k])
        # get the nearest k points
        vote = Counter(self.labels[k_neighbors])
        return vote.most_common()[0][0]

