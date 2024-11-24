import numpy as np
from rtree import index
import heapq


class SequentialKNN:
    def __init__(self, collection):
        self.collection = collection
        self.collection_size = collection.shape[0]

    def knn_search(self, query, k=5):
        query = np.array(query)
        max_heap = []

        for i in range(self.collection_size):
            dist = np.linalg.norm(query - self.collection[i])

            if len(max_heap) < k:
                heapq.heappush(max_heap, (-dist, i))
            elif -dist > max_heap[0][0]:
                heapq.heappop(max_heap)
                heapq.heappush(max_heap, (-dist, i))

        top_k = [(idx, -dist) for dist, idx in sorted(max_heap, key=lambda x: -x[0])]
        return top_k


    def range_search(self, query, radius=5):
        query = np.array(query)
        results = []

        for i in range(self.collection_size):
            dist = np.linalg.norm(query - self.collection[i])

            if dist <= radius:
                results.append((i, dist))

        results = sorted(results, key=lambda x: x[1])
        return results

