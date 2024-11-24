import numpy as np
from rtree import index
import heapq

def euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b)


class SequentialKNN:
    def __init__(self, collection):
        self.collection = collection
        self.collection_size = collection.shape[0]

    def knn_search(self, query, k=5):
        query = np.array(query)
        max_heap = []

        for i in range(self.collection_size):
            dist = euclidean(query, self.collection[i])

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
            dist = euclidean(query, self.collection[i])

            if dist <= radius:
                results.append((i, dist))

        results = sorted(results, key=lambda x: x[1])
        return results

class RTreeKNN:
    def __init__(self, collection):
        self.collection_size = collection.shape[0]
        p = index.Property()
        p.dimension = collection.shape[1]
        self.index = index.Index(properties=p)

        for i in range(self.collection_size):
            embedding = tuple(collection[i])
            bounding_box = embedding + embedding
            self.index.insert(i, bounding_box)


    def knn_search(self, query, k=5):
        query = tuple(query[0])
        bounding_box = query + query
        nearest_neighbors = list(self.index.nearest(bounding_box, k))

        results = [(i, euclidean(np.array(query), np.array(self.index.get(i)))) for i in nearest_neighbors]
        return sorted(results, key = lambda x: x[1])


    def range_search(self, query, radius=5):
        query = tuple(query[0])

        # bounding box al rededeor de la query, de size = radius
        bounding_box = tuple(
            coord - radius if i % 2 == 0 else coord + radius
            for i, coord in enumerate(query * 2)
        )

        candidates = list(self.index.intersection(bounding_box))

        # filtrar los candidatos con ed > radius
        results = [
            (i, euclidean(np.array(query), np.array(self.index.get(i))))
            for i in candidates
            if euclidean(np.array(query), np.array(self.index.get(i))) <= radius
        ]

        return sorted(results, lambda x: x[1])
