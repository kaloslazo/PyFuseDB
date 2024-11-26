import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
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
        self.collection = collection
        p = index.Property()
        p.dimension = collection.shape[1]
        self.index = index.Index(properties=p)

        for i in range(self.collection_size):
            embedding = tuple(collection[i])
            bounding_box = embedding + embedding
            self.index.insert(i, bounding_box)


    def knn_search(self, query, k=5):
        np_query = np.array(query)
        tuple_query = tuple(np_query)

        bounding_box = tuple_query + tuple_query
        nearest_neighbors = list(self.index.nearest(bounding_box, k))

        results = [(i, euclidean(np_query, self.collection[i])) for i in nearest_neighbors]
        return sorted(results, key = lambda x: x[1])


    def range_search(self, query, radius=5):
        np_query = np.array(query)
        tuple_query = tuple(np_query)

        # bounding box al rededeor de la query, de size = radius
        bounding_box = tuple(
            coord - radius if i % 2 == 0 else coord + radius
            for i, coord in enumerate(tuple_query * 2)
        )

        candidates = list(self.index.intersection(bounding_box))

        # filtrar los candidatos con ed > radius
        results = [
            (i, euclidean(np_query, self.collection[i]))
            for i in candidates
            if euclidean(np_query, self.collection[i]) <= radius
        ]

        return sorted(results, key = lambda x: x[1])


class FaissKNN:
    def __init__(self, collection):
        self.collection = collection
        self.collection_size = collection.shape[0]
        self.dim = collection.shape[1]
        # TODO
        # self.index = faiss.???

    def knn_search(self, query, k=5):
        pass

def extract_feature_vector(image_path, model = Model ):
    """Extract a feature vector from an image using a CNN."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array)
    return features.flatten()


def build_vector_collection(image_folder):
    """Build a collection of feature vectors for a folder of images."""
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    collection = []
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for image_path in image_files:
        try:
            feature_vector = extract_feature_vector(image_path, base_model)
            collection.append(feature_vector)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return np.array(collection)
