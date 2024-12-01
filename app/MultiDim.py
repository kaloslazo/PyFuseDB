import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


from rtree import index
import heapq
import faiss


# --------------------------------
# File Paths
# --------------------------------

feature_vector_path = "data/imagenette/feature_vector.npy"
reduced_vector_path = "data/imagenette/reduced_vector.npy"
reducer_path = "data/imagenette/reducer.pkl"
filenames_path = "data/imagenette/filenames.npy"
image_folder = "data/imagenette/images"

# --------------------------------
# Model and Preprocessing
# --------------------------------

base_model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(base_model.children())[:-1]).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def distance_metric(a: np.ndarray, b: np.ndarray):
    return np.linalg.norm(a - b)


class SequentialKNN:
    def __init__(self, collection):
        self.collection = collection
        self.collection_size = collection.shape[0]

    def knn_search(self, query, k=5):
        query = np.array(query)
        max_heap = []

        for i in range(self.collection_size):
            dist = distance_metric(query, self.collection[i])

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
            dist = distance_metric(query, self.collection[i])

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

        results = [(i, distance_metric(np_query, self.collection[i])) for i in nearest_neighbors]
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
            (i, distance_metric(np_query, self.collection[i]))
            for i in candidates
            if distance_metric(np_query, self.collection[i]) <= radius
        ]

        return sorted(results, key = lambda x: x[1])


class FaissKNN:
    def __init__(self, collection):
        self.collection = collection
        self.collection_size = collection.shape[0]
        self.dim = collection.shape[1]
        self.index = faiss.IndexHNSWFlat(self.dim, 32)
        self.index.add(self.collection)

    def knn_search(self, query, k=5):
        if query.shape[0] != self.dim:
            raise ValueError(f"Dimensión del query ({query.shape[0]}) no coincide con la colección ({self.dim})")
        query = query.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query, k)
        return [(int(i), float(d)) for i, d in zip(indices[0], distances[0])]



# --------------------------------
# Feature Extraction
# --------------------------------

def extract_feature_vector(image_path):
    """
    Extract feature vector from an image using ResNet50.
    Args:
        image_path (str): Path to the image.
    Returns:
        torch.Tensor: Feature vector.
    """
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        embedding = model(input_tensor).squeeze(0).flatten()
        normalized_embedding = embedding / torch.norm(embedding)
    return normalized_embedding.numpy()


def build_vector_collection(image_folder, output_file, filename_list_file):
    """
    Build and save feature vectors for all images in a folder.
    Args:
        image_folder (str): Path to the folder containing images.
        output_file (str): Path to save the feature matrix.
        filename_list_file (str): Path to save the list of filenames.
    """
    print("Building vector collection!")
    collection = []
    filenames = []
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Extract features from images
    for image_path in image_files:
        try:
            feature_vector = extract_feature_vector(image_path)
            collection.append(feature_vector)
            filenames.append(os.path.basename(image_path))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    feature_matrix = np.array(collection)
    np.save(output_file, feature_matrix)
    np.save(filename_list_file, filenames)
    print(f"Feature matrix saved to {output_file}")
    print(f"Filename list saved to {filename_list_file}")


def reduce_collection_dimensionality(collection, reducer):
    """
    Reduce dimensionality of collection using reducer.
    Args:
        collection (np.ndarray): Embedding matrix.
        reducer (umap / pca): dimensionality reducer with .fit_transform() method
    """
    print("Generating reduced dimensionality collection")
    reduced_collection = reducer.fit_transform(collection)
    return reduced_collection


def reduce_query_dimensionality(query_embedding, reducer):
    """
    Reduce dimensionality of query embedding.
    Args:
        query_embedding (np.ndarray): Query embedding vector.
        reducer (umap / pca): dimensionality reducer with .transform() method
    """
    reduced_embedding = reducer.transform(query_embedding.reshape(-1, 1))
    return reduced_embedding[0]


def load_reducer():
    """
    Load dimensionality reducer from file.
    """
    try:
        with open(reducer_path, "rb") as f:
            reducer = pickle.load(f)
        return reducer
    except Exception as e:
        print(f"Error loading dimensionality reducer: {e}")
        return None


# --------------------------------
# Visualization and Debugging
# --------------------------------

def visualize_embeddings(collection, filenames):
    """
    Visualize embeddings using t-SNE.
    Args:
        collection (np.ndarray): Embedding matrix.
        filenames (list): List of filenames corresponding to embeddings.
    """
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(collection)

    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    plt.title("t-SNE Visualization of Embeddings")
    plt.show()


def show_query_and_neighbors(query_image_path, neighbor_paths):
    """
    Visualize query image and its nearest neighbors.
    Args:
        query_image_path (str): Path to the query image.
        neighbor_paths (list): List of paths to nearest neighbor images.
    """
    plt.figure(figsize=(10, 5))

    # Query Image
    plt.subplot(1, len(neighbor_paths) + 1, 1)
    plt.imshow(Image.open(query_image_path))
    plt.title("Query")
    plt.axis("off")

    # Nearest Neighbors
    for i, neighbor_path in enumerate(neighbor_paths):
        plt.subplot(1, len(neighbor_paths) + 1, i + 2)
        plt.imshow(Image.open(neighbor_path))
        plt.title(f"Neighbor {i+1}")
        plt.axis("off")

    plt.show()


# --------------------------------
# Main Function
# --------------------------------

def main():
    if not os.path.exists(feature_vector_path) or not os.path.exists(filenames_path):
        build_vector_collection(image_folder, feature_vector_path, filenames_path)

    collection = np.load(feature_vector_path)
    filenames = np.load(filenames_path)

    if not os.path.exists(reduced_vector_path) or not os.path.exists(filenames_path):
        reducer = UMAP(n_components=128)
        reduced_collection = reduce_collection_dimensionality(collection, reducer)
        np.save(reduced_vector_path, reduced_collection)
        with open(reducer_path, "wb") as f:
            pickle.dump(reducer, f)


    print(f"Collection shape: {collection.shape}")
    print(f"Number of filenames: {len(filenames)}")

    # visualize_embeddings(collection, filenames)

    reduced_collection = np.load(reduced_vector_path)
    reducer = load_reducer()

    # TESTING
    db = RTreeKNN(reduced_collection[:-15])
    print("k-NN database built.")

    for i in range(1, 16):
        query_embedding = collection[-i]
        query_embedding = reduce_query_dimensionality(query_embedding, reducer)
        query_filename = filenames[-i]
        results = db.knn_search(query_embedding, k=5)

        neighbor_filenames = [filenames[idx] for idx, _ in results]
        print(f"Query: {query_filename}")
        print(f"Nearest Neighbors: {neighbor_filenames}")

        show_query_and_neighbors(os.path.join(image_folder, query_filename),
                                 [os.path.join(image_folder, fname) for fname in neighbor_filenames])


if __name__ == '__main__':
    main()
