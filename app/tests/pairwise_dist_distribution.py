import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt


def pairwise_distance_analysis_sampled(
    collection,
    num_pairs=10000,
    metric="euclidean",
    plot=True,
    outlier_threshold=None,
    plot_title=None,
    plot_limits=None,
):

    print(f"Sampling {num_pairs} random pairs from the dataset...")

    # Randomly sample pairs
    n_samples = collection.shape[0]
    random_pairs = [
        (random.randint(0, n_samples - 1), random.randint(0, n_samples - 1))
        for _ in range(num_pairs)
    ]

    distances = []
    for i, j in random_pairs:
        if metric == "euclidean":
            dist = np.linalg.norm(collection[i] - collection[j])
        elif metric == "cosine":
            dist = 1 - np.dot(collection[i], collection[j]) / (
                np.linalg.norm(collection[i]) * np.linalg.norm(collection[j])
            )
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        distances.append(dist)

    print(f"Computed distances for {num_pairs} pairs.")

    # Plot histogram
    if plot:
        sns.histplot(distances, bins=50, kde=True)
        if plot_title:
            plt.title(plot_title)
        if plot_limits:
            x_limits, y_limits = plot_limits
            plt.xlim(x_limits)
            plt.ylim(y_limits)
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Frequency")
        plt.show()

    # Outlier detection
    outliers = []
    if outlier_threshold:
        print(f"Identifying outliers with distances > {outlier_threshold}...")
        outliers = [
            (i, j, d)
            for (i, j), d in zip(random_pairs, distances)
            if d > outlier_threshold
        ]
        print(f"Found {len(outliers)} outlier pairs.")

    return {"distances": distances, "outliers": outliers}


# Example Usage
def main():
    # Load embeddings
    feature_vector_path = "data/imagenette/feature_vector.npy"
    reduced_vector_path = "data/imagenette/reduced_vector.npy"
    collection = np.load(feature_vector_path)
    reduced_collection = np.load(reduced_vector_path)

    # Perform pairwise distance analysis with sampling
    pairwise_distance_analysis_sampled(
        collection,
        num_pairs=10000,
        metric="euclidean",
        plot=True,
        plot_title="10000 Random Pairs ED distribution for 2048 dims",
        plot_limits=((0, 10), (0, 1000))
    )

    pairwise_distance_analysis_sampled(
        reduced_collection,
        num_pairs=10000,
        metric="euclidean",
        plot=True,
        plot_title="10000 Random Pairs ED distribution for 128 dims",
        plot_limits=((0, 10), (0, 1000))
    )


if __name__ == "__main__":
    main()
