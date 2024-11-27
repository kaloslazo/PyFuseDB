import time
import pandas as pd
from DataLoader import DataLoader
import matplotlib.pyplot as plt

dataset_sizes = [1000, 5000, 10000, 25000, 50000, 75000, 98420]
data_path = "data/afs/spotifySongsTextConcatenated.csv"

test_query = """
    SELECT track_name, track_artist
    FROM songs 
    WHERE to_tsvector('english', texto_concatenado) @@ plainto_tsquery('your birthday');
"""
top_k = 10

data_loader = DataLoader(data_path)

def load_subset(data_loader, num_records):
    data_loader.data = pd.read_csv(data_path, nrows=num_records)
    data_loader._check_existing_index()
    data_loader._load_existing_index()

results = []

for size in dataset_sizes:
    print(f"Testing with {size} records...")

    load_subset(data_loader, size)

    start_time = time.time()
    try:
        custom_results = data_loader.executeQuery(test_query, top_k)
        custom_duration = time.time() - start_time
    except Exception as e:
        custom_results = None
        custom_duration = -1
        print(f"Custom index failed: {e}")

    start_time = time.time()
    try:
        postgres_results = data_loader.executeQueryPostgreSQL(test_query, top_k)
        postgres_duration = time.time() - start_time
    except Exception as e:
        postgres_results = None
        postgres_duration = -1
        print(f"PostgreSQL index failed: {e}")

    results.append({
        "size": size,
        "custom_duration": custom_duration,
        "postgres_duration": postgres_duration,
    })

print("\nPerformance Comparison:")
for result in results:
    print(f"Records: {result['size']} | Custom Index: {result['custom_duration']:.4f}s | PostgreSQL: {result['postgres_duration']:.4f}s")

sizes = [result["size"] for result in results]
custom_durations = [result["custom_duration"] for result in results]
postgres_durations = [result["postgres_duration"] for result in results]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(sizes, custom_durations, label="Custom Inverted Index", marker="o")
plt.plot(sizes, postgres_durations, label="PostgreSQL", marker="s")
plt.title("Performance Comparison (Linear Scale)")
plt.xlabel("Number of Records")
plt.ylabel("Execution Time (s)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(sizes, custom_durations, label="Custom Inverted Index", marker="o")
plt.plot(sizes, postgres_durations, label="PostgreSQL", marker="s")
plt.xscale("log")
plt.yscale("log")
plt.title("Performance Comparison (Logarithmic Scale)")
plt.xlabel("Number of Records (log scale)")
plt.ylabel("Execution Time (log scale)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("result.png")
plt.show()
