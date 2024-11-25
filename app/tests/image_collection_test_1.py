import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MultiDim import SequentialKNN, RTreeKNN

class TestKNNMethods(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset with random feature vectors
        np.random.seed(42)
        self.mock_data = np.random.rand(10, 2048)  # 10 vectors of size 2048
        self.query = self.mock_data[0]  # Use the first vector as query
        self.sequential_knn = SequentialKNN(self.mock_data)
        self.rtree_knn = RTreeKNN(self.mock_data)

    def test_sequential_knn(self):
        results = self.sequential_knn.knn_search(self.query, k=3)
        self.assertEqual(len(results), 3)  # Should return exactly k results

    def test_sequential_range(self):
        results = self.sequential_knn.range_search(self.query, radius=0.5)
        self.assertTrue(all(r[1] <= 0.5 for r in results))  # All distances <= radius

    def test_rtree_knn(self):
        results = self.rtree_knn.knn_search(self.query, k=3)
        self.assertEqual(len(results), 3)  # Should return exactly k results

    def test_rtree_range(self):
        results = self.rtree_knn.range_search(self.query, radius=0.5)
        self.assertTrue(all(r[1] <= 0.5 for r in results))  # All distances <= radius

if __name__ == "__main__":
    unittest.main(verbosity=2)
