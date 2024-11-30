import unittest
import numpy as np
import sys
import os
import logging
import colorlog
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MultiDim import SequentialKNN, RTreeKNN, FaissKNN

def setup_logger():
    """Configure a minimal elegant logger"""
    # Minimal formatting characters
    SEPARATOR = "─" * 50

    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(message)s",
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    # Setup console handler only
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)

    logger = logging.getLogger('KNNTests')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.separator = SEPARATOR

    return logger

class TestKNNMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = setup_logger()
        cls.logger.info("\n" + cls.logger.separator)
        cls.logger.info("KNN Methods Test Suite")
        cls.logger.info(cls.logger.separator)

        np.random.seed(42)
        cls.mock_data = np.load("../data/feature_vector.npy")
        cls.query = cls.mock_data[0]

        try:
            start_time = time.time()
            cls.sequential_knn = SequentialKNN(cls.mock_data)
            elapsed_time = time.time() - start_time
            cls.logger.info(f"SequentialKNN initialized in {elapsed_time:.4f} seconds")

            start_time = time.time()
            cls.rtree_knn = RTreeKNN(cls.mock_data)
            elapsed_time = time.time() - start_time
            cls.logger.info(f"RTreeKNN initialized in {elapsed_time:.4f} seconds")

            start_time = time.time()
            cls.faiss_knn = FaissKNN(cls.mock_data)
            elapsed_time = time.time() - start_time
            cls.logger.info(f"FaissKNN initialized in {elapsed_time:.4f} seconds")

        except Exception as e:
            cls.logger.error(f"! {str(e)}")
            raise

    def test_sequential_knn(self):
        self.logger.info("\n→ Sequential KNN Search")
        start_time = time.time()
        try:
            results = self.sequential_knn.knn_search(self.query, k=3)
            self.assertEqual(
                len(results), 3,
                "Sequential KNN debe retornar exactamente k=3 resultados"
            )
            self.logger.info("✓ Passed")
            self.logger.debug(f"  Results: {results}")
        except Exception as e:
            self.logger.error(f"✗ Failed: {str(e)}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            self.logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")

    def test_sequential_range(self):
        self.logger.info("\n→ Sequential Range Search")
        start_time = time.time()
        try:
            results = self.sequential_knn.range_search(self.query, radius=0.5)
            self.assertTrue(
                all(r[1] <= 0.5 for r in results),
                "Todos los puntos deben estar dentro del radio especificado (0.5)"
            )
            self.logger.info("✓ Passed")
            self.logger.debug(f"  Points found: {len(results)}")
        except Exception as e:
            self.logger.error(f"✗ Failed: {str(e)}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            self.logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")

    def test_rtree_knn(self):
        self.logger.info("\n→ RTree KNN Search")
        start_time = time.time()
        try:
            results = self.rtree_knn.knn_search(self.query, k=3)
            self.assertEqual(
                len(results), 3,
                "RTree KNN debe retornar exactamente k=3 resultados"
            )
            self.logger.info("✓ Passed")
            self.logger.debug(f"  Results: {results}")
        except Exception as e:
            self.logger.error(f"✗ Failed: {str(e)}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            self.logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")

    def test_rtree_range(self):
        self.logger.info("\n→ RTree Range Search")
        start_time = time.time()
        try:
            results = self.rtree_knn.range_search(self.query, radius=0.5)
            self.assertTrue(
                all(r[1] <= 0.5 for r in results),
                "Todos los puntos deben estar dentro del radio especificado (0.5)"
            )
            self.logger.info("✓ Passed")
            self.logger.debug(f"  Points found: {len(results)}")
        except Exception as e:
            self.logger.error(f"✗ Failed: {str(e)}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            self.logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")

    def test_faiss_knn(self):
        self.logger.info("\n→ Faiss KNN Search")
        start_time = time.time()
        try:
            results = self.faiss_knn.knn_search(self.query, k=3)
            self.assertTrue(
                np.array_equal(self.mock_data[int(results[0][0])], self.query),
                "El primer resultado debe ser el punto query (distancia = 0)"
            )
            self.assertEqual(
                len(results), 3,
                "Faiss KNN debe retornar exactamente k=3 resultados"
            )
            self.logger.info("✓ Passed")
            self.logger.debug(f"  First result: {results[0]}")
        except Exception as e:
            self.logger.error(f"✗ Failed: {str(e)}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            self.logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")

    @classmethod
    def tearDownClass(cls):
        cls.logger.info("\n" + cls.logger.separator)
        cls.logger.info("Test Suite Completed")
        cls.logger.info(cls.logger.separator + "\n")

if __name__ == "__main__":
    unittest.main(verbosity=2)
