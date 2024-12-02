import unittest
import numpy as np
import sys
import os
import logging
import colorlog
import time
import pandas as pd
from tabulate import tabulate


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MultiDim import SequentialKNN, RTreeKNN, FaissKNN, load_collection

def setup_logger():
    """Configure a minimal elegant logger"""
    SEPARATOR = "─" * 50
    logger = logging.getLogger('KNNTests')
    
    if logger.handlers:
        logger.handlers.clear()
        
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.separator = SEPARATOR
    
    return logger

class TestSearchMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = setup_logger()
        cls.logger.info("\n" + cls.logger.separator)
        cls.logger.info("Search Test Suite")
        cls.logger.info(cls.logger.separator)

        np.random.seed(42)
        cls.feature_vector = load_collection(False)
        cls.reduced_collection = load_collection(True)
        
        cls.sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
        cls.results_feature = {size: {} for size in cls.sizes}
        cls.results_reduced = {size: {} for size in cls.sizes}

        pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))

    def test_methods_with_varying_sizes(self):
        for size in self.sizes:
            self.logger.info(f"\nPruebas con N = {size}")
            
            if size <= len(self.feature_vector):
                indices = np.random.choice(len(self.feature_vector), size, replace=False)
                feature_subset = self.feature_vector[indices]
                reduced_subset = self.reduced_collection[indices]
            else:
                repeats = size // len(self.feature_vector) + 1
                feature_subset = np.tile(self.feature_vector, (repeats, 1))[:size]
                reduced_subset = np.tile(self.reduced_collection, (repeats, 1))[:size]
                feature_subset += np.random.normal(0, 0.01, feature_subset.shape)
                reduced_subset += np.random.normal(0, 0.01, reduced_subset.shape)

            query_feature = feature_subset[0]
            query_reduced = reduced_subset[0]

            self.logger.info(f"\nPruebas feature_vector (N={size})")
            
            sequential = SequentialKNN(feature_subset)
            start_time = time.time()
            results = sequential.knn_search(query_feature, k=8)
            sequential_time = time.time() - start_time
            self.results_feature[size]['Sequential'] = sequential_time

            faiss = FaissKNN(feature_subset)
            start_time = time.time()
            results = faiss.knn_search(query_feature, k=8)
            faiss_time = time.time() - start_time
            self.results_feature[size]['Faiss'] = faiss_time

            self.logger.info(f"Sequential: {sequential_time:.4f}s")
            self.logger.info(f"Faiss: {faiss_time:.4f}s")

            self.logger.info(f"\nPruebas reduced_collection (N={size})")
            
            sequential = SequentialKNN(reduced_subset)
            start_time = time.time()
            results = sequential.knn_search(query_reduced, k=8)
            sequential_time = time.time() - start_time
            self.results_reduced[size]['Sequential'] = sequential_time

            rtree = RTreeKNN(reduced_subset)
            start_time = time.time()
            results = rtree.knn_search(query_reduced, k=8)
            rtree_time = time.time() - start_time
            self.results_reduced[size]['RTree'] = rtree_time

            faiss = FaissKNN(reduced_subset)
            start_time = time.time()
            results = faiss.knn_search(query_reduced, k=8)
            faiss_time = time.time() - start_time
            self.results_reduced[size]['Faiss'] = faiss_time

            self.logger.info(f"Sequential: {sequential_time:.4f}s")
            self.logger.info(f"RTree: {rtree_time:.4f}s")
            self.logger.info(f"Faiss: {faiss_time:.4f}s")

    @classmethod
    def tearDownClass(cls):
        df_feature = pd.DataFrame({size: {k: f"{v:.4f}" for k, v in values.items()}
                                 for size, values in cls.results_feature.items()}).T
        df_feature.index.name = 'N'
        
        df_reduced = pd.DataFrame({size: {k: f"{v:.4f}" for k, v in values.items()}
                                 for size, values in cls.results_reduced.items()}).T
        df_reduced.index.name = 'N'
        
        # Convertir strings a float para mantener formato
        for df in [df_feature, df_reduced]:
            for col in df.columns:
                df[col] = df[col].astype(float)
        
        cls.logger.info("\n" + cls.logger.separator)
        cls.logger.info("Resultados Feature Vector (Alta dimensionalidad):")
        cls.logger.info("\n" + tabulate(df_feature, headers='keys', tablefmt='pretty', 
                                      floatfmt='.4f', numalign='right'))
        cls.logger.info("\nResultados Reduced Collection (Baja dimensionalidad):")
        cls.logger.info("\n" + tabulate(df_reduced, headers='keys', tablefmt='pretty', 
                                      floatfmt='.4f', numalign='right'))
        cls.logger.info(cls.logger.separator + "\n")

        df_feature.to_csv('knn_benchmark_feature_vector.csv', float_format='%.4f')
        df_reduced.to_csv('knn_benchmark_reduced_collection.csv', float_format='%.4f')

class TestRangeSearchMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = setup_logger()
        cls.logger.info("\n" + cls.logger.separator)
        cls.logger.info("Range Search Test Suite")
        cls.logger.info(cls.logger.separator)

        np.random.seed(42)

        cls.feature_vector = load_collection(False)
        cls.reduced_collection = load_collection(True)
        
        cls.size = 1000
        cls.radii = [0.5, 1.0, 2.0, 5.0]
        cls.results_reduced = {f"r={r}": {} for r in cls.radii}

        indices = np.random.choice(len(cls.feature_vector), cls.size, replace=False)
        cls.reduced_subset = cls.reduced_collection[indices]
        cls.query_reduced = cls.reduced_subset[0]

    def test_range_search_with_varying_radii(self):
        for radius in self.radii:
            self.logger.info(f"\nPruebas con radio = {radius}")
            self.logger.info(f"Reduced collection (N={self.size})")
            
            sequential = SequentialKNN(self.reduced_subset)
            start_time = time.time()
            results = sequential.range_search(self.query_reduced, radius=radius)
            sequential_time = time.time() - start_time
            self.results_reduced[f"r={radius}"]['Sequential'] = {
                'time': sequential_time,
                'count': len(results)
            }

            rtree = RTreeKNN(self.reduced_subset)
            start_time = time.time()
            results = rtree.range_search(self.query_reduced, radius=radius)
            rtree_time = time.time() - start_time
            self.results_reduced[f"r={radius}"]['RTree'] = {
                'time': rtree_time,
                'count': len(results)
            }

            self.logger.info(f"Sequential: {sequential_time:.4f}s ({len(results)} resultados)")
            self.logger.info(f"RTree: {rtree_time:.4f}s ({len(results)} resultados)")

    @classmethod
    def tearDownClass(cls):
        df_reduced = pd.DataFrame({
            radius: {
                f"{method}_time": f"{cls.results_reduced[radius][method]['time']:.4f}"
                for method in ['Sequential', 'RTree']
            } for radius in cls.results_reduced.keys()
        }).T
        df_reduced.index.name = 'Radio'

        # Convertir strings a float para mantener formato
        for col in df_reduced.columns:
            df_reduced[col] = df_reduced[col].astype(float)

        cls.logger.info("\n" + cls.logger.separator)
        cls.logger.info(f"\nResultados Reduced Collection (N={cls.size}):")
        cls.logger.info("\n" + tabulate(df_reduced, headers='keys', tablefmt='pretty', 
                                      floatfmt='.4f', numalign='right'))
        
        df_reduced.to_csv('range_search_reduced_collection.csv', float_format='%.4f')

if __name__ == "__main__":
    unittest.main(verbosity=2)