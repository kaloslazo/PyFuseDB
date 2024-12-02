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
from MultiDim import SequentialKNN, RTreeKNN, FaissKNN

def setup_logger():
    """Configure a minimal elegant logger"""
    SEPARATOR = "─" * 50
    logger = logging.getLogger('KNNTests')
    
    # Limpiar handlers existentes para evitar duplicación
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

        # Cargar datos originales
        np.random.seed(42)
        cls.feature_vector = np.load("./data/imagenette/feature_vector.npy")
        cls.reduced_collection = np.load("./data/imagenette/reduced_vector.npy")
        
        # Definir tamaños a probar
        cls.sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
        cls.results_feature = {size: {} for size in cls.sizes}
        cls.results_reduced = {size: {} for size in cls.sizes}

    def test_methods_with_varying_sizes(self):
        for size in self.sizes:
            self.logger.info(f"\nPruebas con N = {size}")
            
            # Preparar datos
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

            # Pruebas con feature_vector (alta dimensionalidad)
            self.logger.info(f"\nPruebas feature_vector (N={size})")
            
            # Sequential KNN - feature_vector
            sequential = SequentialKNN(feature_subset)
            start_time = time.time()
            results = sequential.knn_search(query_feature, k=3)
            sequential_time = time.time() - start_time
            self.results_feature[size]['Sequential'] = sequential_time

            # RTree KNN - feature_vector
            rtree = RTreeKNN(feature_subset)
            start_time = time.time()
            results = rtree.knn_search(query_feature, k=3)
            rtree_time = time.time() - start_time
            self.results_feature[size]['RTree'] = rtree_time

            # Faiss KNN - feature_vector
            faiss = FaissKNN(feature_subset)
            start_time = time.time()
            results = faiss.knn_search(query_feature, k=3)
            faiss_time = time.time() - start_time
            self.results_feature[size]['Faiss'] = faiss_time

            self.logger.info(f"Sequential: {sequential_time:.4f}s")
            self.logger.info(f"Rtree: {rtree_time:.4f}s")
            self.logger.info(f"Faiss: {faiss_time:.4f}s")

            # Pruebas con reduced_collection (baja dimensionalidad)
            self.logger.info(f"\nPruebas reduced_collection (N={size})")
            
            # Sequential KNN - reduced
            sequential = SequentialKNN(reduced_subset)
            start_time = time.time()
            results = sequential.knn_search(query_reduced, k=3)
            sequential_time = time.time() - start_time
            self.results_reduced[size]['Sequential'] = sequential_time

            # RTree KNN - reduced
            rtree = RTreeKNN(reduced_subset)
            start_time = time.time()
            results = rtree.knn_search(query_reduced, k=3)
            rtree_time = time.time() - start_time
            self.results_reduced[size]['RTree'] = rtree_time

            # Faiss KNN - reduced
            faiss = FaissKNN(reduced_subset)
            start_time = time.time()
            results = faiss.knn_search(query_reduced, k=3)
            faiss_time = time.time() - start_time
            self.results_reduced[size]['Faiss'] = faiss_time

            self.logger.info(f"Sequential: {sequential_time:.4f}s")
            self.logger.info(f"RTree: {rtree_time:.4f}s")
            self.logger.info(f"Faiss: {faiss_time:.4f}s")

    @classmethod
    def tearDownClass(cls):
        # DataFrame para feature_vector
        df_feature = pd.DataFrame(cls.results_feature).T
        df_feature.index.name = 'N'
        
        # DataFrame para reduced_collection
        df_reduced = pd.DataFrame(cls.results_reduced).T
        df_reduced.index.name = 'N'
        
        cls.logger.info("\n" + cls.logger.separator)
        cls.logger.info("Resultados Feature Vector (Alta dimensionalidad):")
        cls.logger.info("\n" + tabulate(df_feature, headers='keys', tablefmt='pretty'))
        cls.logger.info("\nResultados Reduced Collection (Baja dimensionalidad):")
        cls.logger.info("\n" + tabulate(df_reduced, headers='keys', tablefmt='pretty'))
        cls.logger.info(cls.logger.separator + "\n")

        # Guardar resultados
        df_feature.to_csv('knn_benchmark_feature_vector.csv')
        df_reduced.to_csv('knn_benchmark_reduced_collection.csv')

class TestRangeSearchMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = setup_logger()
        cls.logger.info("\n" + cls.logger.separator)
        cls.logger.info("Range Search Test Suite")
        cls.logger.info(cls.logger.separator)

        # Cargar datos originales
        np.random.seed(42)
        cls.feature_vector = np.load("./data/imagenette/feature_vector.npy")
        cls.reduced_collection = np.load("./data/imagenette/reduced_vector.npy")
        
        # Tamaño fijo N=1000 y radios a probar
        cls.size = 1000
        cls.radii = [0.5, 1.0, 2.0, 5.0]
        cls.results_feature = {f"r={r}": {} for r in cls.radii}
        cls.results_reduced = {f"r={r}": {} for r in cls.radii}

        # Preparar subconjuntos de datos
        indices = np.random.choice(len(cls.feature_vector), cls.size, replace=False)
        cls.feature_subset = cls.feature_vector[indices]
        cls.reduced_subset = cls.reduced_collection[indices]
        cls.query_feature = cls.feature_subset[0]
        cls.query_reduced = cls.reduced_subset[0]

    def test_range_search_with_varying_radii(self):
        for radius in self.radii:
            self.logger.info(f"\nPruebas con radio = {radius}")

            # Pruebas con feature_vector
            self.logger.info(f"Feature vector (N={self.size})")
            
            # Sequential Range Search
            sequential = SequentialKNN(self.feature_subset)
            start_time = time.time()
            results = sequential.range_search(self.query_feature, radius=radius)
            sequential_time = time.time() - start_time
            self.results_feature[f"r={radius}"]['Sequential'] = {
                'time': sequential_time,
                'count': len(results)
            }

            # RTree Range Search
            rtree = RTreeKNN(self.feature_subset)
            start_time = time.time()
            results = rtree.range_search(self.query_feature, radius=radius)
            rtree_time = time.time() - start_time
            self.results_feature[f"r={radius}"]['RTree'] = {
                'time': rtree_time,
                'count': len(results)
            }

            self.logger.info(f"Sequential: {sequential_time:.4f}s ({len(results)} resultados)")
            self.logger.info(f"RTree: {rtree_time:.4f}s ({len(results)} resultados)")

            # Pruebas con reduced_collection
            self.logger.info(f"Reduced collection (N={self.size})")
            
            # Sequential Range Search
            sequential = SequentialKNN(self.reduced_subset)
            start_time = time.time()
            results = sequential.range_search(self.query_reduced, radius=radius)
            sequential_time = time.time() - start_time
            self.results_reduced[f"r={radius}"]['Sequential'] = {
                'time': sequential_time,
                'count': len(results)
            }

            # RTree Range Search
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
        # Crear DataFrames para cada tipo de datos
        df_feature = pd.DataFrame({
            radius: {
                f"{method}_time": cls.results_feature[radius][method]['time']
                for method in ['Sequential', 'RTree']
            } for radius in cls.results_feature.keys()
        }).T
        df_feature.index.name = 'Radio'
        
        df_reduced = pd.DataFrame({
            radius: {
                f"{method}_time": cls.results_reduced[radius][method]['time']
                for method in ['Sequential', 'RTree']
            } for radius in cls.results_reduced.keys()
        }).T
        df_reduced.index.name = 'Radio'

        cls.logger.info("\n" + cls.logger.separator)
        cls.logger.info(f"Resultados Feature Vector (N={cls.size}):")
        cls.logger.info("\n" + tabulate(df_feature, headers='keys', tablefmt='pretty'))
        cls.logger.info(f"\nResultados Reduced Collection (N={cls.size}):")
        cls.logger.info("\n" + tabulate(df_reduced, headers='keys', tablefmt='pretty'))
        
        # Guardar resultados
        df_feature.to_csv('range_search_feature_vector.csv')
        df_reduced.to_csv('range_search_reduced_collection.csv')


if __name__ == "__main__":
    unittest.main(verbosity=2)