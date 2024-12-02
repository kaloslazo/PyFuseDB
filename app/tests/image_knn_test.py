import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MultiDim import SequentialKNN, RTreeKNN, FaissKNN

class TestSearchFunctionalityReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Cargar datos reales
        cls.reduced_collection = np.load("./data/imagenette/reduced_vector.npy")
        
        # Tomar una muestra pequeña para las pruebas
        np.random.seed(42)
        sample_size = 100
        indices = np.random.choice(len(cls.reduced_collection), sample_size, replace=False)
        cls.test_data = cls.reduced_collection[indices]
        
        # Query point será el primer vector
        cls.query_point = cls.test_data[0]
        
        # Inicializar métodos de búsqueda
        cls.sequential = SequentialKNN(cls.test_data)
        cls.rtree = RTreeKNN(cls.test_data)
        cls.faiss = FaissKNN(cls.test_data)

    def test_knn_search_real(self):
        """Prueba KNN con datos reales"""
        k = 5
        
        sequential_results = self.sequential.knn_search(self.query_point, k)
        rtree_results = self.rtree.knn_search(self.query_point, k)
        faiss_results = self.faiss.knn_search(self.query_point, k)
        
        # Verificar número de resultados
        self.assertEqual(len(sequential_results), k)
        self.assertEqual(len(rtree_results), k)
        self.assertEqual(len(faiss_results), k)
        
        # El primer resultado debe ser el punto query (distancia = 0)
        self.assertAlmostEqual(sequential_results[0][1], 0, places=5)
        self.assertAlmostEqual(rtree_results[0][1], 0, places=5)
        self.assertAlmostEqual(faiss_results[0][1], 0, places=5)

    def test_range_search_real(self):
        """Prueba range search con datos reales"""
        # Usar radio pequeño para obtener pocos resultados
        radius = 0.1
        
        sequential_results = self.sequential.range_search(self.query_point, radius)
        rtree_results = self.rtree.range_search(self.query_point, radius)
        
        # Verificar que ambos métodos encuentren el mismo número de puntos
        self.assertEqual(len(sequential_results), len(rtree_results))
        
        # Verificar que todas las distancias sean menores al radio
        for _, dist in sequential_results:
            self.assertLessEqual(dist, radius)
        for _, dist in rtree_results:
            self.assertLessEqual(dist, radius)

    def test_consistency_between_methods(self):
        """Verifica consistencia entre diferentes métodos"""
        k = 3
        
        sequential_results = self.sequential.knn_search(self.query_point, k)
        rtree_results = self.rtree.knn_search(self.query_point, k)
        faiss_results = self.faiss.knn_search(self.query_point, k)
        
        # Convertir resultados a sets de índices para comparar
        sequential_indices = {idx for idx, _ in sequential_results}
        rtree_indices = {idx for idx, _ in rtree_results}
        faiss_indices = {idx for idx, _ in faiss_results}
        
        # Los tres métodos deberían encontrar puntos similares
        # Permitimos algunas diferencias debido a empates en distancias
        common_indices = sequential_indices.intersection(rtree_indices).intersection(faiss_indices)
        self.assertGreater(len(common_indices), 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)