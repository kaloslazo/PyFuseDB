import unittest
import os
import shutil
import sys

# Ajustar el path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from InvertedIndex import InvertedIndex

class InvertedIndexTest(unittest.TestCase):
    bin_path = os.path.join("app", "data", "bin")

    def setUp(self):
        """Se ejecuta antes de cada prueba"""
        # Crear directorio bin si no existe
        if not os.path.exists(self.bin_path):
            os.makedirs(self.bin_path)
        
        # Limpiar directorio bin
        self.clean_bin_directory()
        
        self.index = InvertedIndex(block_size=2, dict_size=3)
        self.test_docs = [
            "The quick brown fox jumps over the lazy dog",
            "A quick brown dog jumps over the lazy fox",
            "The lazy fox sleeps",
            "Quick brown brown fox fox jumps"
        ]

    def tearDown(self):
        """Se ejecuta después de cada prueba"""
        self.clean_bin_directory()

    def clean_bin_directory(self):
        """Limpia todos los archivos .bin del directorio"""
        if os.path.exists(self.bin_path):
            for file in os.listdir(self.bin_path):
                if file.endswith('.bin'):
                    os.remove(os.path.join(self.bin_path, file))

    def test_build_index(self):
        """Test básico de construcción del índice"""
        self.index.build_index(self.test_docs)
        self.assertTrue(os.path.exists(os.path.join(self.index.bin_path, "block_0.bin")))
        self.assertTrue(os.path.exists(os.path.join(self.index.bin_path, "dict_0.bin")))

    def test_search_single_term(self):
        """Test de búsqueda de un solo término"""
        self.index.build_index(self.test_docs)
        results = self.index.search("fox")
        self.assertTrue(len(results) > 0)
        scores = [score for doc_id, score in results]
        self.assertTrue(scores[0] >= scores[-1])

    def test_search_multiple_terms(self):
        """Test de búsqueda con múltiples términos"""
        self.index.build_index(self.test_docs)
        results = self.index.search("quick brown")
        self.assertTrue(len(results) > 0)
        doc_ids = {result[0] for result in results[:3]}
        self.assertTrue(all(id in doc_ids for id in [0, 1, 3]))

    def test_search_nonexistent_term(self):
        """Test de búsqueda de término inexistente"""
        self.index.build_index(self.test_docs)
        results = self.index.search("unknownterm")
        self.assertEqual(len(results), 0)

    def test_empty_search(self):
        """Test de búsqueda vacía"""
        self.index.build_index(self.test_docs)
        results = self.index.search("")
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)