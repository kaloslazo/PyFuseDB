import unittest
import os
import sys
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from InvertedIndex import InvertedIndex

class InvertedIndexTest(unittest.TestCase):
    bin_path = os.path.join("app", "data", "bin")

    def setUp(self):
        """Se ejecuta antes de cada prueba"""
        if not os.path.exists(self.bin_path):
            os.makedirs(self.bin_path)
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
        
        # Verificar que al menos un archivo block_*.bin existe
        block_files = [f for f in os.listdir(self.bin_path) if f.startswith('block_')]
        self.assertTrue(len(block_files) > 0, "No se encontraron archivos de bloque")
        
        # Verificar que existe el archivo de normas
        self.assertTrue(
            os.path.exists(os.path.join(self.bin_path, "norms.bin")),
            "No se encontró el archivo de normas"
        )
        
        # Verificar contenido de algún bloque
        with open(os.path.join(self.bin_path, block_files[0]), 'rb') as f:
            block_data = pickle.load(f)
            self.assertTrue(len(block_data) > 0, "Bloque vacío encontrado")

    def test_search_single_term(self):
        """Test de búsqueda de un solo término"""
        self.index.build_index(self.test_docs)
        results = self.index.search("fox")
        self.assertTrue(len(results) > 0, "No se encontraron resultados")
        
        # Verificar que los scores están ordenados
        scores = [score for doc_id, score in results]
        self.assertEqual(
            scores, 
            sorted(scores, reverse=True),
            "Los resultados no están ordenados por score"
        )

    def test_search_multiple_terms(self):
        """Test de búsqueda con múltiples términos"""
        self.index.build_index(self.test_docs)
        results = self.index.search("quick brown")
        self.assertTrue(len(results) > 0, "No se encontraron resultados")
        
        # Verificar que los documentos esperados están en los primeros resultados
        doc_ids = {result[0] for result in results[:3]}
        expected_docs = {0, 1, 3}  # documentos que contienen 'quick' y/o 'brown'
        self.assertTrue(
            expected_docs.issubset(doc_ids),
            f"No se encontraron todos los documentos esperados. Esperados: {expected_docs}, Encontrados: {doc_ids}"
        )

    def test_search_nonexistent_term(self):
        """Test de búsqueda de término inexistente"""
        self.index.build_index(self.test_docs)
        results = self.index.search("unknownterm")
        self.assertEqual(len(results), 0, "Se encontraron resultados para un término inexistente")

    def test_empty_search(self):
        """Test de búsqueda vacía"""
        self.index.build_index(self.test_docs)
        results = self.index.search("")
        self.assertEqual(len(results), 0, "Se encontraron resultados para una búsqueda vacía")

    def test_document_norms(self):
        """Test de cálculo de normas de documentos"""
        self.index.build_index(self.test_docs)
        self.assertIsNotNone(self.index.document_norms)
        self.assertEqual(
            len(self.index.document_norms),
            len(self.test_docs),
            "El número de normas no coincide con el número de documentos"
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)