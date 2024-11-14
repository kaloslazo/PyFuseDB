import unittest
import os
import sys
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from InvertedIndex import InvertedIndex

class InvertedIndexTest(unittest.TestCase):
    bin_path = os.path.join("app", "data", "bin")

    def setUp(self):
        if not os.path.exists(self.bin_path):
            os.makedirs(self.bin_path)
        self.clean_bin_directory()
        self.index = InvertedIndex(block_size=1000, dict_size=100000)
        
        # Documentos de prueba
        self.test_docs = [
            "Beautiful flowers bloom in spring time",
            "The garden is full of red and yellow flowers",
            "I love flowers, especially roses and lilies",
            "Flowers make people happy and bring joy",
            "Spring flowers are the first sign of the season",
            "These flowers smell wonderful",
            "The flower shop sells fresh bouquets",
            "Wild flowers grow in the meadow",
            "She likes to pick flowers in the morning",
            "The flower arrangement looks perfect"
        ]

    def tearDown(self):
        self.clean_bin_directory()

    def clean_bin_directory(self):
        if os.path.exists(self.bin_path):
            for file in os.listdir(self.bin_path):
                if file.endswith('.bin'):
                    os.remove(os.path.join(self.bin_path, file))

    def test_index_content(self):
        """Test para verificar el contenido del índice"""
        self.index.build_index(self.test_docs)
        
        self.assertTrue(
            os.path.exists(os.path.join(self.bin_path, "dictionary.bin")),
            "No se encontró el diccionario principal"
        )
        
        # Cargar el diccionario y verificar términos específicos
        self.index.load_main_dictionary()
        
        # Términos después del stemming
        important_terms = {
            'flower',  # stems de 'flower' y 'flowers'
            'spring',
            'garden',
            'love'
        }
        
        for term in important_terms:
            self.assertIn(
                term, 
                self.index.main_dictionary,
                f"Término '{term}' no encontrado en el índice"
            )
            
            df, postings = self.index.main_dictionary[term]
            if term == 'flower':
                # Debería aparecer en casi todos los documentos
                self.assertGreaterEqual(
                    df, 
                    8, 
                    f"El término 'flower' debería aparecer en al menos 8 documentos, apareció en {df}"
                )
            
            self.assertTrue(
                len(postings) > 0,
                f"No hay postings para el término '{term}'"
            )

    def test_search_flowers(self):
        """Test específico para búsqueda de 'flowers'"""
        self.index.build_index(self.test_docs)
        
        # Probar tanto 'flower' como 'flowers'
        for query in ['flower', 'flowers']:
            results = self.index.search(query)
            self.assertTrue(
                len(results) > 0,
                f"No se encontraron resultados para '{query}'"
            )
            
            # Verificar que los documentos relevantes están en los primeros resultados
            top_docs = [doc_id for doc_id, score in results[:5]]
            flower_docs = [i for i, doc in enumerate(self.test_docs) 
                         if 'flower' in doc.lower() or 'flowers' in doc.lower()]
            
            self.assertTrue(
                any(doc_id in flower_docs for doc_id in top_docs),
                "Los documentos más relevantes no están en los primeros resultados"
            )

    def test_search_variations(self):
        """Test para variaciones de búsqueda"""
        self.index.build_index(self.test_docs)
        
        queries = [
            "flower",
            "flowers",
            "spring flower",
            "beautiful flowers",
            "flower garden"
        ]
        
        for query in queries:
            results = self.index.search(query)
            self.assertTrue(
                len(results) > 0,
                f"No se encontraron resultados para '{query}'"
            )
            
            # Verificar ranking
            scores = [score for _, score in results]
            self.assertEqual(
                scores,
                sorted(scores, reverse=True),
                f"Resultados no ordenados correctamente para '{query}'"
            )

    def test_index_statistics(self):
        """Test para verificar estadísticas del índice"""
        self.index.build_index(self.test_docs)
        
        self.assertEqual(
            self.index.doc_count,
            len(self.test_docs),
            "Número incorrecto de documentos en el índice"
        )
        
        self.assertTrue(
            all(norm > 0 for norm in self.index.document_norms),
            "Algunas normas de documentos son 0"
        )
        
        self.index.load_main_dictionary()
        self.assertTrue(
            len(self.index.main_dictionary) > 0,
            "Diccionario principal vacío"
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)