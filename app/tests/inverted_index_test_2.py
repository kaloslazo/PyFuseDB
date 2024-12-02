import os
import sys
import unittest
from collections import defaultdict
from io import StringIO
from unittest.runner import TextTestRunner
from unittest.result import TestResult
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from InvertedIndex import InvertedIndex

class ColoredTestResult(TestResult):
    """Resultado de prueba personalizado con colores y emojis"""
    def __init__(self):
        super().__init__()
        self.successes = []
        
    def addSuccess(self, test):
        self.successes.append(test)
        
    def wasSuccessful(self):
        return len(self.failures) == len(self.errors) == 0

class ColoredTestRunner(TextTestRunner):
    """Runner personalizado que usa nuestro resultado con colores"""
    def _makeResult(self):
        return ColoredTestResult()
        
    def run(self, test):
        result = super().run(test)
        
        # Imprimir resumen personalizado
        print("\n" + "="*50)
        print("📊 RESUMEN DE PRUEBAS")
        print("="*50)
        
        # Pruebas exitosas
        print("\n✅ Pruebas Exitosas:")
        for test in result.successes:
            print(f"  ✓ {test.shortDescription() or test._testMethodName}")
            
        # Pruebas fallidas
        if result.failures:
            print("\n❌ Pruebas Fallidas:")
            for test, trace in result.failures:
                print(f"  ✗ {test.shortDescription() or test._testMethodName}")
                print(f"    Razón: {trace.split('AssertionError: ')[-1].split('\n')[0]}")
                
        # Errores
        if result.errors:
            print("\n⚠️ Errores:")
            for test, trace in result.errors:
                print(f"  ⚠ {test.shortDescription() or test._testMethodName}")
                print(f"    Error: {trace.split('Error: ')[-1].split('\n')[0]}")
                
        # Estadísticas finales
        total = result.testsRun
        passed = len(result.successes)
        failed = len(result.failures)
        errors = len(result.errors)
        
        print("\n" + "="*50)
        print("📈 ESTADÍSTICAS FINALES")
        print("="*50)
        print(f"Total de pruebas: {total}")
        print(f"✅ Exitosas: {passed}")
        print(f"❌ Fallidas: {failed}")
        print(f"⚠️ Errores: {errors}")
        print(f"Porcentaje de éxito: {(passed/total)*100:.1f}%")
        
        # Veredicto final
        print("\n" + "="*50)
        if result.wasSuccessful():
            print("🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        else:
            print("❌ ALGUNAS PRUEBAS FALLARON")
        print("="*50 + "\n")
        
        return result

class TestInvertedIndex(unittest.TestCase):
    def setUp(self):
        """Configura el ambiente de pruebas"""
        self.index = InvertedIndex(block_size=2, dict_size=20)
        self.documents = [
            "Spring is a season of renewal spring and fresh beginnings.",
            "Flowers bloom in abundance during the spring season.",
            "In spring, the days grow longer, and the weather becomes warmer.",
            "Spring brings colorful flowers and fresh green leaves on trees.",
            "Many animals come out of hibernation in spring.",
            "The arrival of spring means the return of chirping birds.",
            "Spring is a popular time for planting gardens and growing flowers.",
            "Summer, Winter, Fall, and many other seasons."
        ]
        self.index.build_index(self.documents)

    def test_dictionary_creation(self):
        """Verifica la creación y estructura del diccionario"""
        final_dict = self.index._read_dict(0)
        self.assertGreater(len(final_dict), 0)
        
        # Verificar términos esperados
        expected_terms = ['spring', 'flower', 'season']
        for term in expected_terms:
            self.assertTrue(
                any(term in key for key in final_dict.keys()),
                f"No se encontró el término '{term}'"
            )

    def test_search_functionality(self):
        """Verifica la funcionalidad básica de búsqueda"""
        results = self.index.search("flowers")
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), len(self.documents))

    def test_document_scoring(self):
        """Verifica el sistema de puntuación de documentos"""
        query = "Spring flowers"
        results = self.index.search(query)
        
        # Verificar orden descendente de scores
        scores = [score for _, score in results]
        self.assertEqual(
            scores, sorted(scores, reverse=True),
            "Los resultados no están ordenados por relevancia"
        )

    def test_term_frequencies(self):
        """Verifica el cálculo correcto de frecuencias de términos"""
        final_dict = self.index._read_dict(0)
        spring_entry = next((v for k, v in final_dict.items() if 'spring' in k.lower()), None)
        self.assertIsNotNone(spring_entry, "No se encontró el término 'spring'")
        self.assertGreater(spring_entry[0], 1, "La frecuencia de 'spring' es incorrecta")

if __name__ == '__main__':
    # Usar nuestro runner personalizado
    runner = ColoredTestRunner(verbosity=2)
    unittest.main(testRunner=runner)