import os
import struct
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from InvertedIndexFinal import InvertedIndexFinal

def test_inverted_index():
    """
    Suite completa de pruebas para SPIMI
    """
    print("\n=== SPIMI Implementation Test Suite ===")
    
    test_documents = [
        "El gato negro salta sobre el tejado rojo",
        "Un gato blanco duerme en el jardín",
        "El perro ladra al gato negro en el jardín",
        "Los pájaros cantan en el jardín verde",
        "El gato negro y el gato blanco juegan juntos"
    ]
    test_documents = [
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
    
    # Usar tamaños más pequeños para forzar múltiples bloques
    index = InvertedIndexFinal(block_size=100, dict_size=5)
    
    try:
        # Fase 1: Construcción del índice
        print("\n1. Building Index Phase:")
        print("------------------------")
        index.build_index(test_documents)
        
        # Fase 2: Análisis de bloques
        print("\n2. Block Statistics:")
        print("-------------------")
        for block_path in index.temp_blocks:
            with open(os.path.join(block_path, "terms.bin"), "rb") as f:
                terms = set()
                try:
                    while True:
                        term_len = struct.unpack('I', f.read(4))[0]
                        term = f.read(term_len).decode('utf-8')
                        terms.add(term)
                        f.seek(12, 1)  # Saltar position y df
                except:
                    pass
                print(f"Block {os.path.basename(block_path)}: {len(terms)} términos únicos")
        
        # Fase 3: Merge
        print("\n3. Executing Merge Phase:")
        print("------------------------")
        index.merge_blocks()
        
        # Fase 4: Verificación
        print("\n4. Index Verification:")
        print("---------------------")
        expected_terms = {
            'flower': 10,   # Aparece 5 veces
            'like': 1,  # Aparece 3 veces
            'season': 1  # Aparece 3 veces
        }
        
        # Verificar el índice final
        with open(os.path.join(index.bin_path, "final_index.bin"), "rb") as f:
            while True:
                try:
                    term_len = struct.unpack('I', f.read(4))[0]
                    term = f.read(term_len).decode('utf-8')
                    df = struct.unpack('I', f.read(4))[0]
                    
                    if term in expected_terms:
                        print(f"Término '{term}': encontrado {df} veces (esperado: {expected_terms[term]})")
                    
                    # Saltar postings
                    for _ in range(df):
                        f.seek(8, 1)  # Saltar doc_id y freq
                except:
                    break
                    
        print("\nPrueba completada exitosamente!")
        return True
        
    except Exception as e:
        print(f"\nError en la prueba: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
test_inverted_index()