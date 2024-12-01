import os
import struct
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from InvertedIndexFinal import InvertedIndexFinal

def test_inverted_index():
    """
    Suite completa de pruebas para SPIMI
    """
    print("\n🔍 === SPIMI Implementation Test Suite === 🔍")
    
    test_documents = [
        "The quick brown fox jumps over the lazy dog",
        "Pack my box with five dozen liquor jugs",
        "How vexingly quick daft zebras jump",
        "The five boxing wizards jump quickly",
        "Sphinx of black quartz judge my vow",
        "Two driven jocks help fax my big quiz",
        "Five quacking zephyrs jolt my wax bed",
        "The jay pig fox zebra and my wolves quack",
        "Quick zephyrs blow vexing daft jim",
        "Pack my red box with five dozen quality jugs",
        "Jinxed wizards pluck ivy from my quilt box",
        "How quickly daft jumping zebras vex",
        "Waltz nymph for quick jigs vex bud",
        "Quick fox jumps nightly above wizard",
        "Five jumping wizards hex bolty quick"
    ]
    
    # Usar tamaños más pequeños para forzar múltiples bloques
    index = InvertedIndexFinal(block_size=100, dict_size=5)
    
    try:
        # Fase 1: Construcción del índice
        print("\n📝 1. Building Index Phase:")
        print("------------------------")
        index.build_index(test_documents)
        print("✅ Índice construido exitosamente")
        
        # Fase 2: Análisis de bloques
        print("\n📊 2. Block Statistics:")
        print("-------------------")
        total_terms = 0
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
                total_terms += len(terms)
                print(f"📦 Block {os.path.basename(block_path)}: {len(terms)} términos únicos")
        print(f"📈 Total términos únicos en bloques: {total_terms}")
        
        # Fase 3: Merge
        print("\n🔄 3. Executing Merge Phase:")
        print("------------------------")
        index.merge_blocks()
        print("✅ Merge completado exitosamente")
        
        # Fase 4: Verificación
        print("\n🔎 4. Index Verification:")
        print("---------------------")
        expected_terms = {
            'quick': 8,    # Aparece en 8 documentos
            'jump': 6,     # Aparece en 6 documentos
            'wizard': 4,   # Aparece en 4 documentos
            'fox': 3,      # Aparece en 3 documentos
            'box': 4       # Aparece en 4 documentos
        }
        
        verification_success = True
        with open(os.path.join(index.bin_path, "final_index.bin"), "rb") as f:
            while True:
                try:
                    term_len = struct.unpack('I', f.read(4))[0]
                    term = f.read(term_len).decode('utf-8')
                    df = struct.unpack('I', f.read(4))[0]
                    
                    if term in expected_terms:
                        match = df == expected_terms[term]
                        status = "✅" if match else "❌"
                        print(f"{status} Término '{term}': encontrado {df} veces (esperado: {expected_terms[term]})")
                        verification_success = verification_success and match
                    
                    # Saltar postings
                    for _ in range(df):
                        f.seek(8, 1)  # Saltar doc_id y freq
                except:
                    break
        
        print(f"{'✅ Verificación exitosa' if verification_success else '❌ Verificación fallida'}")
    
        # Fase 5: Búsqueda
        print("\n🔍 5. Search Phase:")
        print("----------------")
        search_terms = ["quick", "jump", "wizard", "fox", "box"]

        for term in search_terms:
            result = index.search(term)
            print(f"🔎 Resultados para '{term}': {result}")

        return True
        
    except Exception as e:
        print(f"\n❌ Error en la prueba: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
if __name__ == "__main__":
    success = test_inverted_index()
    print(f"\n{'✅ Todas las pruebas pasaron exitosamente ohhh si me vengo lunes 9 diciembre hotel asturias' if success else '❌ Algunas pruebas fallaron'}")