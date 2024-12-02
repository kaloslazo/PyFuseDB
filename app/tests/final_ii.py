import os
import struct
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from InvertedIndexFinal import InvertedIndexFinal

def test_inverted_index():
    """suite completa de pruebas para spimi"""
    print("\n🔍 === pruebas de implementacion spimi === 🔍")
    
    # documentos de prueba con variaciones morfologicas y repeticiones
    test_documents = [
        "The quick brown foxes jump over the lazy dogs",
        "Pack my boxes with five dozen liquor jugs", 
        "How vexingly quick daft zebras jumping",
        "The five boxing wizards jumped quickly",
        "Sphinx of black quartz judge my vows",
        "Two driven jocks help fax my big quizzes",
        "Five quacking zephyrs jolt my wax bed",
        "The jay pigs foxes zebras and my wolves quack",
        "Quick zephyrs blow vexing daft jim",
        "Pack my red boxes with five dozen quality jugs",
        "Jinxed wizards plucking ivy from my quilt box",
        "How quickly daft jumping zebras vex",
        "Waltz nymph for quick jigs vex bud",
        "Quick foxes jumping nightly above wizard",
        "Five jumping wizards hex bolty quick",
        "The flowers are blooming in the gardens",
        "Many flowers bloomed last spring season",
        "Running dogs chase playing cats daily",
        "Cats running and dogs playing together",
        "Birds flying over blooming flower fields"
    ]
    
    index = InvertedIndexFinal(block_size=25, dict_size=5)
    
    try:
        # fase 1: construccion del indice
        print("\n📝 1. fase de construccion:")
        print("------------------------")
        index.build_index(test_documents)
        print("✓ indice construido")
        
        # fase 2: analisis de bloques
        print("\n📊 2. estadisticas de bloques:")
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
                        f.seek(12, 1)  # saltar position y df
                except:
                    pass
                total_terms += len(terms)
                print(f"📦 bloque {os.path.basename(block_path)}: {len(terms)} terminos unicos")
        print(f"📈 total terminos unicos en bloques: {total_terms}")
        
        # fase 3: merge
        print("\n🔄 3. fase de merge:")
        print("------------------------")
        index.merge_blocks()
        print("✓ merge completado")
        
        # fase 4: verificacion
        print("\n🔎 4. verificacion del indice:")
        print("---------------------")
        expected_terms = {
            'quick': 8,    # quick, quickly, quickly
            'jump': 6,     # jump, jumping, jumped, jumping, jumping
            'wizard': 4,   # wizard, wizards, wizards
            'fox': 3,      # foxes, foxes
            'box': 4,      # boxes, boxing, box
            'flower': 3,   # flowers, flower
            'bloom': 3,    # blooming, bloomed, blooming
            'run': 2,      # running, running
            'dog': 3,      # dogs, dogs
            'cat': 2       # cats, cats
        }
        
        verification_success = True
        with open(os.path.join(index.bin_path, "final_index.bin"), "rb") as f:
            found_terms = {}
            while True:
                try:
                    term_len = struct.unpack('I', f.read(4))[0]
                    term = f.read(term_len).decode('utf-8')
                    df = struct.unpack('I', f.read(4))[0]
                    
                    if term in expected_terms:
                        found_terms[term] = df
                        if df != expected_terms[term]:
                            print(f"✗ Error: termino '{term}' encontrado {df} veces pero se esperaban {expected_terms[term]}")
                            verification_success = False
                        else:
                            print(f"✓ termino '{term}': encontrado {df} veces correctamente")
                    
                    # saltar postings
                    for _ in range(df):
                        f.seek(8, 1)  # saltar doc_id y freq
                except:
                    break

        # Verificar que se encontraron todos los términos esperados
        for term in expected_terms:
            if term not in found_terms:
                print(f"✗ Error: termino '{term}' no encontrado en el índice")
                verification_success = False
        
        if verification_success:
            print("✓ verificacion exitosa: todos los términos coinciden con lo esperado")
        else:
            print("✗ verificacion fallida: algunos términos no coinciden con lo esperado")
            raise Exception("La verificación del índice falló")
    
        # fase 5: busqueda
        print("\n🔍 5. fase de busqueda:")
        print("----------------")
        search_terms = ["quick fox", "jump", "wizard", "fox", "box", "flower", "bloom", "run", "dog", "cat"]

        for term in search_terms:
            result = index.search(term)
            if not result:
                print(f"✗ Error: búsqueda de '{term}' no produjo resultados")

        return verification_success
        
    except Exception as e:
        print(f"\n✗ error en la prueba: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
if __name__ == "__main__":
    success = test_inverted_index()
    print(f"\n{'✓ pruebas completadas' if success else '✗ pruebas fallidas'}")