import pickle
import pandas as pd
from tqdm import tqdm
import os
from SqlParser import SqlParser
from InvertedIndex import InvertedIndex

class DataLoader:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.data = None
        # Usar parámetros más apropiados para el tamaño del dataset
        self.index = InvertedIndex(block_size=5000, dict_size=100000)
        self.sqlParser = SqlParser()
        print("DataLoader inicializado.")

    def loadData(self):
        print("Cargando dataset...")
        try:
            self.data = pd.read_csv(self.dataPath)
            print(f"Dataset cargado exitosamente.\nColumnas: {self.data.columns}\nFilas: {len(self.data)}")
            
            # Forzar reconstrucción del índice si los documentos han cambiado
            if self._check_existing_index() and self._verify_index_size():
                print("Cargando índice existente...")
                self._load_existing_index()
            else:
                print("Construyendo nuevo índice...")
                # Limpiar archivos antiguos
                self.index.clear_files()
                # Construir nuevo índice
                self.index.build_index(self.data["texto_concatenado"].astype(str).tolist())
                print("Índice construido exitosamente")
            
            self._verify_index()
            
        except Exception as e:
            print(f"Error durante la carga de datos: {e}")
            import traceback
            print("Stacktrace:")
            print(traceback.format_exc())

    def _verify_index_size(self):
        """Verifica que el índice existente coincida con el número de documentos"""
        try:
            with open(os.path.join(self.index.bin_path, "dict_0.bin"), "rb") as f:
                dict_data = picxkle.load(f)
                # Obtener el máximo doc_id de los postings
                max_doc_id = max(doc_id for term_data in dict_data.values() 
                               for doc_id, _ in term_data[1])
                # Verificar que coincida con el número de documentos
                return max_doc_id + 1 == len(self.data)
        except:
            return False

    def _check_existing_index(self):
        """Verifica si existe un índice válido"""
        required_files = [
            os.path.join(self.index.bin_path, "dictionary.bin"),
            os.path.join(self.index.bin_path, "norms.bin"),
            os.path.join(self.index.bin_path, "dict_0.bin")
        ]
        return all(os.path.exists(f) for f in required_files)

    def _load_existing_index(self):
        """Carga un índice existente"""
        try:
            self.index.load_main_dictionary()
            self.index.load_norms()
            # Establecer doc_count desde el dataset actual
            self.index.doc_count = len(self.data)
            print(f"Índice cargado con {len(self.index.main_dictionary)} términos")
            print("Muestra de términos:", list(self.index.main_dictionary.keys())[:5])
            
        except Exception as e:
            print(f"Error cargando índice existente: {e}")
            print("Reconstruyendo índice...")
            self.index.build_index(self.data["texto_concatenado"].astype(str).tolist())

    def executeQuery(self, query, topK=10):
        print(f"Ejecutando query: {query}\nTop K: {topK}")

        parsed_query = self.sqlParser.parseQuery(query)
        fields = parsed_query['fields']
        like_term = parsed_query['like_term']

        if '*' in fields: fields = list(self.data.columns)
        print(f"Campos seleccionados: {fields}")
        print(f"Término de búsqueda: {like_term}")

        if like_term: results = self.index.search(like_term, topK)
        else:
            # Si no hay término de búsqueda, devolver los primeros topK resultados
            results = [(i, 1.0) for i in range(min(topK, len(self.data)))]

        if not results:
            print("No se encontraron resultados.")
            return []

        formatted_results = []
        for doc_id, score in results:
            row = self.data.iloc[doc_id]
            row_data = [str(row[field]) if field in self.data.columns else 'N/A' for field in fields]
            row_data.append(f"{score * 100:.2f}")
            formatted_results.append(row_data)

        print(f"Resultados formateados: {formatted_results}")
        return formatted_results