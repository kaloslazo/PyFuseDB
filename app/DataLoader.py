import pandas as pd
from tqdm import tqdm
from SqlParser import SqlParser
from InvertedIndex import InvertedIndex

class DataLoader:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.data = None
        self.index = InvertedIndex()
        self.sqlParser = SqlParser()
        print("DataLoader inicializado.")

    def loadData(self):
        print("Cargando dataset...")
        try:
            self.data = pd.read_csv(self.dataPath)
            print(f"Dataset cargado exitosamente.\nColumnas: {self.data.columns}\nFilas: {len(self.data)}")

            print("Construyendo índice invertido...")
            for index, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Procesando documentos"):
                self.index.add_document(index, str(row["texto_concatenado"]))

            print("Calculando TF-IDF...")
            self.index.build_index(self.data["texto_concatenado"].astype(str).tolist())
            print("Índice invertido construido exitosamente.")
        except Exception as e:
            print(f"Error durante la carga de datos: {str(e)}")
            raise

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
