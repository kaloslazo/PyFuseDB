import pandas as pd

class DataLoader:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.data = None

    def getDataPath(self):
        return self.dataPath

    def loadData(self):
        print("Cargando dataset...")

        try:
            self.data = pd.read_csv(self.dataPath)
            print(f"Dataset cargado exitosamente.\nColumnas: {self.data.columns}\nFilas: {len(self.data)}")
        except FileNotFoundError:
            print("No se ha encontrado el archivo.")
        except pd.errors.ParserError:
            print("No se ha podido cargar el archivo.")

    def executeQuery(self, query, topK):
        print(f"Ejecutando query: {query}\nTop K: {topK}")
        # esta es data fake - reemplazar por lo que retorne la similitud de coseno :)
        return [
            ["Canción de amor", "Artista 1", 95],
            ["Melodía nocturna", "Artista 2", 87],
            ["Ritmo de verano", "Artista 3", 82],
            ["Balada del viento", "Artista 4", 78],
            ["Sonata del mar", "Artista 5", 75],
        ][:topK]
