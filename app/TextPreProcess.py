import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

nltk.download('punkt')

class TextPreProcess:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.stopListPath = "./app/data/stopList.txt"
        self.stemmer = SnowballStemmer("english")
        self.stopList = []
        print("TextPreProcess inicializado.")

    def loadStopList(self):
        try:
            with open(self.stopListPath, encoding="utf-8") as file:
                self.stopList = [line.rstrip().lower() for line in file]
                print("Lista de stopwords cargada exitosamente.")

        except FileNotFoundError:
            print("No se ha encontrado la lista de stopwords.")
        self.stopList = []

    def removeStopWords(self, text):
        return [word for word in text if word not in self.stopList]

    def removePunctuation(self, text):
        return [word for word in text if word.isalnum()]

    def tokenize(self, text):
        return nltk.word_tokenize(text.lower())

    def stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def processText(self, text):
        # 1. Tokenización
        tokens = self.tokenize(text)
        # 2. Filtrar stopwords
        tokens = self.removeStopWords(tokens)
        # 3. Eliminar no alfanuméricos
        tokens = self.removePunctuation(tokens)
        # 4. Aplicar stemming (reduccion de palabras)
        tokens = self.stem(tokens)

        return tokens
    
    def prepareDataset(self):
        data = pd.read_csv(self.dataPath)
        data['texto_procesado'] = data['texto_concatenado'].apply(self.processText)

        return data
