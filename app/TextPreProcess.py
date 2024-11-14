import nltk
import re
import os
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict

nltk.download('punkt')
nltk.download('punkt_tab')

class TextPreProcess:
    def __init__(self):
        self.stopListPath = os.path.join(os.path.dirname(__file__), "data", "stopList.txt")
        self.stemmer = SnowballStemmer("english")
        self.stopList = set()  # Change 6: Using a set for faster lookup
        print("TextPreProcess initialized.")

    def loadStopList(self):
        try:
            with open(self.stopListPath, encoding="utf-8") as file:
                self.stopList = {line.strip().lower() for line in file}
            print("Stopwords list loaded successfully.")
        except FileNotFoundError:
            print(f"Stopwords list not found at {self.stopListPath}")
            # Crear una lista de stopwords básica si no se encuentra el archivo
            self.stopList = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", 
                           "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", 
                           "to", "was", "were", "will", "with"}

    def removeStopWords(self, tokens):
        return [word for word in tokens if word not in self.stopList]  # Change 2: Stopword removal after lowercasing

    def removePunctuation(self, tokens):
        return [re.sub(r'\W+', '', word) for word in tokens if re.sub(r'\W+', '', word)]  # Change 3: Using regex

    def tokenize(self, text):
        return nltk.word_tokenize(text.lower())  # Tokenizing and lowercasing

    def stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def processText(self, text):
        # 1. Tokenization and lowercasing
        tokens = self.tokenize(text)
        # 2. Stopword filtering
        tokens = self.removeStopWords(tokens)
        # 3. Removing non-alphanumeric characters
        tokens = self.removePunctuation(tokens)
        # 4. Applying stemming
        tokens = self.stem(tokens)

        return tokens

    def preprocess_query(self, text):
        """Preprocesa una consulta"""
        # Convertir todo a minúsculas
        text = text.lower()
        
        # Tokenizar
        tokens = self.processText(text)
        
        # Contar frecuencias
        term_freq = defaultdict(int)
        for token in tokens:
            if len(token) > 1:  # Ignorar tokens de un solo carácter
                term_freq[token] += 1
                
        print(f"Tokens procesados: {dict(term_freq)}")
        return term_freq