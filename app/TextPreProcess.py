import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict

nltk.download('punkt')
nltk.download('punkt_tab')

class TextPreProcess:
    def __init__(self):
        self.stopListPath = "./data/stopList.txt"
        self.stemmer = SnowballStemmer("english")
        self.stopList = set()  # Change 6: Using a set for faster lookup
        print("TextPreProcess initialized.")

    def loadStopList(self):
        try:
            with open(self.stopListPath, encoding="utf-8") as file:
                self.stopList = {line.strip().lower() for line in file}  # Change 1: Fixed issue with clearing the list
                print("Stopwords list loaded successfully.")
        except FileNotFoundError:
            print("Stopwords list not found.")

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


    def preprocess_query(self, query):
        tokens = self.processText(query)
        tf_query = defaultdict(int)

        # Count term frequencies in the query
        for token in tokens:
            tf_query[token] += 1

        return tf_query

