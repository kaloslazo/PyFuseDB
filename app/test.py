from InvertedIndex import InvertedIndex
from TextPreProcess import TextPreProcess
import nltk

index = InvertedIndex(block_size=2, dict_size=20)
documents = [
    "Spring is a season of renewal spring and fresh beginnings.",
    "Flowers bloom in abundance during the spring season.",
    "In spring, the days grow longer, and the weather becomes warmer.",
    "Spring brings colorful flowers and fresh green leaves on trees.",
    "Many animals come out of hibernation in spring.",
    "The arrival of spring means the return of chirping birds.",
    "Spring is a popular time for planting gardens and growing flowers."
]


# Testeando el preprocesamiento de texto
preprocessor = TextPreProcess()
preprocessor.loadStopList()

for doc in documents:
    tokens = preprocessor.processText(doc)
    print(tokens)

# Testeando la construcción del índice invertido
index.build_index(documents)
index.debug_blocks()

# Test de las normas
index.load_norms() 
print(index.document_norms)