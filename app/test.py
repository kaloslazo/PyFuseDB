from InvertedIndex import InvertedIndex
from TextPreProcess import TextPreProcess
import nltk

index = InvertedIndex(block_size=2, batch_size=2)
documents = [
    "the quick brown fox jumps over the lazy dog",
    "hello world. Hello brown quick spinning world",
    "my dog's name is warren. He is a good dog",
    "The exam was very difficult. I could not solve it",
    "... so, I decided to make a quick trip to the beach and relax",
    "I still can't believe how fast the red fox was!!!",
    "testing testing testing frequency testing frequency"
]

# Testeando el preprocesamiento de texto
preprocessor = TextPreProcess()
preprocessor.loadStopList()

print(preprocessor.processText(documents[0]))
print(preprocessor.processText(documents[1]))
print(preprocessor.processText(documents[2]))


# Testeando la construcción del índice invertido
index.build_index(documents)
index.debug_blocks()
