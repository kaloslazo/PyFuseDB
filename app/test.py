from InvertedIndex import InvertedIndex
from TextPreProcess import TextPreProcess
import nltk

index = InvertedIndex(block_size=2, dict_size=20)
documents = [
    "Spring is a season of renewal and fresh beginnings.",
    "Flowers bloom in abundance during the spring season.",
    "In spring, the days grow longer, and the weather becomes warmer.",
    "Spring brings colorful flowers and fresh green leaves on trees.",
    "Many animals come out of hibernation in spring.",
    "The arrival of spring means the return of chirping birds.",
    "Spring is a popular time for planting gardens and growing flowers.",
    "In spring, trees and plants begin to grow new leaves and flowers.",
    "People often do spring cleaning to refresh their homes and surroundings.",
    "Spring rains help nourish the soil for new plant growth.",
    "The beauty of spring flowers brightens up gardens and parks.",
    "Many festivals and holidays are celebrated during the spring season.",
    "Butterflies and bees are often seen buzzing around flowers in spring.",
    "Spring is the time when farmers start preparing their fields for planting.",
    "Mild temperatures in spring make it ideal for outdoor activities.",
    "Spring symbolizes rebirth and the start of a new life cycle.",
    "Birds build nests in spring to prepare for the arrival of their young.",
    "Many animals shed their winter coats as spring arrives.",
    "People enjoy spending time outside to enjoy the pleasant spring weather.",
    "Spring showers bring new life to grass, flowers, and trees.",
    "Children love playing outside in the warmer spring weather.",
    "In spring, flowers like tulips, daffodils, and lilacs bloom beautifully.",
    "Spring is known as a time for renewal and growth in nature.",
    "The days continue to get warmer and longer as spring progresses.",
    "Many people enjoy picnics and hikes in the springtime.",
    "Spring brings a sense of energy and freshness after winter.",
    "Spring fashion often includes bright colors and lighter clothing.",
    "Cherry blossoms bloom spectacularly in spring, attracting many visitors.",
    "Gardeners enjoy planting seeds and watching them grow in spring.",
    "In many cultures, spring is a time for festivals celebrating new life."
]

documents = [
    "Spring is a season of renewal spring and fresh beginnings.",
    "Flowers bloom in abundance during the spring season.",
    "In spring, the days grow longer, and the weather becomes warmer.",
    "Spring brings colorful flowers and fresh green leaves on trees.",
    "Many animals come out of hibernation in spring.",
    "The arrival of spring means the return of chirping birds.",
    "Spring is a popular time for planting gardens and growing flowers.",
    "Summer, Winter, Fall, and many other seasons."
]


# Testeando el preprocesamiento de texto
# preprocessor = TextPreProcess()
# preprocessor.loadStopList()

# for doc in documents:
#     tokens = preprocessor.processText(doc)
#     print(tokens)

# Testeando la construcción del índice invertido
index.build_index(documents)

index.debug_blocks()

query ="Spring vacation is my favourite season. many flowers, many animals, and more"
result = index.search(query)

for (doc_id, score) in result:
    print(f"DOC {doc_id} : {float(score)}")

print("Bye!")

