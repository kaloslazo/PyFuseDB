import pickle
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class InvertedIndex:
    def __init__(self, block_size=1000, batch_size=100):
        self.block_size = block_size
        self.batch_size = batch_size
        self.current_block = defaultdict(list)
        self.block_count = 0
        self.document_norms = []
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        self.doc_count = 0

    def add_document(self, doc_id, text):
        tokens = self.vectorizer.build_analyzer()(text)
        for token in tokens:
            self.current_block[token].append(doc_id)

        self.doc_count += 1

        if len(self.current_block) >= self.block_size:
            self.write_block()

    def write_block(self):
        block_file = f'block_{self.block_count}.pkl'
        with open(block_file, 'wb') as f:
            pickle.dump(dict(self.current_block), f)
        self.block_count += 1
        self.current_block.clear()

    def build_index(self, documents):
        print(f"Construyendo índice con {len(documents)} documentos")
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i+self.batch_size]
            tfidf_matrix = self.vectorizer.fit_transform(batch)
            batch_norms = np.linalg.norm(tfidf_matrix.toarray(), axis=1)
            self.document_norms.extend(batch_norms)
            
            for j, doc in enumerate(batch):
                doc_id = i + j
                tokens = self.vectorizer.build_analyzer()(doc)
                for token in tokens:
                    self.current_block[token].append(doc_id)
            
            if len(self.current_block) >= self.block_size:
                self.write_block()
            
            print(f"Procesado lote {i//self.batch_size + 1}/{len(documents)//self.batch_size + 1}")
        
        if self.current_block:
            self.write_block()
        
        print(f"Índice construido. Tamaño del vocabulario: {len(self.vectorizer.vocabulary_)}")
        print(f"Número de bloques: {self.block_count}")

    def search(self, query, top_k=10):
        print(f"Buscando: '{query}'")
        
        query_vector = self.vectorizer.transform([query])
        terms = [self.vectorizer.get_feature_names_out()[i] for i in query_vector.indices]
        print(f"Términos en la consulta: {terms}")
        
        scores = []
        for i in range(self.block_count):
            with open(f'block_{i}.pkl', 'rb') as f:
                block = pickle.load(f)
            for term in query_vector.indices:
                if term in block:
                    for doc_id in block[term]:
                        tfidf = query_vector[0, term] * self.document_norms[doc_id]
                        scores.append((doc_id, tfidf))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        print(f"Número de resultados encontrados: {len(scores)}")
        return scores[:top_k]