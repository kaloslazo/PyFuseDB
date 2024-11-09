import pickle
from collections import defaultdict
import numpy as np
from TextPreProcess import TextPreProcess

class InvertedIndex:
    def __init__(self, block_size=1000, batch_size=100):
        self.block_size = block_size
        self.batch_size = batch_size
        self.current_block = defaultdict(list)
        self.block_count = 0

        self.preprocessor = TextPreProcess()

        self.doc_count = 0
        self.document_norms = []


    def build_index(self, documents):
        # reset index
        self.__init__(self.block_size, self.batch_size)

        print(f"Construyendo índice con {len(documents)} documentos")
        for batch_index in range(0, len(documents), self.batch_size):
            batch = documents[batch_index:batch_index+self.batch_size]

            # create postings list for each token
            # self.current_block = {
            #   w1: [(doc1, tf1), (doc2, tf2), ...]
            #   w2: [(doc1, tf1), (doc3, tf3), ...]
            #   ... }

            for i, doc in enumerate(batch):
                self.doc_count += 1
                doc_id = batch_index + i

                doc_len = 0
                tokens = self.tokenize(doc)

                for token in tokens:
                    doc_len += 1
                    if doc_id not in self.current_block[token]:
                        self.current_block[token].append((doc_id, 1))
                    else:
                        for (d_id, tf) in self.current_block[token]:
                            if d_id == doc_id:
                                tf += 1
                                break

                if len(self.current_block) >= self.block_size:
                    self.write_block()

            print(f"Procesado lote {batch_index//self.batch_size + 1}/{len(documents)//self.batch_size + 1}")

        if self.current_block:
            self.write_block()

        # TODO: Merge blocks
        # self.block_merge_sort_file()
    
        print(f"Índice construido. Numero de bloques: {self.block_count}")

        # calculate document norms
        self.document_norms = np.zeros(self.doc_count)

        for block in range(self.block_count):
            with open(f'block_{block}.bin', 'rb') as f:
                block_data = pickle.load(f)

            for token, postings_list in block_data.items():
                for (doc_id, tf) in postings_list:
                    tfidf = self.get_tfidf(tf, len(postings_list))
                    self.document_norms[doc_id] += tfidf ** 2

        self.document_norms = np.sqrt(self.document_norms)
        self.save_norms()


    def search(self, query, top_k: int = 10):
        print(f"Buscando: '{query}'")

        scores = defaultdict(float)

        tokens = self.tokenize(query)
        query_tfidf = np.zeros(len(tokens))
        query_norm = 0

        for i, term in enumerate(tokens):
            # fetch postings list for token
            postings_list = [(0, 1)]

            if not postings_list:
                continue

            for (doc_id, tf) in postings_list: 
                tfidf = self.get_tfidf(tf, len(postings_list))
                scores[doc_id] += tfidf * query_tfidf[i]


        if not self.document_norms:
            self.load_norms()
        
        query_norm = np.sqrt(query_norm)

        for doc_id in scores.keys():
            scores[doc_id] /= (self.document_norms[doc_id] * query_norm)


        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print(f"Número de resultados encontrados: {len(scores)}")
        return scores[:top_k]


    def tokenize(self, text):
        return self.preprocessor.tokenize(text)

    def get_tfidf(self, term_freq, doc_freq):
        return np.log10(1 + term_freq) * np.log10(self.doc_count / doc_freq)

    def save_norms(self):
        with open('document_norms.bin', 'wb') as f:
            pickle.dump(self.document_norms, f)

    def load_norms(self):
        with open('document_norms.bin', 'rb') as f:
            self.document_norms = pickle.load(f)

    def write_block(self):
        block_file = f'block_{self.block_count}.bin'
        with open(block_file, 'wb') as f:
            pickle.dump(dict(self.current_block), f)
        self.block_count += 1
        self.current_block.clear()
