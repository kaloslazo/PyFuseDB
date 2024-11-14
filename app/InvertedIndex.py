import pickle
import os
import glob
from collections import defaultdict
import numpy as np
from TextPreProcess import TextPreProcess

class InvertedIndex:
    def __init__(self, block_size=1000, dict_size=100):
        self.block_size = block_size
        self.dict_size = dict_size
        self.index = defaultdict(list)  # Agregado aquí
        self.current_block = defaultdict(list)
        self.dictionary = defaultdict(lambda: [0, None])
        self.doc_count = 0
        self.document_norms = None
        
        # Asegurar que el directorio bin existe
        self.bin_path = os.path.join("app", "data", "bin")
        if not os.path.exists(self.bin_path):
            os.makedirs(self.bin_path)

        self.preprocessor = TextPreProcess()
        self.preprocessor.loadStopList()

    def build_index(self, documents):
        print(f"Construyendo índice con {len(documents)} documentos")
        self.clear_files()
        self.doc_count = len(documents)

        # Primera pasada: Construir diccionario en memoria
        for doc_id, document in enumerate(documents):
            if doc_id % 100 == 0:
                print(f"Procesando documento {doc_id}/{len(documents)}")
            
            # Procesar documento
            tokens = self.tokenize(document)
            term_freq = defaultdict(int)
            
            # Contar frecuencias
            for token in tokens:
                term_freq[token] += 1
            
            # Actualizar índice
            for token, freq in term_freq.items():
                self.index[token].append((doc_id, freq))
                
            # Si el índice es muy grande, escribir a disco
            if len(self.index) >= self.dict_size:
                self._write_index_to_disk()
                
        # Escribir índice restante
        if self.index:
            print("Escribiendo índice final a disco...")
            self._write_index_to_disk()

        # Calcular y guardar normas
        print("Calculando normas de documentos...")
        self._calculate_document_norms()
        self.save_norms()

    def _write_index_to_disk(self):
        """Escribe el índice a disco de forma ordenada"""
        sorted_terms = sorted(self.index.keys())
        current_block = {}
        block_count = len([f for f in os.listdir(self.bin_path) if f.startswith("block_")])
        
        for term in sorted_terms:
            postings = self.index[term]
            current_block[term] = (len(postings), postings)
            
            if len(current_block) >= self.block_size:
                self._write_block(current_block, block_count)
                block_count += 1
                current_block = {}
        
        if current_block:
            self._write_block(current_block, block_count)
        
        self.index.clear()

    def _write_block(self, block, block_num):
        """Escribe un bloque de índice a disco"""
        block_file = os.path.join(self.bin_path, f"block_{block_num}.bin")
        with open(block_file, "wb") as f:
            pickle.dump(block, f)

    def search(self, query, top_k=10):
        """Realiza una búsqueda en el índice"""
        print(f"Buscando: '{query}'")
        scores = defaultdict(float)
        
        # Procesar consulta
        terms_dict = self.preprocessor.preprocess_query(query)
        query_weights = {}
        query_norm = 0
        
        for term, tf in terms_dict.items():
            postings = self._get_postings(term)
            if not postings:
                continue
                
            df = len(postings)
            w_tq = self._calculate_weight(tf, df)
            query_weights[term] = w_tq
            query_norm += w_tq * w_tq
            
            for doc_id, doc_tf in postings:
                w_td = self._calculate_weight(doc_tf, df)
                scores[doc_id] += w_td * w_tq
        
        if query_norm > 0:
            self.load_norms()
            query_norm = np.sqrt(query_norm)
            for doc_id in scores:
                if self.document_norms[doc_id] != 0:  # Evitar división por cero
                    scores[doc_id] /= (self.document_norms[doc_id] * query_norm)
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _get_postings(self, term):
        """Obtiene la lista de postings para un término"""
        postings = []
        for block_file in sorted(glob.glob(os.path.join(self.bin_path, "block_*.bin"))):
            with open(block_file, "rb") as f:
                block = pickle.load(f)
                if term in block:
                    _, term_postings = block[term]
                    postings.extend(term_postings)
        return postings

    def _calculate_weight(self, tf, df):
        """Calcula el peso TF-IDF"""
        if tf <= 0:
            return 0
        return (1 + np.log10(tf)) * np.log10(self.doc_count / df)

    def _calculate_document_norms(self):
        """Calcula las normas de los documentos"""
        self.document_norms = np.zeros(self.doc_count)
        
        for block_file in sorted(glob.glob(os.path.join(self.bin_path, "block_*.bin"))):
            with open(block_file, "rb") as f:
                block = pickle.load(f)
                for term, (df, postings) in block.items():
                    for doc_id, tf in postings:
                        weight = self._calculate_weight(tf, df)
                        self.document_norms[doc_id] += weight * weight
        
        self.document_norms = np.sqrt(self.document_norms)
        self.document_norms[self.document_norms == 0] = 1  # Evitar división por cero

    def tokenize(self, text):
        return self.preprocessor.processText(text)

    def save_norms(self):
        with open(os.path.join(self.bin_path, "norms.bin"), "wb") as f:
            pickle.dump(self.document_norms, f)

    def load_norms(self):
        if self.document_norms is None:
            with open(os.path.join(self.bin_path, "norms.bin"), "rb") as f:
                self.document_norms = pickle.load(f)

    def clear_files(self):
        for file in os.listdir(self.bin_path):
            file_path = os.path.join(self.bin_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error eliminando {file_path}: {e}")

    def debug_blocks(self):
        print("\nDEBUGGING INVERTED INDEX:")
        block_files = sorted(glob.glob(os.path.join(self.bin_path, "block_*.bin")))
        
        print(f"Número de documentos: {self.doc_count}")
        print(f"Número de bloques: {len(block_files)}")
        
        for block_file in block_files[:3]:  # Mostrar solo los primeros 3 bloques
            with open(block_file, "rb") as f:
                block = pickle.load(f)
                print(f"\n{os.path.basename(block_file)}:")
                print(f"Número de términos: {len(block)}")
                
                # Mostrar los primeros 3 términos como ejemplo
                for term in list(block.keys())[:3]:
                    df, postings = block[term]
                    print(f"Término '{term}': df={df}, primeros 3 postings={postings[:3]}")