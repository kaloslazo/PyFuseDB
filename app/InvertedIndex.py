import pickle
import os
from collections import defaultdict
import numpy as np
from TextPreProcess import TextPreProcess


class InvertedIndex:
    def __init__(self, block_size=1000, dict_size=100):
        self.block_size = block_size
        self.dict_size = dict_size
        self.current_block = defaultdict(list)
        self.dict_keyPointer = defaultdict()
        self.block_count = 0
        self.dict_count = 0
        self.doc_count = 0
        self.document_norms = []

        self.preprocessor = TextPreProcess()
        self.preprocessor.loadStopList()

    def build_index(self, documents):
        # Limpieza inicial
        if self.block_count > 0:
            for block in range(self.block_count):
                file_path = f'block_{block}.bin'
                if os.path.exists(file_path):
                    os.remove(file_path)
            self.__init__(self.block_size, self.dict_size)

        print(f"Construyendo índice con {len(documents)} documentos")
        self.doc_count = len(documents)
        
        for doc_id, document in enumerate(documents):
            # Procesar documento
            tokens = self.tokenize(document)
            term_freq = defaultdict(int)
            
            # Contar frecuencias
            for token in tokens:
                term_freq[token] += 1

            # Actualizar el índice
            for token, tf in term_freq.items():
                self.current_block[token].append((doc_id, tf))
                
                # Si el bloque actual para este término está lleno
                if len(self.current_block[token]) >= self.block_size:
                    self.flush_term(token)

                # Si el diccionario en memoria está lleno
                if len(self.current_block) >= self.dict_size:
                    self.flush_all_terms()

        # Flush final de términos pendientes
        if self.current_block:
            self.flush_all_terms()

    def flush_term(self, token):
        """Escribe un término específico a disco"""
        postings = self.current_block[token]
        
        # Crear nuevo bloque
        block_data = {
            'postings': postings,
            'next_block': -1  # -1 indica que es el último bloque
        }
        
        # Escribir bloque
        block_file = f'block_{self.block_count}.bin'
        with open(block_file, 'wb') as f:
            pickle.dump(block_data, f)
            
        # Actualizar punteros
        if token in self.dict_keyPointer:
            # Actualizar el puntero del bloque anterior
                current_pointer = self.dict_keyPointer[token]
                prev_block = self.read_block(current_pointer)
                
                # Navegar hasta el último bloque
                while prev_block['next_block'] != -1:
                    current_pointer = prev_block['next_block']
                    prev_block = self.read_block(current_pointer)

                # Actualizar el puntero del último bloque al nuevo bloque
                prev_block['next_block'] = self.block_count
                self.write_block(current_pointer, prev_block) 
        else:
            # Nuevo término
            self.dict_keyPointer[token] = self.block_count
            
        self.block_count += 1
        del self.current_block[token]

    def flush_all_terms(self):
        #Escribe todos los términos actuales a disco
        for token in list(self.current_block.keys()):
            self.flush_term(token)
            
        # Escribir diccionario
        dict_file = f'dict_{self.dict_count}.bin'
        with open(dict_file, 'wb') as f:
            #ordena el diccionario por clave
            pickle.dump(dict(sorted(self.dict_keyPointer.items())), f)
        self.dict_count += 1
        self.dict_keyPointer.clear()

    def read_block(self, block_num):
        #Lee un bloque específico del disco
        with open(f'block_{block_num}.bin', 'rb') as f:
            return pickle.load(f)

    def write_block(self, block_num, block_data):
        #Escribe un bloque específico al disco
        with open(f'block_{block_num}.bin', 'wb') as f:
            pickle.dump(block_data, f)

    
    def tokenize(self, text):
        return self.preprocessor.processText(text)

    def get_tfidf(self, term_freq, doc_freq):
        return np.log10(1 + term_freq) * np.log10(self.doc_count / doc_freq)

    def save_norms(self):
        with open('document_norms.bin', 'wb') as f:
            pickle.dump(self.document_norms, f)

    def load_norms(self):
        with open('document_norms.bin', 'rb') as f:
            self.document_norms = pickle.load(f)

    def debug_blocks(self):
        print("\n\nDEBUGGING INVERTED INDEX BLOCKS:")
        print("Buckets:")
        for block in range(self.block_count):
            with open(f'block_{block}.bin', 'rb') as f:
                block_data = pickle.load(f)
            print(f"Block {block}: {block_data}")
        print("\nDictionary:")
        for block in range(self.dict_count):
            with open(f'dict_{block}.bin', 'rb') as f:
                dict_data = pickle.load(f)
            print(f"Dict {block}: {dict_data}")

        #print(f"Document norms: {self.document_norms}")
