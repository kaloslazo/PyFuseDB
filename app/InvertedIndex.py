import pickle
import os
from collections import defaultdict
import numpy as np
from TextPreProcess import TextPreProcess
import math


class InvertedIndex:
    def __init__(self, block_size=1000, dict_size=100):
        self.block_size = block_size
        self.dict_size = dict_size
        self.current_block = defaultdict(list)
        self.dict_keyPointer = defaultdict(lambda: [0, 0])
        self.block_count = 0
        self.dict_count = 0
        self.doc_count = 0
        self.document_norms = []
        self.count_output = 0
        self.output = defaultdict()
        self.newListParameters = []

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
                    #print("????????????????",len(self.current_block))
                    #print("????????????????",len(self.dict_keyPointer))
                    self.flush_all_terms()

        # Flush final de términos pendientes
        if self.current_block:
            self.flush_all_terms()

        # Merge de bloques
        self.merge_blocks_old()

        #Calcular normas de documentos
        self.calculate_document_norms()
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
    
    def merge_lotes(self, P, Q, PSIZE, QSIZE):
        i = 0
        j = 0
        p = P
        q = Q

        # Leer diccionarios iniciales de p y q
        dict1 = self.read_dict(p)
        keys1 = list(dict1.keys())
        dict2 = self.read_dict(q) if Q < self.dict_count else {}
        
        keys2 = list(dict2.keys()) if dict2 else []

        initial_count = self.count_output
        # Bucle principal de mezcla
        while p < min(P + PSIZE, self.dict_count) and q < min(Q + PSIZE, self.dict_count) and i < len(keys1) and j < len(keys2):
            if keys1[i] == keys2[j]:
                if dict1[keys1[i]][0] > dict2[keys2[j]][0]:
                    self.output[keys1[i]] = dict1[keys1[i]]
                    self.concatenate_postings(self.output[keys1[i]][1], dict2[keys2[j]][1])
                    self.output[keys1[i]][0] += dict2[keys2[j]][0]
                else:
                    self.output[keys2[j]] = dict2[keys2[j]]
                    self.concatenate_postings(self.output[keys2[j]][1], dict1[keys1[i]][1])
                    self.output[keys2[j]][0] += dict1[keys1[i]][0]
                i += 1
                j += 1
            elif keys1[i] < keys2[j]:
                self.output[keys1[i]] = dict1[keys1[i]]
                i += 1
            else:
                self.output[keys2[j]] = dict2[keys2[j]]
                j += 1

            # Reiniciar y leer nuevos bloques si es necesario
            if i == len(keys1) and p + 1 < min(P + PSIZE, self.dict_count):
                p += 1
                dict1 = self.read_dict(p)
                keys1 = list(dict1.keys())
                i = 0

            if j == len(keys2) and q + 1 < min(Q + QSIZE, self.dict_count):
                q += 1
                dict2 = self.read_dict(q)
                keys2 = list(dict2.keys())
                j = 0

            # Escribir output al alcanzar el tamaño del diccionario
            if len(self.output) >= self.dict_size:
                self.write_temp_dict()
                self.output = defaultdict()

        # Procesar elementos restantes de la izquierda
        while p < min(P + PSIZE, self.dict_count) and i < len(keys1):
            self.output[keys1[i]] = dict1[keys1[i]]
            i += 1
            if i == len(keys1) and p + 1 < min(P + PSIZE, self.dict_count):
                p += 1
                dict1 = self.read_dict(p)
                keys1 = list(dict1.keys())
                i = 0
            if len(self.output) >= self.dict_size:
                self.write_temp_dict()
                self.output = defaultdict()

        # Procesar elementos restantes de la derecha
        while q < min(Q + QSIZE, self.dict_count) and j < len(keys2):
            self.output[keys2[j]] = dict2[keys2[j]]
            j += 1
            if j == len(keys2) and q + 1 < min(Q + QSIZE, self.dict_count):
                q += 1
                dict2 = self.read_dict(q)
                keys2 = list(dict2.keys())
                j = 0
            if len(self.output) >= self.dict_size:
                self.write_temp_dict()
                self.output = defaultdict()
        
        # Escribir output restante
        if self.output:
            self.write_temp_dict()
            self.output = defaultdict()


        # actualizar newListParameters
        salto = self.count_output - initial_count
        if len(self.newListParameters) == 0: #vacia
            self.newListParameters.append([initial_count, None, salto, None])
        elif self.newListParameters[-1][0]!=None and self.newListParameters[-1][1] != None: #ultima llena
            self.newListParameters.append([initial_count, None, salto, None])
        elif self.newListParameters[-1][0] != None and self.newListParameters[-1][1] == None: #ultima semi llena
            self.newListParameters[-1][1] = initial_count
            self.newListParameters[-1][3] = salto


    def merge_blocks(self):
        print("Antes de combinar bloques:")
        self.debug_blocks()
        print("Combinando bloques...")
        print("Número de diccionarios:", self.dict_count)
        size = 1
        while size < self.dict_count:
            print(f"Combinando lotes de tamaño {size}")
            for i in range(0, self.dict_count, 2*size):
                print(f"Combinando lotes {i} y {i+size}")
                self.merge_lotes(i, i+size, size)
            

            # eliminar diccionarios antiguos
            for i in range(0, self.dict_count):
                os.remove(f'dict_{i}.bin')
            
            # renombrar diccionarios temporales
            for i in range(self.count_output):
                os.rename(f'dict_temp_{i}.bin', f'dict_{i}.bin')
            self.dict_count = self.count_output
            self.count_output = 0

            self.debug_blocks()

            size *= 2

    def merge_blocks_old(self):
        listParameters = []
        for i in range(0, self.dict_count, 2):
            listParameters.append([i, i+1, 1, 1])
        
        print ("Antes de combinar bloques:")
        self.debug_blocks()
        print("Combinando bloques...")
        while (True):
            self.newListParameters = []
            for a, b, aSize, bSize in listParameters:
                self.merge_lotes(a, b, aSize, bSize)

            # asegurar que el último lote se cierre
            if self.newListParameters[-1][1] == None:
                self.newListParameters[-1][1] = self.count_output
                self.newListParameters[-1][3] = self.count_output + 1

            listParameters = self.newListParameters

            # eliminar diccionarios antiguos
            for i in range(0, self.dict_count):
                os.remove(f'dict_{i}.bin')
            
            # renombrar diccionarios temporales
            for i in range(self.count_output):
                os.rename(f'dict_temp_{i}.bin', f'dict_{i}.bin')
            self.dict_count = self.count_output
            self.count_output = 0

            self.debug_blocks()
            #print("ListParameters:", listParameters)
            #input("Press Enter to continue...")

            if len(listParameters) == 1 and listParameters[0][1] == self.dict_count:
                break

    
    def flush_term(self, token):
        """Escribe un término específico a disco"""
        postings = self.current_block[token]
        
        if len(postings) > 0:
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
                    current_pointer = self.dict_keyPointer[token][1]
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
                self.dict_keyPointer[token][1] = self.block_count
            
            self.dict_keyPointer[token][0] += len(postings)
                
            self.block_count += 1
            self.current_block[token] = []  # Limpiar el bloque actual

    def flush_all_terms(self):
        # Hacer flush de los términos
        #print("edrftgyhu7yhgtrfwsaz", len(self.dict_keyPointer))        
        for token in list(self.current_block.keys()):
            self.flush_term(token)
        
        # Escribir diccionario ordenado
        dict_file = f'dict_{self.dict_count}.bin'
        #print("edrftgyhu7yhgtrfwsaz", len(self.dict_keyPointer))
        with open(dict_file, 'wb') as f:
            pickle.dump(dict(sorted(self.dict_keyPointer.items())), f)
        
        self.dict_count += 1
        self.dict_keyPointer.clear()
        self.current_block.clear()  # Limpiar el bloque actual

    def read_block(self, block_num):
        #Lee un bloque específico del disco
        with open(f'block_{block_num}.bin', 'rb') as f:
            return pickle.load(f)

    def write_block(self, block_num, block_data):
        #Escribe un bloque específico al disco
        with open(f'block_{block_num}.bin', 'wb') as f:
            pickle.dump(block_data, f)

    def concatenate_postings(self, bucketNum1, bucketNum2):
        #print(f"Concatenando {bucketNum1} y {bucketNum2}")
        bucket1 = self.read_block(bucketNum1)
        bucket2 = self.read_block(bucketNum2)

        current_pointer = bucketNum1
        while bucket1['next_block'] != -1:
            current_pointer = bucket1['next_block']
            bucket1 = self.read_block(current_pointer)

        bucket1['next_block'] = bucketNum2
        self.write_block(current_pointer, bucket1) 

    def write_temp_dict(self):
        #Escribe un diccionario temporal al disco
        dict_file = f'dict_temp_{self.count_output}.bin'
        with open(dict_file, 'wb') as f:
            pickle.dump(dict(self.output.items()), f)
        self.output.clear()
        self.count_output += 1
        


    def read_dict(self, dict_num):
        #Lee un diccionario específico del disco
        with open(f'dict_{dict_num}.bin', 'rb') as f:
            return pickle.load(f)

    def tokenize(self, text):
        return self.preprocessor.processText(text)


    def calculate_document_norms(self):
        #Calcula la norma de cada documento
        self.document_norms = np.zeros(self.doc_count)

        for i in range(self.dict_count):
            with open(f'dict_{i}.bin', 'rb') as f:
                dict_data = pickle.load(f)
            
            for token, (df, block_num) in dict_data.items():
                while(True):
                    block = self.read_block(block_num)
                    for doc_id, tf in block['postings']:
                        self.document_norms[doc_id] += self.get_tfidf(tf, df) ** 2
                    if block['next_block'] == -1:
                        break
                    block_num = block['next_block']
        
        self.document_norms = np.sqrt(self.document_norms)


    def get_tfidf(self, term_freq, doc_freq):
        #print(f"Term freq: {term_freq}, Doc freq: {doc_freq}")
        if term_freq > 0:
            x = np.log10(1 + term_freq)
        else:
            x = 0

        y = np.log10(self.doc_count / doc_freq)
        return x * y

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
            print(f"Len_Dict {block}: {len(dict_data)}")
            print(f"Dict {block}: {dict_data}")

        #print(f"Document norms: {self.document_norms}")
