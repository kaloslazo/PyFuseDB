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
        self.current_block = defaultdict(list)
        self.dict_keyPointer = defaultdict(lambda: [0, 0])
        self.block_count = 0
        self.dict_count = 0
        self.doc_count = 0
        self.document_norms = None
        self.count_output = 0
        self.output = defaultdict()
        self.newListParameters = []
        
        # Asegurar que el directorio bin existe
        self.bin_path = os.path.join("app", "data", "bin")
        if not os.path.exists(self.bin_path):
            os.makedirs(self.bin_path)

        self.preprocessor = TextPreProcess()
        self.preprocessor.loadStopList()

    def build_index(self, documents):
        self.clear_files()

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
                    # print("????????????????",len(self.current_block))
                    # print("????????????????",len(self.dict_keyPointer))
                    self.flush_all_terms()

        # Flush final de términos pendientes
        if self.current_block:
            self.flush_all_terms()

        # Merge de bloques
        self.merge_blocks()

        # Calcular normas de documentos
        self.calculate_document_norms()
        self.save_norms()

    def binary_search(self, keys, term):
        low, high = 0, len(keys) - 1
        while low <= high:
            mid = (low + high) // 2
            if keys[mid] == term:
                return mid
            elif term < keys[mid]:
                high = mid - 1
            else:
                low = mid + 1
        return None

    def find_postings_list(self, term):
        low = 0
        high = self.dict_count - 1

        while low <= high:
            mid = (low + high) // 2
            dictionary = self.read_dict(mid)

            keys = dictionary.keys()

            index = self.binary_search(keys, term)

            if index is not None:
                doc_freq, block_num = dictionary[term]

                postings_list = []
                while True:
                    block_data = self.read_block(block_num)
                    postings_list.extend(block_data["postings"])

                    if block_data["next_block"] == -1:
                        break

                    block_num = block_data["next_block"]

                return postings_list

            if term < keys[0]:
                high = mid - 1
            else:
                low = mid + 1

        return []


    def search(self, query, top_k: int = 10):
        print(f"Buscando: '{query}'")

        scores = defaultdict(float)

        terms_dict = self.preprocessor.preprocess_query(query)
        query_tfidf = {}
        query_norm = 0

        for term, tf in terms_dict.items():
            # fetch postings list for token
            postings_list = self.fetch_postings_list(term)

            if not postings_list:
                continue # term not in disk

            df = len(postings_list)
            query_tfidf[term] = self.get_tfidf(tf, df)
            query_norm += query_tfidf[term] ** 2

            # print(f"postings list for {term}: {postings_list}")
            # print(f"query tf for {term}: {tf}")

            for (doc_id, doc_tf) in postings_list:
                doc_tfidf = self.get_tfidf(doc_tf, df)
                scores[doc_id] += doc_tfidf * query_tfidf[term]
        
        if query_norm != 0:
            if self.document_norms is None:
                self.load_norms()

            query_norm = np.sqrt(query_norm)
            print(f"query_norm = {query_norm}")
            for doc_id in scores.keys():
                scores[doc_id] /= (self.document_norms[doc_id] * query_norm)

        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print(f"Número de resultados encontrados: {len(scores)}")
        return scores[:top_k]


    def fetch_postings_list(self, term):
        low = 0
        high = self.dict_count - 1

        # binary search en los archivos de diccionario
        # hasta encontrar la lista de postings 
        while low <= high:
            mid = (low + high) // 2
            dictionary = self.read_dict(mid)

            # found
            if term in dictionary:
                doc_freq, block_num = dictionary[term]

                postings_list = []
                while True:
                    block_data = self.read_block(block_num)
                    postings_list.extend(block_data['postings'])
                    block_num = block_data['next_block']

                    if block_num == -1:
                        break

                return postings_list
            
            # continuar la busqueda
            first_key = next(iter(dictionary.keys()))
            if term < first_key:
                high = mid - 1
            else:
                low = mid + 1

        # not found
        return []


    def merge_lotes(self, P, Q, PSIZE, QSIZE):
        i = 0
        j = 0
        p = P
        q = Q

        # Leer diccionarios iniciales de p y q
        dict1 = self.read_dict(p) if p < self.dict_count else {}
        keys1 = list(dict1.keys())
        dict2 = self.read_dict(q) if Q < self.dict_count else {}
        keys2 = list(dict2.keys()) if dict2 else []

        initial_count = self.count_output

        # Bucle principal de mezcla
        while (
            p < min(P + PSIZE, self.dict_count)
            and q < min(Q + PSIZE, self.dict_count)
            and i < len(keys1)
            and j < len(keys2)
        ):
            if keys1[i] == keys2[j]:
                if dict1[keys1[i]][0] > dict2[keys2[j]][0]:
                    self.output[keys1[i]] = dict1[keys1[i]]
                    self.concatenate_postings(
                        self.output[keys1[i]][1], dict2[keys2[j]][1]
                    )
                    self.output[keys1[i]][0] += dict2[keys2[j]][0]
                else:
                    self.output[keys2[j]] = dict2[keys2[j]]
                    self.concatenate_postings(
                        self.output[keys2[j]][1], dict1[keys1[i]][1]
                    )
                    self.output[keys2[j]][0] += dict1[keys1[i]][0]
                i += 1
                j += 1
            elif keys1[i] < keys2[j]:
                self.output[keys1[i]] = dict1[keys1[i]]
                i += 1
            else:
                self.output[keys2[j]] = dict2[keys2[j]]
                j += 1

            if len(self.output) >= self.dict_size:
                self.write_temp_dict()

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
        if len(self.newListParameters) == 0:  # vacia
            self.newListParameters.append([initial_count, None, salto, None])

        elif (
            self.newListParameters[-1][0] != None
            and self.newListParameters[-1][1] != None
        ):  # ultima llena
            self.newListParameters.append([initial_count, None, salto, None])
        elif (
            self.newListParameters[-1][0] != None
            and self.newListParameters[-1][1] == None
        ):  # ultima semi llena
            self.newListParameters[-1][1] = initial_count
            self.newListParameters[-1][3] = salto

    def merge_blocks(self):
        listParameters = []
        for i in range(0, self.dict_count, 2):
            listParameters.append([i, i + 1, 1, 1])

        print("Antes de combinar bloques:")
        self.debug_blocks()
        print("Combinando bloques...")
        while True:
            self.newListParameters = []
            for a, b, aSize, bSize in listParameters:
                self.merge_lotes(a, b, aSize, bSize)

            # asegurar que el último lote se cierre
            if self.newListParameters and self.newListParameters[-1][1] is None:
                self.newListParameters[-1][1] = self.count_output
                self.newListParameters[-1][3] = self.count_output + 1

            listParameters = self.newListParameters

            # eliminar diccionarios antiguos
            for i in range(0, self.dict_count):
                old_dict = os.path.join(self.bin_path, f"dict_{i}.bin")
                if os.path.exists(old_dict):
                    os.remove(old_dict)

            # renombrar diccionarios temporales
            for i in range(self.count_output):
                temp_dict = os.path.join(self.bin_path, f"dict_temp_{i}.bin")
                new_dict = os.path.join(self.bin_path, f"dict_{i}.bin")
                if os.path.exists(temp_dict):
                    os.rename(temp_dict, new_dict)

            self.dict_count = self.count_output
            self.count_output = 0

            self.debug_blocks()

            if len(listParameters) == 1 and listParameters[0][1] == self.dict_count:
                break

    def flush_term(self, token):
        """Escribe un término específico a disco"""
        postings = self.current_block[token]
        if len(postings) > 0:
            # Crear nuevo bloque
            block_data = {
                "postings": postings,
                "next_block": -1,  # -1 indica que es el último bloque
            }

            block_file = os.path.join(self.bin_path, f"block_{self.block_count}.bin")
            with open(block_file, "wb") as f:
                pickle.dump(block_data, f)

            # Actualizar punteros
            if token in self.dict_keyPointer:
                # Actualizar el puntero del bloque anterior
                current_pointer = self.dict_keyPointer[token][1]
                prev_block = self.read_block(current_pointer)

                # Navegar hasta el último bloque
                while prev_block["next_block"] != -1:
                    current_pointer = prev_block["next_block"]
                    prev_block = self.read_block(current_pointer)

                # Actualizar el puntero del último bloque al nuevo bloque
                prev_block["next_block"] = self.block_count
                self.write_block(current_pointer, prev_block)
            else:
                # Nuevo término
                self.dict_keyPointer[token][1] = self.block_count

            self.dict_keyPointer[token][0] += len(postings)
            self.block_count += 1
            self.current_block[token] = []  # Limpiar el bloque actual

    def flush_all_terms(self):
        """Escribe un término específico a disco"""
        for token in list(self.current_block.keys()):
            self.flush_term(token)

        # Escribir diccionario ordenado usando la ruta correcta
        dict_file = os.path.join(self.bin_path, f"dict_{self.dict_count}.bin")
        with open(dict_file, "wb") as f:
            pickle.dump(dict(sorted(self.dict_keyPointer.items())), f)

        self.dict_count += 1
        self.dict_keyPointer.clear()
        self.current_block.clear()

    def read_block(self, block_num):
        with open(os.path.join(self.bin_path, f"block_{block_num}.bin"), "rb") as f:
            return pickle.load(f)

    def write_block(self, block_num, block_data):
        with open(os.path.join(self.bin_path, f"block_{block_num}.bin"), "wb") as f:
            pickle.dump(block_data, f)

    def concatenate_postings(self, bucketNum1, bucketNum2):
        # print(f"Concatenando {bucketNum1} y {bucketNum2}")
        bucket1 = self.read_block(bucketNum1)

        current_pointer = bucketNum1
        while bucket1["next_block"] != -1:
            current_pointer = bucket1["next_block"]
            bucket1 = self.read_block(current_pointer)

        bucket1["next_block"] = bucketNum2
        self.write_block(current_pointer, bucket1)

    def write_temp_dict(self):
        dict_file = os.path.join(self.bin_path, f"dict_temp_{self.count_output}.bin")
        with open(dict_file, "wb") as f:
            pickle.dump(dict(self.output.items()), f)
        self.output.clear()
        self.count_output += 1

    def read_dict(self, dict_num):
        dict_file = os.path.join(self.bin_path, f"dict_{dict_num}.bin")
        if os.path.exists(dict_file):
            with open(dict_file, "rb") as f:
                return pickle.load(f)
        return {}

    def save_norms(self):
        with open(os.path.join(self.bin_path, "document_norms.bin"), "wb") as f:
            pickle.dump(self.document_norms, f)

    def load_norms(self):
        with open(os.path.join(self.bin_path, "document_norms.bin"), "rb") as f:
            self.document_norms = pickle.load(f)


    def tokenize(self, text):
        return self.preprocessor.processText(text)

    def calculate_document_norms(self):
        # Calcula la norma de cada documento
        self.document_norms = np.zeros(self.doc_count)

        for i in range(self.dict_count):
            with open(f"dict_{i}.bin", "rb") as f:
                dict_data = pickle.load(f)

            for token, (df, block_num) in dict_data.items():
                while True:
                    block = self.read_block(block_num)
                    for doc_id, tf in block["postings"]:
                        self.document_norms[doc_id] += self.get_tfidf(tf, df) ** 2
                    if block["next_block"] == -1:
                        break
                    block_num = block["next_block"]

        self.document_norms = np.sqrt(self.document_norms)

    def get_tfidf(self, term_freq, doc_freq):
        # print(f"Term freq: {term_freq}, Doc freq: {doc_freq}")
        if term_freq > 0:
            tf = np.log10(1 + term_freq)
        else:
            tf = 0

        idf = np.log10(self.doc_count / doc_freq)
        return tf * idf

    def save_norms(self):
        with open("document_norms.bin", "wb") as f:
            pickle.dump(self.document_norms, f)

    def load_norms(self):
        with open("document_norms.bin", "rb") as f:
            self.document_norms = pickle.load(f)

    def clear_files(self):
        for pattern in ["block_*.bin", "dict_*.bin", "dict_temp_*.bin", "document_norms.bin"]:
            for file in glob.glob(os.path.join(self.bin_path, pattern)):
                try:
                    os.remove(file)
                except OSError as e:
                    print(f"Error al eliminar {file}: {e}")
    
    def debug_blocks(self):
        print("\n\nDEBUGGING INVERTED INDEX BLOCKS:")

        print("Buckets:")
        for block in range(self.block_count):
            block_path = os.path.join(self.bin_path, f"block_{block}.bin")
            if os.path.exists(block_path):
                with open(block_path, "rb") as f:
                    block_data = pickle.load(f)
                    print(f"Block {block}: {block_data}")

        print("\nDictionary:")
        for block in range(self.dict_count):
            dict_path = os.path.join(self.bin_path, f"dict_{block}.bin")
            if os.path.exists(dict_path):
                with open(dict_path, "rb") as f:
                    dict_data = pickle.load(f)
                    print(f"Dict {block}: {dict_data}")

        if self.document_norms is not None:
            print(f"Document norms: {self.document_norms}")