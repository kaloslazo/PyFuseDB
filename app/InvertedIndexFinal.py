import heapq
import json
import os
import struct
from TextPreProcess import TextPreProcess
from collections import defaultdict

class InvertedIndexFinal:
    """
    implementacion optimizada de spimi para manejar colecciones de documentos
    minimizando uso de ram y optimizando operaciones en disco
    """
    
    def __init__(self, block_size=1000, dict_size=100000):
        """
        inicializa el indice invertido
            block_size (int): tamaño del bloque para posting lists
            dict_size (int): tamaño maximo del diccionario en memoria
        """
        # parametros principales
        self.block_size = block_size
        self.dict_size = dict_size
        self.doc_count = 0
        
        # estructuras temporales en memoria
        self.current_block = defaultdict(list)  # {term: [(doc_id, freq)]}
        self.term_positions = {}  # {term: (block_pos, freq)}
        
        # rutas y archivos
        self.bin_path = os.path.join("app", "data", "bin")
        self.posting_file = None  # almacena posting lists
        self.term_file = None  # almacena vocabulario
        self.doc_norms_file = None  # almacena normas de documentos
        self.term_positions_file = None  # almacena posiciones
        
        # contadores y utilidades
        self.block_counter = 0
        self.temp_blocks = []
        
        # preprocesamiento de texto
        self.preprocessor = TextPreProcess()
        self.preprocessor.loadStopList()
        
    def _initialize_files(self):
        """inicializa archivos binarios para el indice"""
        if not os.path.exists(self.bin_path):
            os.makedirs(self.bin_path)
            
        # crear directorio para bloque actual
        block_path = os.path.join(self.bin_path, f"block_{self.block_counter}")
        if not os.path.exists(block_path):
            os.makedirs(block_path)
            
        # abrir archivos del bloque
        self.posting_file = open(os.path.join(block_path, "postings.bin"), "wb+", buffering=8192)
        self.term_file = open(os.path.join(block_path, "terms.bin"), "wb+", buffering=8192)
        
        # abrir archivos finales
        self.term_positions_file = open(os.path.join(self.bin_path, "term_positions.bin"), "wb+")
        self.doc_norms_file = open(os.path.join(self.bin_path, "norms.bin"), "wb+")
        
        self.temp_blocks.append(block_path)
       
    def _verify_index_structure(self):
        """verifica estructura del indice y muestra estadisticas"""
        print("\nverificando estructura del indice:")
        
        # verificar bloques temporales
        print("\n1. bloques temporales:")
        for block_path in self.temp_blocks:
            print(f"\nbloque: {os.path.basename(block_path)}")
            for file_name in ["postings.bin", "terms.bin"]:
                file_path = os.path.join(block_path, file_name)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  ✓ {file_name}: {size} bytes")
                else:
                    print(f"  ✗ {file_name}: no encontrado")
        
        # verificar archivos finales
        print("\n2. archivos finales:")
        final_files = {
            "term_positions.bin": self.term_positions_file,
            "norms.bin": self.doc_norms_file
        }
        
        for name, file in final_files.items():
            path = os.path.join(self.bin_path, name)
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"  ✓ {name}: {size} bytes")
            else:
                print(f"  ✗ {name}: no encontrado")

        # mostrar muestra de terminos
        print("\n3. muestra de terminos:")
        for block_path in self.temp_blocks:
            term_file = os.path.join(block_path, "terms.bin")
            posting_file = os.path.join(block_path, "postings.bin")
            
            with open(term_file, "rb") as tf, open(posting_file, "rb") as pf:
                try:
                    for i in range(3):
                        term_len = struct.unpack('I', tf.read(4))[0]
                        term = tf.read(term_len).decode('utf-8')
                        position, df = struct.unpack('QI', tf.read(12))
                        
                        print(f"\ntermino en bloque {os.path.basename(block_path)}:")
                        print(f"  - palabra: '{term}'")
                        print(f"  - frecuencia: {df}")
                        
                        pf.seek(position)
                        postings = []
                        for _ in range(df):
                            doc_id, freq = struct.unpack('II', pf.read(8))
                            postings.append((doc_id, freq))
                        print(f"  - postings: {postings[:3]}...")
                        
                except Exception as e:
                    print(f"  ⚠️ error leyendo termino: {str(e)}")
                
    def _detal_decode(self, compressed_postings):
        """decodifica postings lists usando delta encoding"""
        if not compressed_postings: 
            return []
        
        decoded = [compressed_postings[0]]
        current_id = compressed_postings[0][0]
        
        for delta, freq in compressed_postings[1:]:
            current_id += delta
            decoded.append((current_id, freq))
            
        return decoded
        
    def _delta_encode(self, postings):
        """codifica postings lists usando delta encoding"""
        if not postings: 
            return []
        
        sorted_postings = sorted(postings)
        encoded = [(sorted_postings[0][0], sorted_postings[0][1])]
        
        for i in range(1, len(sorted_postings)):
            doc_id_delta = sorted_postings[i][0] - sorted_postings[i-1][0]
            encoded.append((doc_id_delta, sorted_postings[i][1]))
        
        return encoded
        
    def _process_document(self, doc_id, document):
        """procesa un documento y actualiza el bloque actual"""
        terms = self.preprocessor.processText(document)
        
        term_frequencies = {}
        for term in terms:
            term_frequencies[term] = term_frequencies.get(term, 0) + 1
        
        for term, freq in term_frequencies.items():
            self.current_block[term].append((doc_id, freq))
    
    def read_postings(self, position, df):
        """lee posting list desde disco con validacion"""
        try:
            self.posting_file.seek(position)
            
            count_bytes = self.posting_file.read(4)
            if not count_bytes:
                print(f"⚠️ no se pudo leer contador en posicion {position}")
                return []
                
            posting_count = struct.unpack('I', count_bytes)[0]
            if posting_count != df:
                print(f"⚠️ discrepancia: esperado {df}, leido {posting_count}")
            
            postings = []
            for _ in range(posting_count):
                try:
                    doc_id, freq = struct.unpack('II', self.posting_file.read(8))
                    postings.append((doc_id, freq))
                except struct.error:
                    print(f"⚠️ error leyendo posting #{len(postings)+1}")
                    break
                    
            return postings
            
        except Exception as e:
            print(f"error leyendo postings en {position}: {str(e)}")
            return []
            
    def _write_term_position(self, term, position, freq):
        """escribe posicion de termino a disco"""
        term_padded = term.ljust(100).encode('utf-8')
        self.term_positions_file.write(term_padded)
        self.term_positions_file.write(struct.pack('Q', position))
        self.term_positions_file.write(struct.pack('I', freq))
    
    def _write_block(self, block_number):
        """escribe bloque del indice a disco"""
        if len(self.current_block) >= self.dict_size:
            print(f"escribiendo bloque {block_number} ({len(self.current_block)} terminos)")
            sorted_terms = sorted(self.current_block.keys())
            
            for term in sorted_terms:
                postings = sorted(self.current_block[term])
                self._write_posting_list(term, postings)
                
            self.posting_file.close()
            self.term_file.close()
            
            self.block_counter += 1
            self._initialize_files()
            self.current_block.clear()
    
    def _write_posting_list(self, term, postings):
        """escribe posting list con conteo de frecuencias"""
        position = self.posting_file.tell()
        
        doc_freq = {}
        for doc_id, freq in postings:
            doc_freq[doc_id] = doc_freq.get(doc_id, 0) + freq
        
        combined_postings = sorted(doc_freq.items())
        
        self.posting_file.write(struct.pack('I', len(combined_postings)))
        
        for doc_id, total_freq in combined_postings:
            self.posting_file.write(struct.pack('II', doc_id, total_freq))
        
        term_bytes = term.encode('utf-8')
        self.term_file.write(struct.pack('I', len(term_bytes)))
        self.term_file.write(term_bytes)
        self.term_file.write(struct.pack('QI', position, len(combined_postings)))
            
    def _write_merged_postings(self, final_index, term, postings):
        """escribe postings combinados con acumulacion de frecuencias"""
        doc_frequencies = {}
        for doc_id, freq in postings:
            doc_frequencies[doc_id] = doc_frequencies.get(doc_id, 0) + freq
        
        combined_postings = sorted(doc_frequencies.items())
        
        term_bytes = term.encode('utf-8')
        final_index.write(struct.pack('I', len(term_bytes)))
        final_index.write(term_bytes)
        final_index.write(struct.pack('I', len(combined_postings)))
        
        for doc_id, total_freq in combined_postings:
            final_index.write(struct.pack('II', doc_id, total_freq))
        
    def build_index(self, documents):
        """construye indice invertido procesando documentos"""
        print(f"construyendo indice para {len(documents)} documentos")
        self._initialize_files()
        self.doc_count = len(documents)
        block_number = 0
        
        for doc_id, document in enumerate(documents):
            if doc_id % 100 == 0: 
                print(f"procesando documento {doc_id}/{len(documents)}")
            
            self._process_document(doc_id, document)
            
            if len(self.current_block) >= self.dict_size:
                self._write_block(block_number)
                block_number += 1
        
        if self.current_block:
            self._write_block(block_number)
            
    def merge_blocks(self):
        """fusiona bloques con avance correcto de lectura"""
        print("\niniciando fusion de bloques...")
        
        block_readers = []
        heap = []
        
        for block_num, block_path in enumerate(self.temp_blocks):
            try:
                reader = self._BlockReader(block_path, self.bin_path)
                term_info = reader.peek()
                
                if term_info:
                    term, position, df = term_info
                    print(f"  ✓ bloque {os.path.basename(block_path)}: '{term}' (df={df})")
                    block_readers.append(reader)
                    heapq.heappush(heap, (term, block_num, position, df))
                    reader.next()
                else:
                    print(f"  ✗ bloque {os.path.basename(block_path)} sin terminos")
                    reader.term_file.close()
                    reader.posting_file.close()
                    
            except Exception as e:
                print(f"  ⚠️ error en bloque {block_path}: {str(e)}")
        
        with open(os.path.join(self.bin_path, "final_index.bin"), "wb") as final_index:
            current_term = None
            current_postings = []
            terms_merged = 0
            
            while heap:
                term, block_id, position, df = heapq.heappop(heap)
                
                if current_term and current_term != term:
                    print(f"  ✓ mergeando '{current_term}' ({len(current_postings)} postings)")
                    self._write_merged_postings(final_index, current_term, current_postings)
                    terms_merged += 1
                    current_postings = []
                
                current_term = term
                reader = block_readers[block_id]
                current_postings.extend(reader.read_postings(position, df))
                
                term_info = reader.peek()
                if term_info:
                    next_term, next_pos, next_df = term_info
                    heapq.heappush(heap, (next_term, block_id, next_pos, next_df))
                    reader.next()
            
            if current_term and current_postings:
                print(f"  ✓ mergeando '{current_term}' ({len(current_postings)} postings)")
                self._write_merged_postings(final_index, current_term, current_postings)
                terms_merged += 1
        
        print(f"\nmerge completado: {terms_merged} terminos procesados")

    def search(self, query, top_k=5):
        """busca en el indice invertido y retorna documentos relevantes"""
        with open(os.path.join(self.bin_path, "final_index.bin"), "rb") as f:
            term = query.lower()
            term_bytes = term.encode('utf-8')
            term_len = len(term_bytes)
            
            f.seek(0)
            while True:
                try:
                    term_len_read = struct.unpack('I', f.read(4))[0]
                    term_read = f.read(term_len_read).decode('utf-8')
                    df = struct.unpack('I', f.read(4))[0]
                    
                    if term_read == term:
                        postings = []
                        for _ in range(df):
                            doc_id, freq = struct.unpack('II', f.read(8))
                            postings.append((doc_id, freq))
                        return sorted(postings, key=lambda x: (-x[1], x[0]))
                    else:
                        f.seek(df * 8, 1)
                            
                except struct.error:
                    break
                except Exception as e:
                    print(f"error en busqueda: {str(e)}")
                    break
                
        return []

    class _BlockReader:
        def __init__(self, block_path, base_path):
            self.term_file = open(os.path.join(block_path, "terms.bin"), "rb")
            self.posting_file = open(os.path.join(block_path, "postings.bin"), "rb")
            self.current_position = 0
            self.file_size = os.path.getsize(os.path.join(block_path, "terms.bin"))
            
        def peek(self):
            """intenta leer siguiente termino sin avanzar"""
            if self.current_position >= self.file_size:
                return None
                
            try:
                self.term_file.seek(self.current_position)
                
                remaining_bytes = self.file_size - self.current_position
                if remaining_bytes < 4:
                    return None
                    
                term_len_bytes = self.term_file.read(4)
                term_len = struct.unpack('I', term_len_bytes)[0]
                
                if remaining_bytes < (4 + term_len + 12):
                    return None
                    
                term = self.term_file.read(term_len).decode('utf-8')
                position, df = struct.unpack('QI', self.term_file.read(12))
                
                self.term_file.seek(self.current_position)
                return term, position, df
                
            except Exception as e:
                print(f"error en peek (posicion {self.current_position}): {str(e)}")
                return None

        def read_postings(self, position, df):
            """lee postings con manejo de frecuencias"""
            try:
                self.posting_file.seek(position)
                posting_count = struct.unpack('I', self.posting_file.read(4))[0]
                
                postings = []
                for _ in range(posting_count):
                    doc_id, freq = struct.unpack('II', self.posting_file.read(8))
                    postings.append((doc_id, freq))
                    
                return postings
                
            except Exception as e:
                print(f"error leyendo postings: {str(e)}")
                return []

        def next(self):
            """avanza al siguiente termino"""
            if self.current_position >= self.file_size:
                return None
                
            try:
                self.term_file.seek(self.current_position)
                term_len = struct.unpack('I', self.term_file.read(4))[0]
                self.term_file.read(term_len)
                self.term_file.read(12)
                
                self.current_position = self.term_file.tell()
                return True
                
            except Exception as e:
                print(f"error en next: {str(e)}")
                return None