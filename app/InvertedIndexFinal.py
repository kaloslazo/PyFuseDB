import heapq
import json
import os
import struct
from TextPreProcess import TextPreProcess
from collections import defaultdict

class InvertedIndexFinal:
    """
    implementacion optimizada de spimi, maneja colecciones de documentos minimizando el uso de ram
    y optimizando las operaciones en disco !
    """
    
    def __init__(self, block_size=1000, dict_size=100000):
        """
        - block_size: tamaño del bloque para posting lists
        - dict_size: tamaño maximo del diccionario en memoria
        - doct_count: numero de documentos
        """
        self.block_size = block_size
        self.dict_size = dict_size
        self.doc_count = 0
        
        # memoria temporal (utilidades)
        self.current_block = defaultdict(list) # bloque con la forma {term: [(doc_id, freq en doc_id)]}
        self.term_positions = {} # tiene la forma {term: (posicion_bloque, freq en doc)}
        self.bin_path = os.path.join("app", "data", "bin")
        
        # archivos binarios temporales
        self.posting_file = None # "guarda las lista de documentos en las que aparece un termino y su frecuencia"
        self.term_file = None # "guarda vocabulario y donde encontrar posting list"
        self.doc_norms_file = None
        self.term_positions_file = None # archivo para guardar posiciones
        
        # utilidades
        self.block_counter = 0
        self.temp_blocks = []
        
        # preproucesamiento
        self.preprocessor = TextPreProcess()
        self.preprocessor.loadStopList()
        
    def _initialize_files(self):
        """
        Inicializa los archivos binarios para el índice.
        Ahora mantiene un registro de los archivos temporales y finales.
        """
        if not os.path.exists(self.bin_path):
            os.makedirs(self.bin_path)
            
        # Archivos temporales para bloques
        block_path = os.path.join(self.bin_path, f"block_{self.block_counter}")
        if not os.path.exists(block_path):
            os.makedirs(block_path)
            
        # Archivos para el bloque actual
        self.posting_file = open(os.path.join(block_path, "postings.bin"), "wb+", buffering=8192)
        self.term_file = open(os.path.join(block_path, "terms.bin"), "wb+", buffering=8192)
        
        # Archivos finales (se usarán después del merge)
        self.term_positions_file = open(os.path.join(self.bin_path, "term_positions.bin"), "wb+")
        self.doc_norms_file = open(os.path.join(self.bin_path, "norms.bin"), "wb+")
        
        self.temp_blocks.append(block_path)
       
    def _verify_index_structure(self):
        """
        Verifica la estructura del índice, incluyendo bloques temporales y archivos finales.
        """
        print("\nVerificando estructura del índice:")
        
        # Verificar bloques temporales
        print("\n1. Bloques temporales:")
        for block_path in self.temp_blocks:
            print(f"\nBloque: {os.path.basename(block_path)}")
            for file_name in ["postings.bin", "terms.bin"]:
                file_path = os.path.join(block_path, file_name)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  ✓ {file_name}: {size} bytes")
                else:
                    print(f"  ✗ {file_name}: no encontrado")
        
        # Verificar archivos finales
        print("\n2. Archivos finales:")
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

        # Verificar contenido de los bloques
        print("\n3. Muestra de términos:")
        for block_path in self.temp_blocks:
            term_file = os.path.join(block_path, "terms.bin")
            posting_file = os.path.join(block_path, "postings.bin")
            
            with open(term_file, "rb") as tf, open(posting_file, "rb") as pf:
                try:
                    # Leer los primeros términos del bloque
                    for i in range(3):
                        term_len = struct.unpack('I', tf.read(4))[0]
                        term = tf.read(term_len).decode('utf-8')
                        position, df = struct.unpack('QI', tf.read(12))
                        
                        print(f"\nTérmino en bloque {os.path.basename(block_path)}:")
                        print(f"  - Palabra: '{term}'")
                        print(f"  - Frecuencia: {df}")
                        
                        # Leer posting list
                        pf.seek(position)
                        postings = []
                        for _ in range(df):
                            doc_id, freq = struct.unpack('II', pf.read(8))
                            postings.append((doc_id, freq))
                        print(f"  - Postings: {postings[:3]}...")
                        
                except Exception as e:
                    print(f"  ⚠️ Error leyendo término: {str(e)}")
                
    def _detal_decode(self, compressed_postings):
        """
        decoding delta para postings lists
        """
        if not compressed_postings: return []
        
        decoded = [compressed_postings[0]]
        current_id = compressed_postings[0][0]
        
        for delta, freq in compressed_postings[1:]:
            current_id += delta
            decoded.append((current_id, freq))
            
        return decoded
        
    def _delta_encode(self, postings):
        """
        implementacion delta para postings lists
        """
        if not postings: return []
        
        # ordenar postings por doc_id
        sorted_postings = sorted(postings)
        encoded = [(sorted_postings[0][0], sorted_postings[0][1])]
        
        # siguientes terminos se guarda la diferencia
        for i in range(1, len(sorted_postings)):
            doc_id_delta = sorted_postings[i][0] - sorted_postings[i-1][0]
            encoded.append((doc_id_delta, sorted_postings[i][1]))
        
        return encoded
        
    def _process_document(self, doc_id, document):
        """
        procesamos cada documento individualmente y actualizamos el bloque actual.
        """
        # terminos procesados
        terms = self.preprocessor.processText(document)
        
        # lista de términos en un diccionario de frecuencias
        term_frequencies = {}
        for term in terms:
            if term in term_frequencies:
                term_frequencies[term] += 1
            else:
                term_frequencies[term] = 1
        
        # para cada término del documento añadimos la tupla (doc_id, freq) al bloque actual
        for term, freq in term_frequencies.items():
            self.current_block[term].append((doc_id, freq))
    
    def read_postings(self, position, df):
        """
        Reads a posting list from disk with proper validation:
        1. Seek to correct position
        2. Read posting count
        3. Read each posting pair (doc_id, freq)
        """
        try:
            self.posting_file.seek(position)
            
            # Read number of postings
            count_bytes = self.posting_file.read(4)
            if not count_bytes:
                print(f"⚠️ No se pudo leer el contador de postings en posición {position}")
                return []
                
            posting_count = struct.unpack('I', count_bytes)[0]
            if posting_count != df:
                print(f"⚠️ Discrepancia en conteo: esperado {df}, leído {posting_count}")
            
            # Read all postings
            postings = []
            for _ in range(posting_count):
                try:
                    doc_id, freq = struct.unpack('II', self.posting_file.read(8))
                    postings.append((doc_id, freq))
                except struct.error:
                    print(f"⚠️ Error leyendo posting #{len(postings)+1}")
                    break
                    
            return postings
            
        except Exception as e:
            print(f"Error leyendo postings en posición {position}: {str(e)}")
            return []
            
    def _write_term_position(self, term, position, freq):
        """
        escribe la posicion de un termino a disco
        """
        term_padded = term.ljust(100).encode('utf-8')
        self.term_positions_file.write(term_padded)
        self.term_positions_file.write(struct.pack('Q', position))
        self.term_positions_file.write(struct.pack('I', freq))
    
    def _write_block(self, block_number):
        """
        escribe un bloque del indice en disco
        """
        if len(self.current_block) >= self.dict_size:
            print(f"Escribiendo bloque {block_number} con {len(self.current_block)} términos")
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
        """
        Writes a posting list with proper frequency counting
        """
        position = self.posting_file.tell()
        
        # First, combine frequencies for the same document
        doc_freq = {}
        for doc_id, freq in postings:
            doc_freq[doc_id] = doc_freq.get(doc_id, 0) + freq
        
        # Convert back to sorted list
        combined_postings = sorted(doc_freq.items())
        
        # Write number of unique documents
        self.posting_file.write(struct.pack('I', len(combined_postings)))
        
        # Write each posting with its total frequency
        for doc_id, total_freq in combined_postings:
            self.posting_file.write(struct.pack('II', doc_id, total_freq))
        
        # Write term entry with correct document frequency
        term_bytes = term.encode('utf-8')
        self.term_file.write(struct.pack('I', len(term_bytes)))
        self.term_file.write(term_bytes)
        self.term_file.write(struct.pack('QI', position, len(combined_postings)))
            
    def _write_merged_postings(self, final_index, term, postings):
        """
        Writes merged postings with proper frequency accumulation:
        1. Combine frequencies for same doc_id
        2. Write term information
        3. Write combined posting list
        """
        # Combine frequencies per document
        doc_frequencies = {}
        for doc_id, freq in postings:
            doc_frequencies[doc_id] = doc_frequencies.get(doc_id, 0) + freq
        
        # Convert to sorted list of postings
        combined_postings = sorted(doc_frequencies.items())
        
        # Write term information
        term_bytes = term.encode('utf-8')
        final_index.write(struct.pack('I', len(term_bytes)))
        final_index.write(term_bytes)
        
        # Write posting count
        final_index.write(struct.pack('I', len(combined_postings)))
        
        # Write combined postings
        for doc_id, total_freq in combined_postings:
            final_index.write(struct.pack('II', doc_id, total_freq))
        
    def build_index(self, documents):
        """
        Construye el índice invertido procesando todos los documentos.
        Escribe a disco cuando el bloque alcanza el tamaño máximo.
        """
        print(f"Construyendo índice para {len(documents)} documentos")
        self._initialize_files()
        self.doc_count = len(documents)
        block_number = 0
        
        for doc_id, document in enumerate(documents):
            if doc_id % 100 == 0: 
                print(f"Procesando documento {doc_id}/{len(documents)}")
            
            self._process_document(doc_id, document)
            
            # Escribir bloque si alcanza el tamaño máximo
            if len(self.current_block) >= self.dict_size:
                self._write_block(block_number)
                block_number += 1
        
        # Escribir último bloque si quedó algo
        if self.current_block:
            # Forzar escritura del último bloque aunque no alcance el tamaño
            self._write_block(block_number)
            
    def merge_blocks(self):
        """
        Fusión de bloques con avance correcto de lectura.
        """
        print("\nIniciando fusión de bloques...")
        
        # Inicializar readers
        block_readers = []
        heap = []
        
        for block_num, block_path in enumerate(self.temp_blocks):
            try:
                reader = self._BlockReader(block_path, self.bin_path)
                term_info = reader.peek()
                
                if term_info:
                    term, position, df = term_info
                    print(f"  ✓ Bloque {os.path.basename(block_path)}: '{term}' (df={df})")
                    block_readers.append(reader)
                    heapq.heappush(heap, (term, block_num, position, df))
                    # Avanzar al siguiente término
                    reader.next()
                else:
                    print(f"  ✗ Bloque {os.path.basename(block_path)} sin términos")
                    reader.term_file.close()
                    reader.posting_file.close()
                    
            except Exception as e:
                print(f"  ⚠️ Error procesando bloque {block_path}: {str(e)}")
        
        # Proceso de merge
        with open(os.path.join(self.bin_path, "final_index.bin"), "wb") as final_index:
            current_term = None
            current_postings = []
            terms_merged = 0
            
            while heap:
                term, block_id, position, df = heapq.heappop(heap)
                
                # Si es un nuevo término, escribir el anterior
                if current_term and current_term != term:
                    print(f"  ✓ Mergeando '{current_term}' ({len(current_postings)} postings)")
                    self._write_merged_postings(final_index, current_term, current_postings)
                    terms_merged += 1
                    current_postings = []
                
                current_term = term
                reader = block_readers[block_id]
                current_postings.extend(reader.read_postings(position, df))
                
                # Obtener siguiente término del bloque
                term_info = reader.peek()
                if term_info:
                    next_term, next_pos, next_df = term_info
                    heapq.heappush(heap, (next_term, block_id, next_pos, next_df))
                    reader.next()  # Avanzar al siguiente término
            
            # Escribir último término
            if current_term and current_postings:
                print(f"  ✓ Mergeando '{current_term}' ({len(current_postings)} postings)")
                self._write_merged_postings(final_index, current_term, current_postings)
                terms_merged += 1
        
        print(f"\nMerge completado: {terms_merged} términos procesados")

    class _BlockReader:
        def __init__(self, block_path, base_path):
            self.term_file = open(os.path.join(block_path, "terms.bin"), "rb")
            self.posting_file = open(os.path.join(block_path, "postings.bin"), "rb")
            self.current_position = 0
            self.file_size = os.path.getsize(os.path.join(block_path, "terms.bin"))
            
        def peek(self):
            """
            Intenta leer el siguiente término sin avanzar, con mejor manejo de errores.
            """
            if self.current_position >= self.file_size:
                return None
                
            try:
                self.term_file.seek(self.current_position)
                
                # Verificar si hay suficientes bytes para leer
                remaining_bytes = self.file_size - self.current_position
                if remaining_bytes < 4:  # Mínimo necesario para term_len
                    return None
                    
                term_len_bytes = self.term_file.read(4)
                term_len = struct.unpack('I', term_len_bytes)[0]
                
                if remaining_bytes < (4 + term_len + 12):  # Total necesario
                    return None
                    
                term = self.term_file.read(term_len).decode('utf-8')
                position, df = struct.unpack('QI', self.term_file.read(12))
                
                # Volver a la posición original
                self.term_file.seek(self.current_position)
                return term, position, df
                
            except Exception as e:
                print(f"Error en peek (posición {self.current_position}): {str(e)}")
                return None

        def read_postings(self, position, df):
            """
            Reads postings with correct frequency handling
            """
            try:
                self.posting_file.seek(position)
                
                # Read actual number of postings
                posting_count = struct.unpack('I', self.posting_file.read(4))[0]
                
                # Read each posting with its frequency
                postings = []
                for _ in range(posting_count):
                    doc_id, freq = struct.unpack('II', self.posting_file.read(8))
                    postings.append((doc_id, freq))
                    
                return postings
                
            except Exception as e:
                print(f"Error reading postings: {str(e)}")
                return []

        def next(self):
            """
            Avanza al siguiente término con verificación de límites.
            """
            if self.current_position >= self.file_size:
                return None
                
            try:
                self.term_file.seek(self.current_position)
                
                term_len = struct.unpack('I', self.term_file.read(4))[0]
                self.term_file.read(term_len)  # Saltar el término
                self.term_file.read(12)  # Saltar position y df
                
                self.current_position = self.term_file.tell()
                return True
                
            except Exception as e:
                print(f"Error en next: {str(e)}")
                return None
            