import pickle
import os
import glob
from collections import defaultdict
import numpy as np
from TextPreProcess import TextPreProcess

class InvertedIndex:
    def __init__(self, block_size=1000, dict_size=100000):
        """
        Inicializa el índice invertido
        block_size: tamaño del bloque para postings lists
        dict_size: tamaño del diccionario en memoria
        """
        self.block_size = block_size
        self.dict_size = dict_size
        self.current_block = defaultdict(list)
        self.main_dictionary = {}  
        self.doc_count = 0
        self.document_norms = None
        self.count_output = 0
        self.newListParameters = []
        
        self.bin_path = os.path.join("app", "data", "bin")
        if not os.path.exists(self.bin_path):
            os.makedirs(self.bin_path)

        self.preprocessor = TextPreProcess()
        self.preprocessor.loadStopList()

    def build_index(self, documents):
        print(f"\nConstruyendo índice con {len(documents)} documentos")
        self.clear_files()
        self.doc_count = len(documents)
        block_count = 0
        total_terms = set()

        # Primera pasada: Construir bloques ordenados
        for doc_id, document in enumerate(documents):
            if doc_id % 100 == 0:
                print(f"Procesando documento {doc_id}/{len(documents)}")
            
            tokens = self.tokenize(document)
            if doc_id % 1000 == 0:
                print(f"Muestra de tokens del documento {doc_id}: {tokens[:10]}")
                
            term_freq = defaultdict(int)
            
            for token in tokens:
                term_freq[token] += 1
                total_terms.add(token)
            
            for token, freq in term_freq.items():
                self.current_block[token].append((doc_id, freq))
                
            if len(self.current_block) >= self.dict_size:
                print(f"Escribiendo bloque {block_count} con {len(self.current_block)} términos")
                self._write_block_and_update_dictionary(block_count)
                block_count += 1
                self.current_block.clear()
                
        # Escribir último bloque si existe
        if self.current_block:
            print(f"Escribiendo bloque final {block_count} con {len(self.current_block)} términos")
            self._write_block_and_update_dictionary(block_count)
            block_count += 1
            
        print(f"\nEstadísticas antes del merge:")
        print(f"- Número total de términos únicos encontrados: {len(total_terms)}")
        print(f"- Número de bloques creados: {block_count}")
            
        # Segunda pasada: Merge de bloques (SPIMI)
        print("\nRealizando merge de bloques...")
        self.merge_blocks()

        # Cargar y actualizar diccionario principal desde el merge final
        print("\nActualizando diccionario principal desde merge final...")
        self._update_main_dictionary()

        # Verificar el índice final
        self._verify_index()

        # Calcular y guardar normas
        print("\nCalculando normas de documentos...")
        self._calculate_document_norms()
        self.save_norms()

        print("\nVerificando archivos finales:")
        self._verify_files()
        
    def _verify_files(self):
        """Verifica que todos los archivos necesarios existan"""
        files_to_check = {
            "dictionary.bin": "Diccionario principal",
            "norms.bin": "Normas de documentos",
            "dict_0.bin": "Diccionario final del merge"
        }
        
        for filename, description in files_to_check.items():
            filepath = os.path.join(self.bin_path, filename)
            if os.path.exists(filepath):
                filesize = os.path.getsize(filepath)
                print(f"✓ {description} encontrado ({filesize} bytes)")
            else:
                print(f"✗ {description} no encontrado")
            
    def _verify_index(self):
        """Verifica que el índice sea válido"""
        sample_terms = ['love', 'baby', 'yeah', 'like', 'girl', 'flower', 'flowers']
        found_terms = []
        
        # Corrección: usar self.main_dictionary en lugar de self.index.main_dictionary
        for term in sample_terms:
            if term in self.main_dictionary:  # <-- Aquí está la corrección
                df, _ = self.main_dictionary[term]  # <-- Y aquí también
                found_terms.append(f"{term}({df})")
                    
        print(f"Términos de prueba encontrados: {', '.join(found_terms)}")
        
        if not found_terms:
            print("ADVERTENCIA: No se encontraron términos comunes en el índice")
            return False
        return True

    def _write_block_and_update_dictionary(self, block_num):
        """Escribe un bloque ordenado y actualiza el diccionario principal"""
        sorted_terms = sorted(self.current_block.keys())
        block_dict = {}
        
        for term in sorted_terms:
            postings = sorted(self.current_block[term])
            block_dict[term] = (len(postings), postings)

        block_file = os.path.join(self.bin_path, f"dict_{block_num}.bin")
        with open(block_file, "wb") as f:
            pickle.dump(block_dict, f)

    def merge_blocks(self):
        """SPIMI merge con verificación mejorada"""
        print("\nIniciando proceso de merge SPIMI")
        list_parameters = []
        dict_files = sorted(glob.glob(os.path.join(self.bin_path, "dict_*.bin")))
        dict_count = len(dict_files)
        
        if dict_count == 0:
            print("No hay bloques para mergear")
            return
        elif dict_count == 1:
            print("Solo hay un bloque, no es necesario mergear")
            return
            
        print(f"Número total de bloques iniciales: {dict_count}")
        
        # Contar términos totales antes del merge
        total_terms = set()
        for i in range(dict_count):
            dict_i = self._read_dict(i)
            total_terms.update(dict_i.keys())
        print(f"Términos únicos totales antes del merge: {len(total_terms)}")
        
        # Configurar pares para merge
        for i in range(0, dict_count, 2):
            list_parameters.append([i, i + 1, 1, 1])
        
        print(f"Número de pares iniciales a mergear: {len(list_parameters)}")
        iteration = 1

        while True:
            print(f"\n{'='*50}")
            print(f"Iteración de merge #{iteration}")
            print(f"Número de pares en esta iteración: {len(list_parameters)}")
            
            self.newListParameters = []
            
            for idx, (p, q, p_size, q_size) in enumerate(list_parameters, 1):
                print(f"\nMergeando par {idx}/{len(list_parameters)}")
                print(f"  - Bloque {p} con bloque {q}")
                self.merge_pair(p, q, p_size, q_size)

            if self.newListParameters and self.newListParameters[-1][1] is None:
                self.newListParameters[-1][1] = self.count_output
                self.newListParameters[-1][3] = self.count_output + 1

            list_parameters = self.newListParameters

            print(f"\nRenombrando archivos temporales...")
            self._rename_temp_files(dict_count)
            dict_count = self.count_output
            self.count_output = 0

            # Verificar términos después de la iteración
            final_dict = self._read_dict(0)
            print(f"Estado después de la iteración {iteration}:")
            print(f"  - Bloques restantes: {dict_count}")
            print(f"  - Términos en el diccionario actual: {len(final_dict)}")
            
            # Verificar que no se perdieron términos
            if len(final_dict) < len(total_terms):
                print(f"ADVERTENCIA: Se perdieron términos durante el merge")
                print(f"  - Términos originales: {len(total_terms)}")
                print(f"  - Términos actuales: {len(final_dict)}")
                missing_terms = total_terms - set(final_dict.keys())
                print(f"  - Ejemplo de términos perdidos: {list(missing_terms)[:5]}")

            if len(list_parameters) == 1 and list_parameters[0][1] == dict_count:
                print("\nProceso de merge completado!")
                print(f"Total de iteraciones: {iteration}")
                print(f"Términos en el diccionario final: {len(final_dict)}")
                break
                
            iteration += 1

        # Verificación final de términos
        final_dict = self._read_dict(0)
        print("\nVerificación final:")
        print(f"  - Términos originales: {len(total_terms)}")
        print(f"  - Términos en diccionario final: {len(final_dict)}")
        if len(final_dict) < len(total_terms):
            print("  - ADVERTENCIA: Se perdieron términos durante el merge")

    def merge_pair(self, p, q, p_size, q_size):
        """Merge mejorado de un par de bloques"""
        buffer_size = self.dict_size * 4  # Buffer más grande
        output = {}
        initial_count = self.count_output
        terms_processed = 0

        # Leer diccionarios
        dict1 = self._read_dict(p)
        dict2 = self._read_dict(q) if q < self._get_dict_count() else {}
        
        print(f"  Merge de bloques:")
        print(f"    - Bloque {p}: {len(dict1)} términos")
        print(f"    - Bloque {q}: {len(dict2)} términos")

        # Obtener todos los términos únicos
        all_terms = list(set(list(dict1.keys()) + list(dict2.keys())))
        print(f"    - Total términos únicos a procesar: {len(all_terms)}")

        # Procesar todos los términos
        for term in all_terms:
            if len(output) >= buffer_size:
                self._write_temp_dict(output)
                print(f"      Escribiendo bloque temporal con {len(output)} términos")
                output = {}

            # Obtener postings de ambos diccionarios
            postings1 = dict1.get(term, (0, []))[1] if term in dict1 else []
            postings2 = dict2.get(term, (0, []))[1] if term in dict2 else []
            
            # Combinar y ordenar postings
            combined_postings = sorted(set(postings1 + postings2), key=lambda x: x[0])
            output[term] = (len(combined_postings), combined_postings)
            
            terms_processed += 1
            if terms_processed % 1000 == 0:
                print(f"      Procesados {terms_processed}/{len(all_terms)} términos")

        # Escribir último bloque si hay términos
        if output:
            print(f"    Escribiendo bloque final con {len(output)} términos")
            self._write_temp_dict(output)

        # Actualizar parámetros para siguiente iteración
        salto = self.count_output - initial_count
        if not self.newListParameters or self.newListParameters[-1][1] is not None:
            self.newListParameters.append([initial_count, None, salto, None])
        else:
            self.newListParameters[-1][1] = initial_count
            self.newListParameters[-1][3] = salto

        print(f"  Resultados del merge:")
        print(f"    - Términos procesados: {terms_processed}")
        print(f"    - Bloques temporales generados: {salto}")

    def _write_temp_dict(self, output_dict):
        """Escribe un diccionario temporal"""
        dict_file = os.path.join(self.bin_path, f"dict_temp_{self.count_output}.bin")
        with open(dict_file, "wb") as f:
            pickle.dump(dict(output_dict), f)
        self.count_output += 1

    def _rename_temp_files(self, old_count):
        """Renombra archivos temporales y limpia antiguos"""
        # Eliminar archivos antiguos
        for i in range(old_count):
            old_file = os.path.join(self.bin_path, f"dict_{i}.bin")
            if os.path.exists(old_file):
                os.remove(old_file)

        # Renombrar temporales
        for i in range(self.count_output):
            temp_file = os.path.join(self.bin_path, f"dict_temp_{i}.bin")
            new_file = os.path.join(self.bin_path, f"dict_{i}.bin")
            if os.path.exists(temp_file):
                os.rename(temp_file, new_file)

    def _update_main_dictionary(self):
        """Actualiza el diccionario principal con el resultado final del merge"""
        print("\nActualizando diccionario principal...")
        final_dict = {}
        final_file = os.path.join(self.bin_path, "dict_0.bin")
        
        if not os.path.exists(final_file):
            print("ERROR: No se encontró el archivo de diccionario final")
            return
            
        try:
            with open(final_file, "rb") as f:
                merged_dict = pickle.load(f)
                print(f"Diccionario final contiene {len(merged_dict)} términos")
                
                # Guardar el diccionario completo
                for term, (df, postings) in merged_dict.items():
                    final_dict[term] = (df, postings)
                
                print(f"Términos procesados para diccionario principal: {len(final_dict)}")
                
            self.main_dictionary = final_dict
            
            # Guardar el diccionario principal
            dict_path = os.path.join(self.bin_path, "dictionary.bin")
            with open(dict_path, "wb") as f:
                pickle.dump(self.main_dictionary, f)
                
            print(f"Diccionario principal guardado con {len(self.main_dictionary)} términos")
            dict_size = os.path.getsize(dict_path)
            print(f"Tamaño del archivo: {dict_size} bytes")
            
        except Exception as e:
            print(f"ERROR al actualizar diccionario principal: {e}")
            raise

    def search(self, query, top_k=10):
        """Realiza una búsqueda usando el diccionario principal"""
        print(f"\nBuscando: '{query}'")
        
        # Usar el diccionario principal en vez de leer del disco
        if not self.main_dictionary:
            self.load_main_dictionary()
            
        if not self.main_dictionary:
            print("ERROR: Diccionario principal no encontrado")
            return []
        
        print(f"Estado del índice:")
        print(f"- Términos en diccionario principal: {len(self.main_dictionary)}")
        print(f"- Muestra de términos: {list(self.main_dictionary.keys())[:5]}")
        
        terms_dict = self.preprocessor.preprocess_query(query)
        print(f"Términos de búsqueda procesados: {list(terms_dict.keys())}")
        
        scores = defaultdict(float)
        query_weights = {}
        query_norm = 0
        
        for term, tf in terms_dict.items():
            if term not in self.main_dictionary:
                print(f"Término '{term}' no encontrado en el índice")
                continue
                
            df, postings = self.main_dictionary[term]
            print(f"Término '{term}' encontrado con df={df}")
            
            w_tq = self._calculate_weight(tf, df)
            query_weights[term] = w_tq
            query_norm += w_tq * w_tq
            
            for doc_id, doc_tf in postings:
                w_td = self._calculate_weight(doc_tf, df)
                scores[doc_id] += w_td * w_tq
        
        if not scores:
            print("No se encontraron documentos relevantes")
            return []
        
        if query_norm > 0:
            self.load_norms()
            query_norm = np.sqrt(query_norm)
            for doc_id in scores:
                if self.document_norms[doc_id] != 0:
                    scores[doc_id] /= (self.document_norms[doc_id] * query_norm)
        
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        print(f"Encontrados {len(results)} resultados")
        return results

    def _read_dict(self, dict_num):
        """Lee un diccionario específico"""
        dict_file = os.path.join(self.bin_path, f"dict_{dict_num}.bin")
        if os.path.exists(dict_file):
            with open(dict_file, "rb") as f:
                return pickle.load(f)
        return {}

    def _get_dict_count(self):
        """Obtiene el número actual de diccionarios"""
        return len(glob.glob(os.path.join(self.bin_path, "dict_*.bin")))

    def _calculate_weight(self, tf, df):
        """Calcula el peso TF-IDF para un término"""
        if tf <= 0 or df <= 0:  # Evitar log(0)
            return 0
        return (1 + np.log10(tf)) * np.log10(self.doc_count / df)



    def _calculate_document_norms(self):
        """Calcula las normas de los documentos"""
        self.document_norms = np.zeros(self.doc_count)
        
        final_dict = self._read_dict(0)  # Leer diccionario final
        for term, (df, postings) in final_dict.items():
            for doc_id, tf in postings:
                weight = self._calculate_weight(tf, df)
                self.document_norms[doc_id] += weight * weight
        
        self.document_norms = np.sqrt(self.document_norms)
        self.document_norms[self.document_norms == 0] = 1

    def _calculate_weight(self, tf, df):
        if tf <= 0 or df <= 0:  # Evitar log(0)
            return 0
        return (1 + np.log10(tf)) * np.log10(self.doc_count / df)

    def _save_main_dictionary(self):
        with open(os.path.join(self.bin_path, "dictionary.bin"), "wb") as f:
            pickle.dump(self.main_dictionary, f)

    def load_main_dictionary(self):
        with open(os.path.join(self.bin_path, "dictionary.bin"), "rb") as f:
            self.main_dictionary = pickle.load(f)

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