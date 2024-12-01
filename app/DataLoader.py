import pickle
import pandas as pd
import os
from SqlParser import SqlParser
from InvertedIndex import InvertedIndex
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import traceback
import time

from InvertedIndexFinal import InvertedIndexFinal

load_dotenv()

class DataLoader:
    def __init__(self, dataPath, db_name='spotify_songs', user_name=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD")):
        """Inicializa el DataLoader con la nueva implementación SPIMI"""
        self.dataPath = dataPath
        self.data = None
        self.db_name = db_name
        self.user_name = user_name
        self.password = password
        self.connection = None
        self.cursor = None
        # Cambiamos a la nueva implementación del índice
        self.index = InvertedIndexFinal(block_size=1000, dict_size=50000)
        self.sqlParser = SqlParser()

        print("DataLoader inicializado.")
        self._initialize_postgres_connection()

    def loadData(self):
        """Carga los datos y construye el índice SPIMI completo"""
        print("Cargando dataset...")
        try:
            self.data = pd.read_csv(self.dataPath)
            # Eliminar duplicados basados en track_id
            self.data = self.data.drop_duplicates(subset=['track_id'], keep='first')
            print(f"Dataset cargado exitosamente.\nColumnas: {self.data.columns}\nFilas: {len(self.data)}")

            # Construir el índice SPIMI
            print("Construyendo nuevo índice SPIMI...")
            self._clear_index_files()
            
            # Fase 1: Construcción de bloques
            print("\nFase 1: Construcción de bloques temporales...")
            self.index.build_index(self.data["texto_concatenado"].astype(str).tolist())
            
            # Fase 2: Merge de bloques
            print("\nFase 2: Fusionando bloques en índice final...")
            self.index.merge_blocks()
            print("Índice SPIMI construido y fusionado exitosamente.")

            # Verificar creación del índice final
            final_index_path = os.path.join(self.index.bin_path, "final_index.bin")
            if os.path.exists(final_index_path):
                size = os.path.getsize(final_index_path)
                print(f"✓ Índice final creado: {size} bytes")
            else:
                raise Exception("Error: No se creó el índice final")

            # Verificar/crear base de datos PostgreSQL
            if not self._check_existing_index_postgres():
                print("Base de Datos PostgreSQL no existe. Creando...")
                self.create_postgres_db()
            else:
                print("Base de datos PostgreSQL verificada exitosamente.")

        except Exception as e:
            print(f"Error durante la carga de datos: {e}")
            print("Stacktrace:")
            print(traceback.format_exc())
            raise

    def _clear_index_files(self):
        """Limpia archivos antiguos del índice"""
        if os.path.exists(self.index.bin_path):
            for filename in os.listdir(self.index.bin_path):
                file_path = os.path.join(self.index.bin_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Error eliminando {file_path}: {e}")

    def _verify_index_size(self):
        """Verifica que el índice existente coincida con el número de documentos"""
        try:
            with open(os.path.join(self.index.bin_path, "dict_0.bin"), "rb") as f:
                dict_data = pickle.load(f)
                # Obtener el máximo doc_id de los postings
                max_doc_id = max(doc_id for term_data in dict_data.values()
                               for doc_id, _ in term_data[1])
                # Verificar que coincida con el número de documentos
                return max_doc_id + 1 == len(self.data)
        except:
            return False

    def _check_existing_index(self):
        """Verifica si existe un índice válido"""
        required_files = [
            os.path.join(self.index.bin_path, "dictionary.bin"),
            os.path.join(self.index.bin_path, "norms.bin"),
            os.path.join(self.index.bin_path, "dict_0.bin")
        ]
        return all(os.path.exists(f) for f in required_files)

    def _load_existing_index(self):
        """Carga un índice existente"""
        try:
            self.index.load_main_dictionary()
            self.index.load_norms()
            # Establecer doc_count desde el dataset actual
            self.index.doc_count = len(self.data)
            print(f"Índice cargado con {len(self.index.main_dictionary)} términos")
            print("Muestra de términos:", list(self.index.main_dictionary.keys())[:5])

        except Exception as e:
            print(f"Error cargando índice existente: {e}")
            print("Reconstruyendo índice...")
            self.index.build_index(self.data["texto_concatenado"].astype(str).tolist())

    def executeQuery(self, query, topK=10):
        """Ejecuta consulta usando el índice SPIMI con deduplicación mejorada"""
        print(f"Ejecutando query: {query}\nTop K: {topK}")

        # Mapeo de nombres de campos alternativos
        field_mapping = {
            'title': 'track_name',
            'artist': 'track_artist',
            'lyrics': 'lyrics',
            'album': 'track_album_name',
            'genre': 'playlist_genre'
        }

        parsed_query = self.sqlParser.parseQuery(query)
        fields = [field_mapping.get(field, field) for field in parsed_query['fields']]
        like_term = parsed_query['like_term']

        if '*' in fields:
            fields = list(self.data.columns)
        print(f"Campos seleccionados: {fields}")
        print(f"Término de búsqueda: {like_term}")

        if like_term:
            raw_results = self.index.search(like_term, topK)  # Ya no multiplicamos por 3
        else:
            raw_results = [(i, 1.0) for i in range(min(topK, len(self.data)))]

        if not raw_results:
            print("No se encontraron resultados.")
            return []

        # Formatear resultados directamente sin deduplicación adicional
        formatted_results = []
        seen_track_ids = set()
        
        for doc_id, score in raw_results:
            if 0 <= doc_id < len(self.data):
                row = self.data.iloc[doc_id]
                track_id = row['track_id']
                
                if track_id not in seen_track_ids:
                    seen_track_ids.add(track_id)
                    row_data = [str(row[field]) if field in self.data.columns else 'N/A' 
                            for field in fields]
                    # Mantener el score como número con dos decimales, sin el símbolo %
                    row_data.append(f"{score:.2f}")
                    formatted_results.append(row_data)
                    
                    if len(formatted_results) >= topK:
                        break

        return formatted_results

    def _initialize_postgres_connection(self):
        try:
            self.connection = psycopg2.connect(
                dbname='postgres',
                user=self.user_name,
                password=self.password,
                host='127.0.0.1'
            )
            self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            self.cursor = self.connection.cursor()

            # Crear la base de datos si no existe
            self.cursor.execute(
                sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"),
                (self.db_name,)
            )
            if not self.cursor.fetchone():
                print(f"Database {self.db_name} does not exist. Creating...")
                self.cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.db_name)))
            self.cursor.close()
            self.connection.close()

            # Reconectar a la base de datos recién creada
            self.connection = psycopg2.connect(
                dbname=self.db_name,
                user=self.user_name,
                password=self.password,
                host='127.0.0.1'
            )
            self.cursor = self.connection.cursor()
        except Exception as e:
            print(f"Error initializing PostgreSQL connection: {e}")
            import traceback
            print(traceback.format_exc())

    def _check_existing_index_postgres(self):
        try:
            print("Verificando existencia de la tabla 'songs' en PostgreSQL...")
            # Check if the table 'songs' exists
            self.cursor.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'songs'
                );
                """
            )
            table_exists = self.cursor.fetchone()[0]
            if not table_exists:
                print("La tabla 'songs' no existe en la base de datos.")
                return False

            print("La tabla 'songs' existe. Verificando el número de filas...")

            # Check row count
            self.cursor.execute("SELECT COUNT(*) FROM songs;")
            row_count = self.cursor.fetchone()[0]

            if row_count == 0:
                print("La tabla 'songs' está vacía.")
                return False

            print(f"La tabla 'songs' contiene {row_count} filas.")
            return True

        except Exception as e:
            print(f"Error al verificar la tabla 'songs': {e}")
            return False


    def create_postgres_db(self):
        try:
            print("Creando la tabla PostgreSQL e insertando registros...")
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS songs (
                    id SERIAL PRIMARY KEY,
                    track_id TEXT,
                    track_name TEXT,
                    track_artist TEXT,
                    lyrics TEXT,
                    track_album_name TEXT,
                    playlist_name TEXT,
                    playlist_genre TEXT,
                    playlist_subgenre TEXT,
                    language TEXT,
                    texto_concatenado TEXT
                );
                """
            )

            with open(self.dataPath, 'r', encoding="utf-8" ) as f:
                self.cursor.copy_expert(
                    """
                    COPY songs(track_id, track_name, track_artist, lyrics, track_album_name,
                            playlist_name, playlist_genre, playlist_subgenre, language, texto_concatenado)
                    FROM STDIN WITH CSV HEADER;
                    """, f
                )

            self.cursor.execute(
                "CREATE INDEX IF NOT EXISTS songs_text_idx ON songs USING gin(to_tsvector('english', texto_concatenado));"
            )

            self.connection.commit()
            print("Tabla y índice PostgreSQL creados exitosamente.")
        except Exception as e:
            print(f"Error creando la base de datos o índice de PostgreSQL: {e}")
            self.connection.rollback()

    def executeQueryPostgreSQL(self, query, topK=10):
        try:
            print(f"Ejecutando PostgreSQL query: {query}")
            self.cursor.execute(query)
            results = self.cursor.fetchall()

            # Limitar los resultados a topK
            sliced_results = results[:topK]

            # Formatear los resultados como una lista de listas
            formatted_results = [
                [str(col) for col in row] + [f"{100.0:.2f}"] for row in sliced_results
            ]

            print(f"PostgreSQL resultados: {formatted_results}")
            return formatted_results

        except Exception as e:
            print(f"Error ejecutando PostgreSQL query: {e}")
            # Rollback transaction to reset the cursor state
            self.connection.rollback()
            return []
        
    def experiment(self, query, topK=8):
        with open("reporte.txt", "w") as report_file:
            for N in [1000, 2000, 4000, 8000, 16000, 32000, 64000]:
                report_file.write(f">>>>>>>>>>>>>>>>>>>>>>>>>>> Experimento con N={N}\n")
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>>> Experimento con N={N}")
                
                # Limpiar archivos antiguos
                folder_path = "app/data/bin"
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"Error al eliminar {filename}: {e}")

                # Construir un índice con N documentos
                report_file.write(">>>>>>>>>>>> Indice invertido: \n")
                print(">>>>>>>>>>>> Indice invertido: ")
                self.index.clear_files()
                #
                if N < 18000:
                    self.data = pd.read_csv(self.dataPath)[:N]
                elif N < 36000:
                    self.data = pd.concat([pd.read_csv(self.dataPath)] * 2, ignore_index=True)[:N]
                else:
                    self.data = pd.concat([pd.read_csv(self.dataPath)] * 4, ignore_index=True)[:N]
            
                
                self.index.build_index(self.data["texto_concatenado"].astype(str).tolist())

                # Ejecutar la consulta en el índice
                start = time.time()
                queryInvert = self.sqlParser.parseQuery(query)
                like_term = queryInvert['like_term']
                results = self.index.search(like_term, topK)
                end = time.time()
                time_taken = end - start
                report_file.write(f"Consulta ejecutada en {time_taken:.2f} segundos en el índice invertido\n")
                print(f"Consulta ejecutada en {time_taken:.2f} segundos en el índice invertido")

                # Ejecutar la consulta en PostgreSQL
                report_file.write(">>>>>>>>>>>> PostgreSQL: \n")
                print(">>>>>>>>>>>> PostgreSQL: ")
                try:
                    self.cursor.execute("DROP TABLE IF EXISTS songs;")
                    
                    self.cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS songs (
                            id SERIAL PRIMARY KEY,
                            track_id TEXT,
                            track_name TEXT,
                            track_artist TEXT,
                            lyrics TEXT,
                            track_album_name TEXT,
                            playlist_name TEXT,
                            playlist_genre TEXT,
                            playlist_subgenre TEXT,
                            language TEXT,
                            texto_concatenado TEXT
                        );
                        """
                    )

                    # Abrir el archivo CSV
                    with open(self.dataPath, 'r', encoding="utf-8") as f:
                        next(f)  # Saltar el encabezado
                        lines = [next(f) for _ in range(min(N, 18000))]

                        if(N>18000 and N < 36000):
                            lines = lines *2 
                            lines = lines[:N]
                        elif(N > 36000):
                            lines = lines + lines * 4
                            lines = lines[:N]                        
                        
                        # Copiar las primeras N tuplas al DB
                        from io import StringIO
                        temp_file = StringIO(''.join(lines))

                        self.cursor.copy_expert(
                            """
                            COPY songs(track_id, track_name, track_artist, lyrics, track_album_name,
                                    playlist_name, playlist_genre, playlist_subgenre, language, texto_concatenado)
                            FROM STDIN WITH CSV HEADER;
                            """, temp_file
                        )

                    # Crear el índice para la columna texto_concatenado
                    self.cursor.execute(
                        "CREATE INDEX IF NOT EXISTS songs_text_idx ON songs USING gin(to_tsvector('english', texto_concatenado));"
                    )
                    self.connection.commit()

                    start = time.time()
                    queryPostgres = self.sqlParser.parseQueryPostgres(query)
                    self.cursor.execute(queryPostgres)
                    results = self.cursor.fetchall()
                    end = time.time()
                    time_taken = end - start
                    report_file.write(f"Consulta ejecutada en {time_taken:.2f} segundos en PostgreSQL\n")
                    print(f"Consulta ejecutada en {time_taken:.2f} segundos en PostgreSQL")
                except Exception as e:
                    error_message = f"Error al ejecutar la consulta en PostgreSQL: {e}"
                    report_file.write(error_message + "\n")
                    print(error_message)
