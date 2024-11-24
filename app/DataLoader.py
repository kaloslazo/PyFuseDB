import pickle
import pandas as pd
import os
from SqlParser import SqlParser
from InvertedIndex import InvertedIndex
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import traceback

class DataLoader:
    def __init__(self, dataPath, db_name='spotify_songs', user_name='your_user', password='your_password'):
        self.dataPath = dataPath
        self.data = None
        self.db_name = db_name
        self.user_name = user_name
        self.password = password
        self.connection = None
        self.cursor = None
        self.index = InvertedIndex(block_size=5000, dict_size=100000)
        self.sqlParser = SqlParser()

        print("DataLoader inicializado.")
        # Establish PostgreSQL connection
        self._initialize_postgres_connection()

    def loadData(self):
        print("Cargando dataset...")
        try:
            self.data = pd.read_csv(self.dataPath)
            print(f"Dataset cargado exitosamente.\nColumnas: {self.data.columns}\nFilas: {len(self.data)}")
            
            # Forzar reconstrucción del índice si los documentos han cambiado
            if self._check_existing_index() and self._verify_index_size():
                print("Cargando índice existente en memoria...")
                self._load_existing_index()
            else:
                print("Construyendo nuevo índice en memoria...")
                # Limpiar archivos antiguos
                self.index.clear_files()
                self.index.build_index(self.data["texto_concatenado"].astype(str).tolist())
                print("Índice en memoria construido exitosamente.")

            # Verificar si la base de datos y el índice existen en PostgreSQL
            if not self._check_existing_index_postgres():
                print("Base de Datos de PostgreSQL y el index no existen. Creando...")
                self.create_postgres_db()
            else:
                print("PostgreSQL database and index already exist. Verificado exitosamente.")
                
        except Exception as e:
            print(f"Error durante la carga de datos: {e}")
            print("Stacktrace:")
            print(traceback.format_exc())


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
        print(f"Ejecutando query: {query}\nTop K: {topK}")

        parsed_query = self.sqlParser.parseQuery(query)
        fields = parsed_query['fields']
        like_term = parsed_query['like_term']

        if '*' in fields: fields = list(self.data.columns)
        print(f"Campos seleccionados: {fields}")
        print(f"Término de búsqueda: {like_term}")

        if like_term: results = self.index.search(like_term, topK)
        else:
            # Si no hay término de búsqueda, devolver los primeros topK resultados
            results = [(i, 1.0) for i in range(min(topK, len(self.data)))]

        if not results:
            print("No se encontraron resultados.")
            return []

        formatted_results = []
        for doc_id, score in results:
            row = self.data.iloc[doc_id]
            row_data = [str(row[field]) if field in self.data.columns else 'N/A' for field in fields]
            row_data.append(f"{score * 100:.2f}")
            formatted_results.append(row_data)

        print(f"Resultados formateados: {formatted_results}")
        return formatted_results

    def _initialize_postgres_connection(self):
        try:
            self.connection = psycopg2.connect(
                dbname='postgres',
                user=self.user_name,
                password=self.password
            )
            self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            self.cursor = self.connection.cursor()

            # Create the database if it doesn't exist
            self.cursor.execute(
                sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"),
                (self.db_name,)
            )
            if not self.cursor.fetchone():
                print(f"Database {self.db_name} does not exist. Creating...")
                self.cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.db_name)))
            self.cursor.close()
            self.connection.close()

            # Reconnect to the newly created database
            self.connection = psycopg2.connect(
                dbname=self.db_name,
                user=self.user_name,
                password=self.password
            )
            self.cursor = self.connection.cursor()
        except Exception as e:
            print(f"Error initializing PostgreSQL connection: {e}")
            import traceback
            print(traceback.format_exc())

    def _check_existing_index_postgres(self):
        try:
            self.cursor.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'songs'
                );
                """
            )
            table_exists = self.cursor.fetchone()[0]
            if not table_exists:
                return False

            self.cursor.execute("SELECT COUNT(*) FROM songs;")
            row_count = self.cursor.fetchone()[0]
            return row_count == len(self.data)  # Verify index matches dataset size
        except Exception as e:
            print(f"Error checking existing PostgreSQL index: {e}")
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
            
            with open(self.dataPath, 'r') as f:
                next(f)  # Skip header
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

            formatted_results = [
                [str(col) for col in row[:2]] + [f"{score * 100:.2f}"]
                for row, score in zip(results, [1.0] * len(results))[:topK]
            ]

            print(f"PostgreSQL resultados: {formatted_results}")
            return formatted_results
        except Exception as e:
            print(f"Error ejecutando PostgreSQL query: {e}")
            return []

