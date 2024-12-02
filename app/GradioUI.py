import time
import pandas as pd
import gradio as gr
from DataLoader import DataLoader
from SqlParser import SqlParser
from MultiDim import SequentialKNN, RTreeKNN, FaissKNN
from MultiDim import load_collection, load_filenames, load_reducer, extract_feature_vector, reduce_query_dimensionality
import gc
import traceback

# => General data
indexRetrievalChoices = ["Implementación Propia", "PostgreSQL"]
dataPath = "data/afs/spotifySongsTextConcatenated.csv"

multidimRetrievalChoices = ["Sequential", "RTree", "Faiss (HNSW)"]

# => General configuration
sqlParser = SqlParser()
dataLoader = DataLoader(dataPath)

print("Cargando datos y construyendo índice...")
start_time = time.time()
try:
    dataLoader.loadData()
    end_time = time.time()
    print(f"Tiempo total de carga y construcción del índice: {end_time - start_time:.2f} segundos")
except Exception as e:
    print(f"Error durante la carga de datos: {str(e)}")
    print("Stacktrace:")
    traceback.print_exc()
    exit(1)

print("Construyendo Indices Multidimensionales...")

multidim_collection = load_collection()
multidim_reduced_collection = load_collection(reduced=True)
multidim_reducer = load_reducer()
multidim_filenames = load_filenames()

sequential_full = SequentialKNN(multidim_collection)
faiss_full = FaissKNN(multidim_collection)

sequential_red = SequentialKNN(multidim_reduced_collection)
rtree_red = RTreeKNN(multidim_reduced_collection)
faiss_red = FaissKNN(multidim_reduced_collection)

print("Indices Multidimensionales construidos!...")

gc.collect()
def createDemo(dataLoader=dataLoader, sqlParser=sqlParser):
    def updateResultsInverted(query, topK, retrievalModel):
        try:
            startTime = time.time()

            if retrievalModel == "Implementación Propia":
                queryResults = dataLoader.executeQuery(query, int(topK))
            elif retrievalModel == "PostgreSQL":
                queryResults = dataLoader.executeQueryPostgreSQL(query, int(topK))
            else:
                raise gr.Error(f"Modelo de búsqueda no soportado: {retrievalModel}")

            print(f"Ejecutando con {retrievalModel}")

            executionTime = time.time() - startTime

            # Parsea la consulta SQL
            parsed_query = sqlParser.parseQuery(query)
            fields = parsed_query['fields']
            if '*' in fields:
                fields = list(dataLoader.data.columns)

            if not queryResults:
                empty_df = pd.DataFrame(columns=fields + ['Relevancia (%)'])
                return gr.update(value=empty_df, headers=empty_df.columns), f"No se encontraron resultados. Tiempo: {executionTime:.4f} segundos"

            # Eliminar duplicados manteniendo solo la primera ocurrencia con mayor relevancia
            seen_results = {}
            
            for result in queryResults:
                # Usar los primeros n-1 elementos como clave (excluyendo el score)
                key = tuple(result[:-1])
                # Remover el símbolo % antes de convertir a float
                current_score = float(result[-1].rstrip('%'))
                
                if key not in seen_results or current_score > float(seen_results[key][-1].rstrip('%')):
                    seen_results[key] = result
                    
            unique_results = list(seen_results.values())
            unique_results = unique_results[:int(topK)]

            df = pd.DataFrame(unique_results, columns=fields + ['Relevancia (%)'])
            print(df.head())

            return gr.update(value=df, headers=df.columns.tolist()), f"Tiempo: {executionTime:.4f} segundos"



        except Exception as e:
            print(f"Error: {str(e)}")
            traceback.print_exc()
            raise gr.Error(f"Error: {str(e)}")


    def updateResultsMultidim(image_path, retrieval_choice, search_type, search_param, reduce_dimensionality):
        try:
            if retrieval_choice == "RTree" and not reduce_dimensionality:
                return None, "RTree no soporta embeddings completos.", "Se debe reducir la dimensionalidad."
            elif retrieval_choice == "Faiss (HNSW)" and search_type == "Range Search":
                return None, "Faiss (HNSW) no soporta búsqueda por Rango.", "Solo soporta KNN."

            if search_param <= 0.0:
                return None, "Parámetro Inválido.", ""

            # Feature extraction
            startTime = time.time()

            print(f"Extracting embedding of {image_path}")
            
            embedding = extract_feature_vector(image_path)
            if reduce_dimensionality:
                print("Reducing dimensionality of embedding")
                embedding = reduce_query_dimensionality(embedding, multidim_reducer)

            featureExtractTime = time.time() - startTime

            # print(embedding)
            print(f"Embedding of shape: {embedding.shape}")

            # Search
            startTime = time.time()
            queryResults = None
            
            if reduce_dimensionality:
                print("Reduced Dim Query")
                if retrieval_choice == "Sequential":
                    print("Sequential")
                    if search_type == "KNN":
                        queryResults = sequential_red.knn_search(embedding, k=int(search_param))
                    else:
                        queryResults = sequential_red.range_search(embedding, radius=float(search_param))

                elif retrieval_choice == "RTree":
                    if search_type == "KNN":
                        queryResults = rtree_red.knn_search(embedding, k=int(search_param))
                    else:
                        queryResults = rtree_red.range_search(embedding, radius=float(search_param))

                else:
                    if search_type == "KNN":
                        queryResults = faiss_red.knn_search(embedding, k=int(search_param))

            else:
                print("Full Dim Query")
                if retrieval_choice == "Sequential":
                    if search_type == "KNN":
                        queryResults = sequential_full.knn_search(embedding, k=int(search_param))
                    else:
                        queryResults = sequential_full.range_search(embedding, radius=float(search_param))

                elif retrieval_choice == "Faiss (HNSW)":
                    if search_type == "KNN":
                        queryResults = faiss_full.knn_search(embedding, k=int(search_param))

            executionTime = time.time() - startTime

            if queryResults is None:
                None, f"Técnica de Búsqueda No Soportada. Tiempo de extracción de embedding: {featureExtractTime:.4f} segunds", f"Tiempo de búsqueda: {executionTime:.4f} segundos"


            print(f"Index: {retrieval_choice}, search algorithm: {search_type}")
            print(queryResults)

            results = [
                (f"./data/imagenette/images/{multidim_filenames[idx]}", dist)
                for idx, dist in queryResults
            ]
            return results, f"Tiempo de extracción de embedding: {featureExtractTime:.4f} segunds", f"Tiempo de búsqueda: {executionTime:.4f} segundos"

        except Exception as e:
            print(f"Error: {str(e)}")
            traceback.print_exc()
            raise gr.Error(f"Error: {str(e)}")


    with gr.Blocks(title="🐍 PyFuseDB" ,theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")], primary_hue="green")) as demo:
        with gr.Column(scale=1):
            gr.Markdown("# 🐍 PyFuseDB", elem_classes="header")
            gr.Markdown("Sistema que integra varios modelos de datos y técnicas avanzadas de recuperación de información.", elem_classes="subtitle")

            with gr.Tab("Parte 1: Índice Invertido"):
                with gr.Column():
                    gr.Markdown("## Consulta", elem_classes="section-title")
                    with gr.Row():
                        with gr.Column(scale=2):
                            query_input = gr.Textbox(
                                lines=3,
                                label="Consulta SQL",
                                placeholder="SELECT track_artist,track_name,lyrics FROM songs LIKE love music",
                                elem_classes="query-input"
                            )
                        with gr.Column(scale=1):
                            top_k = gr.Number(
                                label="Top K resultados",
                                value=10,
                                minimum=1,
                                maximum=20,
                                step=1
                            )
                            model = gr.Dropdown(
                                label="Modelo de búsqueda",
                                choices=indexRetrievalChoices,
                                value=indexRetrievalChoices[0]
                            )

                    search_button = gr.Button(
                        "Ejecutar búsqueda",
                        variant="primary"
                    )

                    gr.Markdown("## Resultados")
                    results_df = gr.Dataframe(
                        headers=None,
                        datatype=["str"],
                        wrap=True
                    )
                    execution_time = gr.Markdown()

            with gr.Tab("Parte 2: Índice Multidimensional"):
                with gr.Column(scale=1):
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("## Upload Image")
                            image_input = gr.Image(type="filepath", label="Upload Image")
                            gr.Markdown("RTree has to use Reduced Dimensionality. Faiss HNSW does not support range search.")

                        with gr.Column(scale=1):
                            gr.Markdown("### Search Configuration")

                            retrieval_model = gr.Dropdown(
                                label="Search Model",
                                choices=multidimRetrievalChoices,
                                value=multidimRetrievalChoices[0]
                            )

                            reduce_dimensionality = gr.Checkbox(
                                label="Reduce Dimensionality",
                                value=False,
                            )

                            search_type = gr.Dropdown(
                                label="Search Type",
                                choices=["KNN", "Range Search"],
                                value="KNN"
                            )

                            arg_value = gr.Number(
                                label="K / Radius",
                                value=5,
                                step=1,
                            )



                    
                    multidim_search_button = gr.Button("Ejecutar búsqueda", variant="primary")

                    gr.Markdown("### Results")
                    multidim_results_df = gr.Dataset(label="Results Table", components=["image", "number"], headers=["Image", "Distance"])

                    feature_extract_time = gr.Markdown()
                    multidim_execution_time = gr.Markdown()


            search_button.click(
                fn=updateResultsInverted,
                inputs=[query_input, top_k, model],
                outputs=[results_df, execution_time]
            )

            multidim_search_button.click(
                fn=updateResultsMultidim,
                inputs=[image_input, retrieval_model, search_type, arg_value, reduce_dimensionality],
                outputs=[multidim_results_df, feature_extract_time, multidim_execution_time]
            )

    return demo
