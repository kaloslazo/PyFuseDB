import time
import pandas as pd
import gradio as gr
from DataLoader import DataLoader
from SqlParser import SqlParser
import gc
import traceback

# => General data
indexRetrievalChoices = ["Implementación Propia", "PostgreSQL", "MongoDB"]
dataPath = "./app/data/afs/spotifySongsTextConcatenated.csv"

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

gc.collect()
def createDemo(dataLoader=dataLoader, sqlParser=sqlParser):
    def updateResults(query, topK, retrievalModel):
        try:
            startTime = time.time()
            queryResults = dataLoader.executeQuery(query, int(topK))
            executionTime = time.time() - startTime
            
            parsed_query = sqlParser.parseQuery(query)
            fields = parsed_query['fields']
            if '*' in fields:
                fields = list(dataLoader.data.columns)
            
            if not queryResults:
                return None, f"No se encontraron resultados. Tiempo: {executionTime:.2f} segundos"
            
            df = pd.DataFrame(queryResults, columns=fields + ['Relevancia (%)'])
            return df, f"Tiempo: {executionTime:.2f} segundos"
            
        except Exception as e:
            print(f"Error: {str(e)}")
            traceback.print_exc()
            raise gr.Error(f"Error: {str(e)}")

    with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")], primary_hue="blue")) as demo:
        with gr.Column(scale=1):
            gr.Markdown("# PyFuseDB", elem_classes="header")
            gr.Markdown("Sistema que integra varios modelos de datos y técnicas avanzadas de recuperación de información.", elem_classes="subtitle")
            
            with gr.Tab("Parte 1: Índice Invertido"):
                with gr.Column():
                    gr.Markdown("## Consulta", elem_classes="section-title")
                    with gr.Row():
                        with gr.Column(scale=2):
                            query_input = gr.Textbox(
                                lines=3,
                                label="Consulta SQL",
                                placeholder="SELECT artist,song,lyrics FROM songs LIKE love music",
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
                        wrap=True,
                        overflow_row_behaviour="paginate",
                        max_rows=10
                    )
                    execution_time = gr.Markdown()
            
            with gr.Tab("Parte 2: Índice Multidimensional"):
                gr.Markdown("## En desarrollo")
            
            search_button.click(
                fn=updateResults,
                inputs=[query_input, top_k, model],
                outputs=[results_df, execution_time]
            )

    return demo