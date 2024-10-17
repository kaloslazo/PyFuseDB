import time
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
            queryResults = dataLoader.executeQuery(query, topK)
            executionTime = time.time() - startTime

            parsed_query = sqlParser.parseQuery(query)
            fields = parsed_query['fields']
            if '*' in fields:
                fields = list(dataLoader.data.columns)
            headers = fields + ['Relevancia (%)']

            if not queryResults:
                return None, headers, f"No se encontraron resultados. Tiempo de ejecución: {executionTime:.2f} segundos"

            # Asegúrate de que los resultados sean una lista de listas (cada resultado es una fila)
            results = [[str(item) for item in result] for result in queryResults]

            print(f"Resultados de la consulta: {results}")
            print(f"Encabezados: {headers}")

            # Retorna los resultados y los encabezados por separado
            return results, headers, f"Tiempo de ejecución: {executionTime:.2f} segundos"

        except Exception as e:
            print(f"Error in updateResults: {str(e)}")
            traceback.print_exc()
            raise gr.Error(f"Ocurrió un error: {str(e)}")

    with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")], primary_hue="blue")) as demo:
        gr.Markdown("# PyFuseDB")
        gr.Markdown("Sistema que integra varios modelos de datos y técnicas avanzadas de recuperación de información dentro de una única base de datos.")

        with gr.Tab("Parte 1: Índice Invertido"):
            gr.Markdown("## Consulta")
            with gr.Row():
                with gr.Column(scale=2):
                    queryInput = gr.Textbox(
                        lines=5.9,
                        label="Consulta SQL",
                        placeholder="Ingresa tu consulta SQL",
                    )
                with gr.Column(scale=1):
                    topK = gr.Number(label="Top K resultados", value=10, minimum=0, step=1, interactive=True)
                    retrievalModel = gr.Dropdown(
                        label="Modelo de recuperación",
                        interactive=True,
                        choices=indexRetrievalChoices,
                        value=indexRetrievalChoices[0],
                    )
            executeBtn = gr.Button("Ejecutar consulta", variant="primary")

            gr.Markdown("## Resultados")
            resultsDataframe = gr.Dataframe(
                headers=["Resultados"],
                datatype=["str"],
                label="Resultados de la búsqueda",
                row_count=(10, "dynamic"),
                col_count=(1, "dynamic"),
            )
            executionTime = gr.Markdown()
            executeBtn.click(
                fn=updateResults,
                inputs=[queryInput, topK, retrievalModel],
                outputs=[resultsDataframe, gr.Dataframe(), executionTime]
            )

        with gr.Tab("Parte 2: Índice Multidimensional"):
            gr.Markdown("## En desarrollo")

    return demo