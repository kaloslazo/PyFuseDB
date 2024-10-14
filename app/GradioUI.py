import time
import gradio as gr

# => General data
indexRetrievalChoices = ["Implementación Propia", "PostgreSQL", "MongoDB"]
sampleData = [
    ["Canción de amor", "Artista 1", 95],
    ["Melodía nocturna", "Artista 2", 87],
    ["Ritmo de verano", "Artista 3", 82],
    ["Balada del viento", "Artista 4", 78],
    ["Sonata del mar", "Artista 5", 75],
]
resultAttributes = ["Título", "Artista", "Relevancia (%)"]
resultTypes = ["str", "str", "number"]

# => Utility functions
def updateResults():
    return sampleData

# => Main interface
def createDemo(dataLoader, sqlParser):
    def updateResults(query, topK, retrievalModel):
        try:
            startTime = time.time()
            queryParsed = sqlParser.parseQuery(query)
            queryResults = dataLoader.executeQuery(queryParsed, topK)
            executionTime = time.time() - startTime
            
            return queryResults, f"Tiempo de ejecución: {executionTime:.2f} segundos"
        
        except ValueError as e:
            raise gr.Error(f"{str(e)}")

        except Exception as e:
            raise gr.Error(f"{str(e)}")

    with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")], primary_hue="blue")) as demo:
        gr.Markdown("# PyFuseDB"),
        gr.Markdown("Sistema que integra varios modelos de datos y técnicas avanzadas de recuperación de información dentro de una única base de datos. Permite a los usuarios recuperar datos estructurados por medio de un índice invertido y datos no estructurados como imágenes y audio por medio de estructuras multidimensionales que utilizan vectores característicos."),

        # Project part 1
        with gr.Tab("Parte 1: Índice Invertido"):
            gr.Markdown("## Consulta")
            with gr.Row():
                with gr.Column(scale = 2):
                    queryInput = gr.Textbox(
                        lines=5.9,
                        label="Consulta SQL",
                        placeholder="Ingresa tu consulta SQL",
                    );
                with gr.Column(scale = 1):
                    topK = gr.Number(label="Top K resultados", value=10, minimum=0, step=1, interactive=True);
                    retrievalModel = gr.Dropdown(
                        label="Modelo de recuperación",
                        interactive=True,
                        choices=indexRetrievalChoices,
                        value=indexRetrievalChoices[0],
                    );
            executeBtn = gr.Button("Ejecutar consulta", variant="primary");

            gr.Markdown("## Resultados");
            resultsDataframe = gr.Dataframe(
                headers=resultAttributes,
                datatype=resultTypes,
            );
            executionTime = gr.Markdown()
            executeBtn.click(
                fn=updateResults,
                inputs=[queryInput, topK, retrievalModel],
                outputs=[resultsDataframe, executionTime]
            );

        # Project part 2
        with gr.Tab("Parte 2: Índice Multidimensional"):
            gr.Markdown("## En desarrollo")

    return demo
