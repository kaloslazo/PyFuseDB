import gradio as gr;

# => General data
indexRetrievalChoices = ["Implementación Propia", "PostgreSQL", "MongoDB"];
sampleData = [
    ["Canción de amor", "Artista 1", 95],
    ["Melodía nocturna", "Artista 2", 87],
    ["Ritmo de verano", "Artista 3", 82],
    ["Balada del viento", "Artista 4", 78],
    ["Sonata del mar", "Artista 5", 75],
];
resultAttributes = ["Título", "Artista", "Relevancia (%)"];
resultTypes = ["str", "str", "number"];

# => Utility functions
def updateResults():
    return sampleData;

# => Main interface
with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")], primary_hue="blue")) as demo:
    gr.Markdown("# PyFuseDB"),
    gr.Markdown("Sistema avanzado de recuperación de información"),

    gr.Markdown("## Consulta");
    with gr.Row():
        with gr.Column(scale = 2):
            query_input = gr.Textbox(
                lines = 5.9,
                label="Consulta SQL",
                placeholder="Ingresa tu consulta SQL",
            );
        with gr.Column(scale = 1):
            top_k = gr.Number(label="Top K resultados", value=10);
            gr.Dropdown(
                label="Modelo de recuperación",
                choices = indexRetrievalChoices,
                value = indexRetrievalChoices[0],
            );

        executeBtn = gr.Button("Ejecutar consulta", variant="primary");

    gr.Markdown("## Resultados");
    resultsDataframe = gr.Dataframe(
        headers=resultAttributes,
        datatype=resultTypes,
        label="Resultados de la búsqueda"
    );
    executeBtn.click(
        fn=updateResults,
        outputs=resultsDataframe
    );

# => Main entry
demo.launch();
