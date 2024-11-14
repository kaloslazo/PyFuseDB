<h1 align="center">PyFuseDB</h1>

**PyFuseDB** es un sistema que integra varios modelos de datos y técnicas avanzadas de recuperación de información dentro de una única base de datos. Nuestro sistema permite a los usuarios recuperar datos estructurados mediante un **índice invertido** y datos no estructurados, como imágenes y audio, por medio de **estructuras multidimensionales** que utilizan vectores característicos.

## 1. Introducción

### 1.1 Objetivo del Proyecto
El objetivo de PyFuseDB es crear una base de datos robusta y versátil capaz de gestionar y recuperar datos tanto estructurados como no estructurados de manera eficiente. A través de la implementación de un índice invertido para datos textuales y el uso de estructuras multidimensionales para datos en formatos complejos, se busca optimizar la consulta y recuperación de información en distintos contextos de uso.

### 1.2 Descripción del Dominio de Datos
Para este proyecto, se utiliza el **Spotify Million Song Dataset**, una base de datos pública que contiene información sobre canciones. Este conjunto de datos incluye el nombre de la canción, el nombre del artista, el enlace a la canción y la letra, recolectados hasta el año 2022. Este dominio de datos resulta ideal para experimentos en clasificación, recomendación y recuperación de información debido a su riqueza textual y su aplicación en sistemas de recomendación y análisis de música.

### 1.3 Importancia de Técnicas de Indexación
La eficiencia en la recuperación de información es crucial en grandes volúmenes de datos, como los que maneja PyFuseDB. Las técnicas de indexación permiten reducir significativamente el tiempo de respuesta en las consultas, optimizando el rendimiento del sistema. El uso de índices invertidos para el texto (como las letras de las canciones) facilita búsquedas rápidas, mientras que las estructuras multidimensionales mejoran el acceso y recuperación de datos no estructurados a través de características representativas, lo que es esencial en aplicaciones avanzadas como la recuperación de contenido en multimedia.

## 2. Backend: Índice Invertido

El backend de PyFuseDB se basa en la implementación de un índice invertido para permitir una recuperación rápida y eficiente de documentos en función de términos textuales. A continuación, se describen las principales funcionalidades del índice invertido, su construcción y optimización para consultas.

### 2.1 Construcción del Índice Invertido
La clase `InvertedIndex` define el núcleo de la construcción del índice invertido. Este proceso se realiza en bloques, con el objetivo de manejar grandes volúmenes de datos sin sobrecargar la memoria. La implementación permite almacenar el índice en archivos separados de acuerdo con el tamaño del bloque configurado (`block_size`), generando archivos de bloques en formato `.pkl`.

- **Documentos y Tokenización**: Cada documento se analiza y se tokeniza usando `TfidfVectorizer` de `scikit-learn`, el cual elimina las palabras comunes (stop words) y convierte las palabras a minúsculas para un análisis uniforme. Los tokens extraídos se almacenan en el `current_block`, que es un diccionario donde cada token apunta a una lista de identificadores de documentos (`doc_id`) en los que aparece.
  
- **Bloques**: Cuando el número de tokens en `current_block` alcanza el `block_size`, el bloque se serializa y se almacena en un archivo utilizando `pickle`, optimizando el uso de la memoria y permitiendo una construcción de índice en grandes datasets.

- **Normalización**: Durante la construcción del índice, se calcula la norma de cada documento en base a sus valores TF-IDF. Estas normas son fundamentales para calcular la similitud de coseno en las búsquedas y se almacenan en `document_norms` para su posterior uso.

La función `build_index` permite la construcción del índice a partir de una lista de documentos, procesándolos en lotes (`batch_size`) para controlar el flujo de datos. Esta función muestra el progreso de la construcción y asegura que el índice quede distribuido en bloques manejables.

### 2.2 Optimización de Consultas con Similitud de Coseno
Una vez construido el índice, PyFuseDB permite realizar consultas eficientes utilizando similitud de coseno entre el vector de consulta y los vectores de documentos en el índice. La similitud de coseno es adecuada para medir la relevancia en búsquedas textuales, ya que considera tanto la frecuencia de términos como la magnitud de los documentos.

- **Vector de Consulta**: Para cada consulta, se genera un vector de consulta con `TfidfVectorizer`, similar al proceso de tokenización de los documentos.

- **Calculo de Similitud**: La búsqueda de términos de consulta se realiza cargando cada bloque del índice y comprobando si los términos están presentes en el bloque cargado. Si se encuentra un término, se calcula la similitud de coseno multiplicando los valores TF-IDF del término en la consulta por los valores de los documentos almacenados, usando las normas previamente calculadas. Esto permite ordenar los documentos en función de la relevancia y devolver los `top_k` resultados más relevantes.

La función `search` implementa la búsqueda y muestra el progreso de la consulta, incluyendo los términos de la consulta y el número de resultados encontrados.

### 2.3 Construcción del Índice Invertido en PostgreSQL
Además de la implementación en Python, PyFuseDB permite construir un índice invertido en **PostgreSQL**. PostgreSQL cuenta con soporte para índices GIN (Generalized Inverted Index), que son ideales para optimizar búsquedas en datos textuales. La integración con PostgreSQL permite manejar grandes volúmenes de datos y realizar consultas SQL avanzadas.

- **Configuración de PostgreSQL**: En PostgreSQL, se crea una tabla para almacenar los documentos con columnas que incluyen el texto y otros metadatos, como el nombre de la canción y el artista. A esta tabla se le aplica un índice GIN en la columna de texto.

- **Consulta Avanzada**: Una vez creado el índice GIN, se pueden realizar consultas con operadores de búsqueda de texto (`@@` para búsquedas con peso) o búsquedas de similitud de coseno utilizando extensiones de PostgreSQL. Esto permite que las consultas textuales sean mucho más rápidas en comparación con una búsqueda lineal.

Este enfoque de construir un índice invertido en PostgreSQL complementa la solución en Python, permitiendo consultas rápidas tanto a nivel de backend en el sistema de archivos como en una base de datos SQL.


## 3. Backend: Índice Multidimensional

### 3.1 Técnica de Indexación

### 3.2 K-NN Search y Range Search

### 3.3 Análisis y Solución de la Maldición de la Dimensionalidad


## 4. Frontend: Interfaz de Usuario


### 4.1 Diseño de GUI

Para lograr una interfaz intuitiva y funcional, se utilizó la librería de Python: `Gradio`. Esta elección permite crear interfaces de usuario web de manera sencilla y rápida, ofreciendo componentes predefinidos que se configuran en Python, lo que simplifica el desarrollo y mejora la mantenibilidad.

### 4.2 Manual de Usuario

#### 4.2.1 Ingreso de Consultas

Se implementó una sintaxis personalizada inspirada en SQL para facilitar las búsquedas. La interfaz proporciona un campo de texto donde los usuarios pueden ingresar sus consultas. A continuación, se detalla la sintaxis de las consultas soportadas por PyFuseDB:

| Comando | Descripción |
| --- | --- |
| `SELECT` | Selecciona los atributos de la tabla |
| `FROM` | Especifica la tabla de la que se seleccionarán los atributos |
| `WHERE` | Especifica las condiciones que deben cumplir los registros seleccionados |
| `LIKETO` | Encapsula la consulta de similitud de texto |
| `LIMIT` | Limita el número de resultados a mostrar |

A continuación, se muestra un ejemplo de consulta soportada por PyFuseDB. En este ejemplo, se seleccionan los atributos `title` y `artist` de la tabla `Audio` donde el atributo `lyric` es similar al texto `amor en tiempos de guerra`. Además, se limita la cantidad de resultados a 10.

```sql
SELECT title, artist FROM Audio WHERE lyric LIKETO 'amor en tiempos de guerra' LIMIT 10
```

Por otro lado, se incluye una opción para que el usuario especifique la cantidad de documentos a recuperar (Top K), ya sea mediante el comando `LIMIT` en la consulta o a través de un control separado en la interfaz.

#### 4.2.2 Presentación de Resultados
Se tuvo especial cuidado en la presentación de los resultados de las consultas, con el objetivo de que sean claros y fáciles de interpretar. Para ello, se implementó una tabla que muestra los resultados de la consulta, donde cada fila corresponde a un documento recuperado y cada columna a un atributo seleccionado. Además, se incluyó un mensaje que indica el tiempo que tomó realizar la consulta, proporcionando feedback sobre la eficiencia del sistema.

- **Visualización amigable**: Los resultados de la búsqueda se presentan de forma clara y organizada, mostrando los atributos seleccionados en un formato fácil de leer.
- **Tiempo de consulta**: Se muestra el tiempo que toma realizar la consulta, proporcionando feedback sobre la eficiencia del sistema.

#### 4.2.3 Método de Indexación
La interfaz trae una opción para seleccionar el método de indexación a utilizar. Los usuarios pueden elegir entre `Implementación propia (Índice Invertido)`, `PostgreSQL` y `MongoDB`, permitiendo comparar los resultados obtenidos con cada técnica.

#### 4.2.4 Carga de Archivos Multimedia
Se implementó una funcionalidad para cargar imágenes y audio a la base de datos, permitiendo a los usuarios recuperar datos no estructurados. La lógica está compuesta por los siguientes pasos:

- Un botón permite seleccionar los archivos a cargar.
- Los archivos se procesan para extraer vectores característicos.
- Los vectores se almacenan en la base de datos para su posterior recuperación.

### 4.3 Capturas de Pantalla

### 4.2 Análisis Comparativo de otras Implementaciones


## 5. Experimentación

### 5.1 Tablas y Gráficos de Resultados Experimentales

### 5.2 Análisis y Discusión de Resultados Experimentales

