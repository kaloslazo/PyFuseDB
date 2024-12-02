#from DataLoader import DataLoader

#dataLoader = DataLoader("data/afs/spotifySongsTextConcatenated.csv")
#dataLoader.experiment("SELECT track_id FROM songs LIKE love", 8)

import matplotlib.pyplot as plt

#1
# Datos proporcionados
N_values = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
inverted_index_times = [0.00, 0.01, 0.02, 0.02, 0.10, 0.34, 0.53]
postgres_times = [0.00, 0.00, 0.02, 0.01, 0.01, 0.12, 0.34]

# Crear la figura y el eje
plt.figure(figsize=(10, 6))

# Graficar las dos líneas
plt.plot(N_values, inverted_index_times, label='Inverted Index', marker='o', linestyle='-', color='blue')
plt.plot(N_values, postgres_times, label='PostgreSQL', marker='o', linestyle='-', color='green')

# Agregar etiquetas y título
plt.xlabel('N (Número de documentos)', fontsize=12)
plt.ylabel('Tiempo (segundos)', fontsize=12)
plt.title('Comparación de Tiempos de Ejecución', fontsize=14)

# Agregar leyenda
plt.legend(fontsize=12)

# Mostrar la gráfica
plt.grid(True)
plt.tight_layout()
plt.show()

#2

# Datos de la primera tabla
N = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
sequential_times = [0.0021, 0.0043, 0.0089, 0.0166, 0.0334, 0.0716, 0.1449]
rtree_times = [0.0007, 0.0013, 0.0009, 0.0017, 0.0051, 0.0104, 0.0224]
faiss_times = [0.0002, 0.0002, 0.0002, 0.0003, 0.0003, 0.0003, 0.0004]

# Datos de la segunda tabla
radius = [0.5, 1.0, 2.0, 5.0]
sequential_range_times = [0.0033, 0.0032, 0.0031, 0.003]
rtree_range_times = [0.0007, 0.0016, 0.0011, 0.0011]

# Gráfico 1: Tiempos de ejecución para diferentes N
plt.figure(figsize=(12, 6))
plt.plot(N, sequential_times, marker='o', label='Sequential')
plt.plot(N, rtree_times, marker='o', label='RTree')
plt.plot(N, faiss_times, marker='o', label='KNN-HighD')
plt.title('Tiempos de ejecución vs N')
plt.xlabel('N')
plt.ylabel('Tiempo (segundos)')
plt.legend()
plt.grid(True)
plt.savefig("../docs/Experimentos2_1.png")
plt.show()

# Gráfico 2: Tiempos de búsqueda por rango
plt.figure(figsize=(12, 6))
plt.plot(radius, sequential_range_times, marker='o', label='Sequential Range Search')
plt.plot(radius, rtree_range_times, marker='o', label='RTree Range Search')
plt.title('Tiempos de búsqueda por rango')
plt.xlabel('Radio')
plt.ylabel('Tiempo (segundos)')
plt.legend()
plt.grid(True)
plt.savefig("../docs/Experimentos2_2.png")
plt.show()


