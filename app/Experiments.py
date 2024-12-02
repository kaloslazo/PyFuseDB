#from DataLoader import DataLoader

#dataLoader = DataLoader("data/afs/spotifySongsTextConcatenated.csv")
#dataLoader.experiment("SELECT track_id FROM songs LIKE love", 8)

import matplotlib.pyplot as plt

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
