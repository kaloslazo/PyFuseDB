import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re

def load_and_prepare_fma_data(file_path):
    df = pd.read_csv(file_path, low_memory=False, skiprows=[1])
    df.columns = df.columns.str.replace('track.', '')
    df = df.dropna(how='all').reset_index(drop=True)
    return df

def print_separator(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

def analyze_column_types(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    text_columns = df.select_dtypes(include=['object']).columns
    return numeric_columns, text_columns

def describe_numeric_columns(df, numeric_columns):
    print_separator("Descripción de columnas numéricas")
    for col in numeric_columns:
        stats = df[col].describe()
        print(f"\nColumna: {col}")
        print(f"  Conteo: {stats['count']}")
        print(f"  Media: {stats['mean']:.2f}")
        print(f"  Desviación estándar: {stats['std']:.2f}")
        print(f"  Mínimo: {stats['min']}")
        print(f"  25%: {stats['25%']}")
        print(f"  Mediana: {stats['50%']}")
        print(f"  75%: {stats['75%']}")
        print(f"  Máximo: {stats['max']}")
        print(f"  Valores únicos: {df[col].nunique()}")
        print(f"  Valores más comunes: {df[col].value_counts().head().to_dict()}")

def describe_text_columns(df, text_columns):
    print_separator("Descripción de columnas de texto")
    for col in text_columns:
        non_null_count = df[col].count()
        unique_count = df[col].nunique()
        print(f"\nColumna: {col}")
        print(f"  Valores no nulos: {non_null_count}")
        print(f"  Valores únicos: {unique_count}")
        print(f"  Valores más comunes:")
        print(df[col].value_counts().head().to_string())
        print(f"  Longitud promedio: {df[col].astype(str).str.len().mean():.2f}")
        print(f"  Longitud máxima: {df[col].astype(str).str.len().max()}")

def analyze_text_content(df, text_columns):
    print_separator("Análisis de contenido de texto")
    for col in text_columns:
        print(f"\nColumna: {col}")
        text = ' '.join(df[col].astype(str))
        words = re.findall(r'\w+', text.lower())
        word_freq = Counter(words)
        print(f"  Palabras únicas: {len(word_freq)}")
        print("  Palabras más comunes:")
        for word, count in word_freq.most_common(10):
            print(f"    {word}: {count}")

def analyze_missing_data(df):
    print_separator("Análisis de datos faltantes")
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    missing_table = pd.concat([missing, missing_percent], axis=1, keys=['Total', 'Porcentaje'])
    missing_table = missing_table[missing_table['Total'] > 0].sort_values('Total', ascending=False)
    print(missing_table)

def analyze_correlations(df, numeric_columns):
    print_separator("Análisis de correlaciones")
    corr_matrix = df[numeric_columns].corr()
    high_corr = (corr_matrix.abs() > 0.7) & (corr_matrix != 1.0)
    correlated_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                        for i in range(len(corr_matrix.index))
                        for j in range(i+1, len(corr_matrix.columns))
                        if high_corr.iloc[i, j]]
    if correlated_pairs:
        print("Pares altamente correlacionados (|corr| > 0.7):")
        for pair in correlated_pairs:
            print(f"  {pair[0]} - {pair[1]}: {pair[2]:.2f}")
    else:
        print("No se encontraron pares altamente correlacionados.")

def analyze_categorical_relationships(df, text_columns, target_column='genre'):
    if target_column not in df.columns:
        print(f"La columna objetivo '{target_column}' no está presente en el dataset.")
        return

    print_separator(f"Análisis de relaciones categóricas con {target_column}")
    for col in text_columns:
        if col != target_column:
            cross_tab = pd.crosstab(df[col], df[target_column])
            chi2, p, dof, expected = chi2_contingency(cross_tab)
            print(f"\nRelación entre {col} y {target_column}:")
            print(f"  Chi-cuadrado: {chi2:.2f}")
            print(f"  p-valor: {p:.4f}")

def suggest_search_fields(df, text_columns):
    print_separator("Sugerencias para campos de búsqueda")
    suggested_fields = []
    for col in text_columns:
        unique_ratio = df[col].nunique() / len(df)
        avg_length = df[col].astype(str).str.len().mean()
        if unique_ratio > 0.1 and avg_length > 5:
            suggested_fields.append((col, unique_ratio, avg_length))
    
    suggested_fields.sort(key=lambda x: x[1] * x[2], reverse=True)
    print("Campos recomendados para búsqueda (orden de relevancia):")
    for field, unique_ratio, avg_length in suggested_fields[:10]:
        print(f"  {field}:")
        print(f"    Ratio de valores únicos: {unique_ratio:.2f}")
        print(f"    Longitud promedio: {avg_length:.2f}")

def main():
    file_path = "./app/data/fma/tracks.csv"
    df = load_and_prepare_fma_data(file_path)
    
    print(f"Dimensiones del dataset: {df.shape}")
    
    numeric_columns, text_columns = analyze_column_types(df)
    
    describe_numeric_columns(df, numeric_columns)
    describe_text_columns(df, text_columns)
    analyze_text_content(df, text_columns)
    analyze_missing_data(df)
    analyze_correlations(df, numeric_columns)
    analyze_categorical_relationships(df, text_columns)
    suggest_search_fields(df, text_columns)

if __name__ == "__main__":
    main()
