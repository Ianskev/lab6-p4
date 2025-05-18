# Benchmark de Comparación entre Índices GIN vs GIST en PostgreSQL

Este proyecto contiene un script de benchmark para comparar el rendimiento de dos tipos de índices de PostgreSQL para búsquedas de texto completo: GIN (Generalized Inverted Index) y GIST (Generalized Search Tree).

## Descripción

El script `gin_vs_gist_benchmark.py` ejecuta un conjunto de pruebas exhaustivas para comparar el rendimiento de los índices GIN y GIST en diferentes escenarios, midiendo tres métricas principales:

1. **Tamaño de los índices en disco**
   - Compara cuánto espacio ocupa cada tipo de índice en la base de datos

2. **Tiempo de inserción de datos**
   - Evalúa el rendimiento al insertar diferentes volúmenes de datos (desde 10 hasta 1 millón de registros)
   - Utiliza tanto datos reales como datos sintéticos generados para pruebas a gran escala

3. **Tiempo de búsqueda de texto**
   - Mide el tiempo de respuesta para consultas de texto completo
   - Compara búsquedas con diferentes términos y recuperando diferente número de resultados

## Características

- Generación automática de datos sintéticos para pruebas a gran escala
- Pruebas con diferentes volúmenes de datos (10, 100, 10.000, 100.000 y 1.000.000 de registros)
- Evaluación de múltiples términos de búsqueda (health, politics, technology, etc.)
- Diferentes tamaños de resultado (top 10, 50, 100)
- Visualización completa de resultados con gráficos comparativos
- Exportación de resultados a archivos CSV para análisis posterior

## Requisitos

- Python 3.6 o superior
- PostgreSQL con extensión de búsqueda de texto habilitada
- Tablas `articles`, `articles_gin` y `articles_gist` configuradas en la base de datos
- Bibliotecas Python:
  - psycopg2
  - numpy
  - pandas
  - matplotlib

## Configuración de la Base de Datos

El benchmark asume que existen tres tablas en la base de datos:
- `articles`: Tabla principal con los datos de los artículos
- `articles_gin`: Tabla con índice GIN para el benchmark
- `articles_gist`: Tabla con índice GIST para el benchmark

Las tablas de prueba deben tener una columna `text_vector` que almacena los vectores de texto para la búsqueda de texto completo.

## Resultados

El benchmark genera varios archivos de resultados en el directorio `gin_vs_gist_results`:

- **Gráficos**:
  - Comparación de tamaño de índices
  - Tiempo de inserción (normal y escala logarítmica)
  - Rendimiento de búsquedas por término
  - Tiempo promedio de búsqueda
  - Ratio de rendimiento entre índices

- **Datos CSV**:
  - `index_size.csv`: Tamaño de los índices
  - `insertion_performance.csv`: Tiempos de inserción
  - `search_performance.csv`: Tiempos de búsqueda

## Ejecución

Para ejecutar el benchmark:

```bash
python gin_vs_gist_benchmark.py
```

El proceso puede tardar bastante tiempo, especialmente al realizar pruebas con grandes volúmenes de datos.

## Interpretación

- Un ratio GIST/GIN mayor que 1 en búsquedas indica que GIN es más rápido
- Los resultados permiten elegir el índice adecuado según prioridades (velocidad de inserción vs búsqueda, tamaño en disco vs rendimiento)
