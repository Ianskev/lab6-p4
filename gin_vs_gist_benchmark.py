import psycopg2
import time
import logging
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Crear directorio para resultados si no existe
os.makedirs('gin_vs_gist_results', exist_ok=True)

# Función para generar datos sintéticos para pruebas grandes
def generate_synthetic_data(num_records):
    """Generate synthetic article data for benchmarks"""
    logging.info(f"Generating {num_records} synthetic records...")
    
    # Word lists for random content generation
    topics = ['technology', 'health', 'politics', 'science', 'business', 'sports', 'entertainment']
    adjectives = ['latest', 'important', 'critical', 'essential', 'breaking', 'innovative', 'trending']
    verbs = ['announces', 'reveals', 'launches', 'discovers', 'reports', 'analyzes', 'explores']
    
    synthetic_data = []
    for i in range(num_records):
        # Generate random ID starting from 10M to avoid conflicts
        record_id = 10000000 + i
        
        # Generate title
        topic = random.choice(topics)
        adj = random.choice(adjectives)
        verb = random.choice(verbs)
        title = f"{adj.title()} {topic.title()} {verb.title()}: Article #{i}"
        
        # Generate content paragraph
        sentences = []
        sentence_count = random.randint(3, 8)
        for j in range(sentence_count):
            word_count = random.randint(8, 20)
            words = []
            for k in range(word_count):
                # Mix of topic words and common words
                if random.random() < 0.3:
                    words.append(random.choice(topics))
                elif random.random() < 0.5:
                    words.append(random.choice(adjectives))
                else:
                    words.append(random.choice(verbs))
            sentence = ' '.join(words).capitalize() + '.'
            sentences.append(sentence)
        content = ' '.join(sentences)
        
        synthetic_data.append((record_id, title, content))
        
        # Log progress for large generations
        if i % 10000 == 0 and i > 0:
            logging.info(f"Generated {i}/{num_records} synthetic records...")
    
    return synthetic_data

# Conectar a la base de datos
logging.info("Conectando a la base de datos PostgreSQL...")
conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="postgres"
)
conn.autocommit = True
cursor = conn.cursor()

# 1. MEDICIÓN DE TAMAÑO DE ÍNDICES
logging.info("Midiendo tamaño de índices en disco...")

cursor.execute("""
SELECT 
    pg_size_pretty(pg_relation_size('idx_articles_gin_text_vector')) as gin_index_size,
    pg_relation_size('idx_articles_gin_text_vector') as gin_index_bytes,
    pg_size_pretty(pg_relation_size('idx_articles_gist_text_vector')) as gist_index_size,
    pg_relation_size('idx_articles_gist_text_vector') as gist_index_bytes
""")

size_results = cursor.fetchone()
logging.info(f"Tamaño índice GIN: {size_results[0]} ({size_results[1]} bytes)")
logging.info(f"Tamaño índice GIST: {size_results[2]} ({size_results[3]} bytes)")

size_data = {
    'Tipo de Índice': ['GIN', 'GIST'],
    'Tamaño (bytes)': [size_results[1], size_results[3]]
}

# 2. PRUEBA DE INSERCIÓN
logging.info("Midiendo tiempo de inserción de nuevos registros...")

insert_results = []
# Test con diferentes cantidades de registros (desde pequeños hasta muy grandes)
for num_rows in [10, 100, 10000, 100000, 1000000]:
    logging.info(f"Probando inserción de {num_rows} registros...")
    
    try:
        # Determine if we should use real or synthetic data
        if num_rows > 50000:
            # Use synthetic data for large tests
            test_data = generate_synthetic_data(num_rows)
            logging.info(f"Using synthetic data for {num_rows} records")
        else:
            # Use real data for smaller tests
            logging.info(f"Obteniendo {num_rows} registros aleatorios para prueba...")
            cursor.execute(f"""
            SELECT id, title, content 
            FROM articles 
            ORDER BY random() 
            LIMIT {num_rows}
            """)
            test_data = cursor.fetchall()
            logging.info(f"Datos obtenidos. Iniciando pruebas de inserción...")
        
        batch_size = 1000 if num_rows > 1000 else num_rows
        
        # Inserción en tabla con GIN
        logging.info("Insertando en tabla con índice GIN...")
        cursor.execute("TRUNCATE articles_gin RESTART IDENTITY")  # Limpiar tabla de prueba anterior
        
        start_time = time.time()
        
        if num_rows <= 1000:
            # Para cantidades pequeñas, insertar todo de una vez
            placeholders = ", ".join(["%s"] * len(test_data))
            query = f"""
            WITH data(id, title, content) AS (VALUES {placeholders})
            INSERT INTO articles_gin(id, title, content)
            SELECT id, title, content FROM data
            """
            cursor.execute(query, test_data)
            
        else:
            # Para grandes cantidades, usar inserción por lotes
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i+batch_size]
                values_str = []
                for row in batch:
                    values_str.append(cursor.mogrify("(%s,%s,%s)", row).decode('utf-8'))
                
                query = f"""
                INSERT INTO articles_gin(id, title, content) 
                VALUES {','.join(values_str)}
                """
                cursor.execute(query)
                
                if i % 10000 == 0 and i > 0:
                    logging.info(f"GIN: Insertados {i}/{num_rows} registros...")
        
        gin_time = time.time() - start_time
        logging.info(f"Tiempo de inserción GIN para {num_rows} registros: {gin_time:.4f} segundos")
        
        # Inserción en tabla con GIST
        logging.info("Insertando en tabla con índice GIST...")
        cursor.execute("TRUNCATE articles_gist RESTART IDENTITY")  # Limpiar tabla de prueba
        
        start_time = time.time()
        
        if num_rows <= 1000:
            # Para cantidades pequeñas, insertar todo de una vez
            placeholders = ", ".join(["%s"] * len(test_data))
            query = f"""
            WITH data(id, title, content) AS (VALUES {placeholders})
            INSERT INTO articles_gist(id, title, content)
            SELECT id, title, content FROM data
            """
            cursor.execute(query, test_data)
            
        else:
            # Para grandes cantidades, usar inserción por lotes
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i+batch_size]
                values_str = []
                for row in batch:
                    values_str.append(cursor.mogrify("(%s,%s,%s)", row).decode('utf-8'))
                
                query = f"""
                INSERT INTO articles_gist(id, title, content) 
                VALUES {','.join(values_str)}
                """
                cursor.execute(query)
                
                if i % 10000 == 0 and i > 0:
                    logging.info(f"GIST: Insertados {i}/{num_rows} registros...")
        
        gist_time = time.time() - start_time
        logging.info(f"Tiempo de inserción GIST para {num_rows} registros: {gist_time:.4f} segundos")
        
        insert_results.append({
            'num_rows': num_rows,
            'gin_time': gin_time,
            'gist_time': gist_time
        })
    
    except Exception as e:
        logging.error(f"Error en prueba de inserción con {num_rows} registros: {str(e)}")
        # Continuar con el siguiente tamaño
        continue

# 3. PRUEBA DE BÚSQUEDAS
logging.info("Midiendo tiempo de búsquedas...")

search_terms = ['health', 'politics', 'technology', 'climate change', 'economy']
top_k_values = [10, 50, 100]

search_results = []
for search_term in search_terms:
    for top_k in top_k_values:
        logging.info(f"Probando búsqueda de '{search_term}', top {top_k}...")
        
        # Búsqueda con GIN
        query_gin = f"""
        EXPLAIN ANALYZE
        SELECT id, title 
        FROM articles_gin
        WHERE text_vector @@ plainto_tsquery('english', '{search_term}')
        ORDER BY ts_rank(text_vector, plainto_tsquery('english', '{search_term}')) DESC
        LIMIT {top_k}
        """
        start_time = time.time()
        cursor.execute(query_gin)
        results_gin = cursor.fetchall()
        gin_time = time.time() - start_time
        
        # Extraer tiempo de ejecución del EXPLAIN ANALYZE
        gin_time_ms = None
        for row in results_gin:
            if "Execution Time" in row[0]:
                gin_time_ms = float(row[0].split("Execution Time: ")[1].split(" ms")[0])
                break
        
        # Búsqueda con GIST
        query_gist = f"""
        EXPLAIN ANALYZE
        SELECT id, title 
        FROM articles_gist
        WHERE text_vector @@ plainto_tsquery('english', '{search_term}')
        ORDER BY ts_rank(text_vector, plainto_tsquery('english', '{search_term}')) DESC
        LIMIT {top_k}
        """
        start_time = time.time()
        cursor.execute(query_gist)
        results_gist = cursor.fetchall()
        gist_time = time.time() - start_time
        
        # Extraer tiempo de ejecución del EXPLAIN ANALYZE
        gist_time_ms = None
        for row in results_gist:
            if "Execution Time" in row[0]:
                gist_time_ms = float(row[0].split("Execution Time: ")[1].split(" ms")[0])
                break
        
        logging.info(f"Búsqueda '{search_term}', top {top_k}: GIN={gin_time_ms:.2f}ms, GIST={gist_time_ms:.2f}ms")
        
        search_results.append({
            'search_term': search_term,
            'top_k': top_k,
            'gin_time_ms': gin_time_ms,
            'gist_time_ms': gist_time_ms
        })

# 4. VISUALIZACIONES Y ANÁLISIS DE RESULTADOS

# Gráfico de tamaño de índices
plt.figure(figsize=(10, 6))
plt.bar(size_data['Tipo de Índice'], size_data['Tamaño (bytes)'])
plt.title('Comparación de Tamaño de Índices')
plt.ylabel('Tamaño (MB)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('gin_vs_gist_results/index_size_comparison.png', dpi=300, bbox_inches='tight')

# Gráfico de tiempo de inserción
plt.figure(figsize=(14, 7))
x = np.arange(len(insert_results))
bar_width = 0.35
plt.bar(x, [r['gin_time'] for r in insert_results], width=bar_width, label='GIN')
plt.bar(x + bar_width, [r['gist_time'] for r in insert_results], width=bar_width, label='GIST')
plt.xlabel('Número de Registros Insertados')
plt.ylabel('Tiempo (segundos)')
plt.title('Tiempo de Inserción: GIN vs GIST')
plt.xticks(x + bar_width / 2, [f"{r['num_rows']:,}" for r in insert_results], rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('gin_vs_gist_results/insertion_time_comparison.png', dpi=300, bbox_inches='tight')

# Gráfico logarítmico para mejor visualización de diferencias
plt.figure(figsize=(14, 7))
plt.bar(x, [r['gin_time'] for r in insert_results], width=bar_width, label='GIN')
plt.bar(x + bar_width, [r['gist_time'] for r in insert_results], width=bar_width, label='GIST')
plt.xlabel('Número de Registros Insertados')
plt.ylabel('Tiempo (segundos, escala logarítmica)')
plt.title('Tiempo de Inserción: GIN vs GIST (Escala Logarítmica)')
plt.xticks(x + bar_width / 2, [f"{r['num_rows']:,}" for r in insert_results], rotation=45)
plt.yscale('log')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('gin_vs_gist_results/insertion_time_comparison_log.png', dpi=300, bbox_inches='tight')

# Gráfico de tiempo de búsqueda
df_search = pd.DataFrame(search_results)
plt.figure(figsize=(18, 10))

# Agrupar por término de búsqueda
for i, term in enumerate(search_terms):
    plt.subplot(2, 3, i+1)
    term_data = df_search[df_search['search_term'] == term]
    
    plt.plot(term_data['top_k'], term_data['gin_time_ms'], marker='o', label='GIN')
    plt.plot(term_data['top_k'], term_data['gist_time_ms'], marker='s', label='GIST')
    plt.title(f"Búsqueda: '{term}'")
    plt.xlabel('Top-K')
    plt.ylabel('Tiempo (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

plt.tight_layout()
plt.savefig('gin_vs_gist_results/search_performance.png', dpi=300, bbox_inches='tight')

# Gráfico de comparación general de búsqueda
plt.figure(figsize=(10, 6))
gin_avg = df_search['gin_time_ms'].mean()
gist_avg = df_search['gist_time_ms'].mean()

plt.bar(['GIN', 'GIST'], [gin_avg, gist_avg])
plt.title('Tiempo Promedio de Búsqueda')
plt.ylabel('Tiempo (ms)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('gin_vs_gist_results/average_search_time.png', dpi=300, bbox_inches='tight')

# Gráfico de ratio de rendimiento en búsquedas
plt.figure(figsize=(10, 6))
df_search['ratio'] = df_search['gist_time_ms'] / df_search['gin_time_ms']
avg_ratio = df_search['ratio'].mean()

plt.bar(['GIST/GIN'], [avg_ratio])
plt.title('Ratio de Rendimiento en Búsquedas (GIST/GIN)')
plt.ylabel('Ratio (mayor a 1 significa que GIN es más rápido)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(y=1, color='r', linestyle='--')
plt.savefig('gin_vs_gist_results/search_performance_ratio.png', dpi=300, bbox_inches='tight')

# Guardar resultados en CSV
pd.DataFrame(search_results).to_csv('gin_vs_gist_results/search_performance.csv', index=False)
pd.DataFrame(insert_results).to_csv('gin_vs_gist_results/insertion_performance.csv', index=False)
pd.DataFrame([{
    'gin_index_size_bytes': size_results[1],
    'gist_index_size_bytes': size_results[3],
    'gin_index_size_pretty': size_results[0],
    'gist_index_size_pretty': size_results[2]
}]).to_csv('gin_vs_gist_results/index_size.csv', index=False)

# Resumen de resultados
logging.info("\n===== RESUMEN DE RESULTADOS =====")
logging.info(f"Tamaño índice GIN: {size_results[0]} ({size_results[1]} bytes)")
logging.info(f"Tamaño índice GIST: {size_results[2]} ({size_results[3]} bytes)")
logging.info(f"Ratio de tamaño (GIST/GIN): {size_results[3]/size_results[1]:.2f}x")
logging.info("\nRendimiento de inserción:")
for r in insert_results:
    logging.info(f"  {r['num_rows']:,} registros - GIN: {r['gin_time']:.2f}s, GIST: {r['gist_time']:.2f}s, Ratio (GIST/GIN): {r['gist_time']/r['gin_time']:.2f}x")
logging.info(f"\nTiempo promedio de búsqueda - GIN: {gin_avg:.2f}ms, GIST: {gist_avg:.2f}ms")
logging.info(f"Ratio de rendimiento en búsquedas (GIST/GIN): {avg_ratio:.2f}x")

logging.info("\nExperimento completado. Resultados guardados en directorio 'gin_vs_gist_results'")
conn.close()