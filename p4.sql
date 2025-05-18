CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Crear tablas para pruebas
CREATE TABLE articles_gin AS SELECT * FROM articles;
CREATE TABLE articles_gist AS SELECT * FROM articles;

-- Añadir columnas para vectores de texto
ALTER TABLE articles_gin ADD COLUMN text_vector tsvector;
ALTER TABLE articles_gist ADD COLUMN text_vector tsvector;

-- Poblar los vectores de texto
UPDATE articles_gin SET text_vector = to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content, ''));
UPDATE articles_gist SET text_vector = to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content, ''));

-- Crear índice GIN (medir tiempo)
CREATE INDEX idx_articles_gin_text_vector ON articles_gin USING GIN (text_vector);

-- Crear índice GIST (medir tiempo)
CREATE INDEX idx_articles_gist_text_vector ON articles_gist USING GIST (text_vector);