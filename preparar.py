from ri import *

TAMANO = None      # None = corpus completo. Ej.: 20000 si el despliegue tiene poca RAM

docs = cargar_corpus("data/fiqa/corpus/corpus-00000-of-00001.parquet")
queries = cargar_queries("data/fiqa/queries/queries-00000-of-00001.parquet")
qrels = filtrar_qrels(docs, queries, cargar_qrels("data/fiqa/qrels/test.tsv"))
queries = {q: queries[q] for q in qrels}

modelo = SentenceTransformer("all-MiniLM-L6-v2")
index, ids = construir_faiss(docs, modelo)
guardar_sistema(docs, queries, qrels, index, ids)
print(len(docs), "documentos,", len(qrels), "consultas con qrels")