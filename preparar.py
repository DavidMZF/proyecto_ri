from ri import *

docs = cargar_corpus("data/fiqa/corpus/corpus-00000-of-00001.parquet")
queries = cargar_queries("data/fiqa/queries/queries-00000-of-00001.parquet")
qrels = cargar_qrels("data/fiqa/qrels/test.tsv")
docs, queries, qrels = reducir(docs, queries, qrels)

modelo = SentenceTransformer("all-MiniLM-L6-v2")
index, ids = construir_faiss(docs, modelo)
guardar_sistema(docs, queries, qrels, index, ids)
print("Listo:", len(docs), "documentos,", len(queries), "queries")