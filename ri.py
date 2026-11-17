import re, math, random
from collections import Counter, defaultdict
import numpy as np
import faiss
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
import html
import os, json

nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))
STEM = PorterStemmer()


def cargar_corpus(ruta):
    df = pd.read_parquet(ruta)
    return {str(i): html.unescape(f"{t} {x}").replace('""', '"')
            for i, t, x in zip(df["_id"], df["title"], df["text"])}

def cargar_queries(ruta):
    df = pd.read_parquet(ruta)                       # columnas: _id, text
    return {str(i): t for i, t in zip(df["_id"], df["text"])}

def cargar_qrels(ruta):
    qrels = defaultdict(set)              # query_id -> {doc_id relevantes}
    with open(ruta, encoding="utf-8") as f:
        next(f)                           # saltar encabezado
        for linea in f:
            qid, did, score = linea.strip().split("\t")
            if int(score) > 0:
                qrels[qid].add(did)
    return qrels

def reducir(docs, queries, qrels, n_queries=50, n_extra=5000):
    qids = [q for q in qrels if q in queries][:n_queries]
    qrels = {q: qrels[q] for q in qids}
    queries = {q: queries[q] for q in qids}
    relevantes = set().union(*qrels.values())
    resto = [d for d in docs if d not in relevantes]
    random.seed(0)
    elegidos = relevantes | set(random.sample(resto, n_extra))
    docs = {d: docs[d] for d in elegidos if d in docs}
    return docs, queries, qrels

def preprocesar(texto):
    tokens = re.findall(r"[a-z]+", texto.lower())          # tokenización + normalización
    return [STEM.stem(t) for t in tokens if t not in STOP and len(t) > 2]

def construir_indice(docs):
    indice = defaultdict(dict)        # término -> {doc: frecuencia}
    longitud, unicos = {}, {}
    for doc_id, texto in docs.items():
        tokens = preprocesar(texto)
        longitud[doc_id] = len(tokens)
        unicos[doc_id] = len(set(tokens))
        for t, f in Counter(tokens).items():
            indice[t][doc_id] = f
    N = len(docs)
    return {"indice": indice, "len": longitud, "unicos": unicos,
            "N": N, "avgdl": sum(longitud.values()) / N}

# --- Binario: Jaccard ---
def jaccard(consulta, idx):
    q = set(preprocesar(consulta))
    inter = Counter()
    for t in q:
        for d in idx["indice"].get(t, {}):
            inter[d] += 1
    return {d: c / (len(q) + idx["unicos"][d] - c) for d, c in inter.items()}


# --- Vectorial: TF-IDF + coseno ---
def idf(t, idx):
    return math.log(1 + idx["N"] / len(idx["indice"][t]))

def peso_tf(f):
    return 1 + math.log(f)

def preparar_tfidf(idx):
    normas = defaultdict(float)
    for t, postings in idx["indice"].items():
        w_idf = idf(t, idx)
        for d, f in postings.items():
            normas[d] += (peso_tf(f) * w_idf) ** 2
    idx["normas"] = {d: math.sqrt(v) for d, v in normas.items()}

def tfidf(consulta, idx):
    q = Counter(preprocesar(consulta))
    scores, norma_q = defaultdict(float), 0
    for t, fq in q.items():
        if t not in idx["indice"]:
            continue
        w_idf = idf(t, idx)
        wq = peso_tf(fq) * w_idf
        norma_q += wq ** 2
        for d, f in idx["indice"][t].items():
            scores[d] += wq * peso_tf(f) * w_idf
    norma_q = math.sqrt(norma_q)
    return {d: s / (norma_q * idx["normas"][d]) for d, s in scores.items()}


# --- Probabilístico: BM25 ---
def bm25(consulta, idx, k1=1.5, b=0.75):
    scores = defaultdict(float)
    for t in set(preprocesar(consulta)):
        if t not in idx["indice"]:
            continue
        df = len(idx["indice"][t])
        w = math.log((idx["N"] - df + 0.5) / (df + 0.5) + 1)
        for d, f in idx["indice"][t].items():
            denom = f + k1 * (1 - b + b * idx["len"][d] / idx["avgdl"])
            scores[d] += w * f * (k1 + 1) / denom
    return scores

def construir_faiss(docs, modelo):
    ids = list(docs.keys())
    emb = modelo.encode([docs[i] for i in ids], normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])     # producto interno = coseno (vectores normalizados)
    index.add(np.array(emb, dtype="float32"))
    return index, ids

def semantico(consulta, sis, n=10):
    q = sis["modelo"].encode([consulta], normalize_embeddings=True)
    sims, pos = sis["faiss"].search(np.array(q, dtype="float32"), n)
    return {sis["ids"][p]: float(s) for s, p in zip(sims[0], pos[0])}

def construir_sistema(docs):
    idx = construir_indice(docs)
    preparar_tfidf(idx)
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    index, ids = construir_faiss(docs, modelo)
    return {"docs": docs, "idx": idx, "modelo": modelo, "faiss": index, "ids": ids}

def buscar(sis, nombre, consulta, n=10):
    if nombre == "binario":
        scores = jaccard(consulta, sis["idx"])
    elif nombre == "tfidf":
        scores = tfidf(consulta, sis["idx"])
    elif nombre == "bm25":
        scores = bm25(consulta, sis["idx"])
    else:
        scores = semantico(consulta, sis, n)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

def precision_recall(ranking, relevantes):
    aciertos = sum(1 for d in ranking if d in relevantes)
    return aciertos / max(len(ranking), 1), aciertos / len(relevantes)

def average_precision(ranking, relevantes):
    aciertos, suma = 0, 0
    for i, d in enumerate(ranking, 1):
        if d in relevantes:
            aciertos += 1
            suma += aciertos / i
    return suma / len(relevantes)

def evaluar(sis, nombre, queries, qrels, k=10):
    filas, aps = [], []
    for qid, rel in qrels.items():
        ranking = [d for d, _ in buscar(sis, nombre, queries[qid], k)]
        p, r = precision_recall(ranking, rel)
        ap = average_precision(ranking, rel)
        filas.append({"consulta": queries[qid], "precision": p, "recall": r, "ap": ap})
        aps.append(ap)
    return filas, sum(aps) / len(aps)       # (métricas por consulta, MAP)

def guardar_sistema(docs, queries, qrels, index, ids, carpeta="datos"):
    os.makedirs(carpeta, exist_ok=True)
    datos = {"docs": docs, "queries": queries, "ids": ids,
             "qrels": {q: sorted(r) for q, r in qrels.items()}}
    for nombre, contenido in datos.items():
        with open(f"{carpeta}/{nombre}.json", "w", encoding="utf-8") as f:
            json.dump(contenido, f)
    faiss.write_index(index, f"{carpeta}/faiss.index")

def cargar_sistema(carpeta="datos"):
    datos = {}
    for nombre in ["docs", "queries", "ids", "qrels"]:
        with open(f"{carpeta}/{nombre}.json", encoding="utf-8") as f:
            datos[nombre] = json.load(f)
    qrels = {q: set(r) for q, r in datos["qrels"].items()}
    idx = construir_indice(datos["docs"])
    preparar_tfidf(idx)
    sis = {"docs": datos["docs"], "idx": idx, "ids": datos["ids"],
           "modelo": SentenceTransformer("all-MiniLM-L6-v2"),
           "faiss": faiss.read_index(f"{carpeta}/faiss.index")}
    return sis, datos["queries"], qrels