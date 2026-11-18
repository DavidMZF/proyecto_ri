import json
import numpy as np
from ri import cargar_sistema, evaluar, buscar

MODELOS = ["binario", "tfidf", "bm25", "embeddings"]
KS = (5, 10, 20)
METRICA = "NDCG@10"     # métrica principal para comparar (calidad del orden en el top-10)
UMBRAL = 0.02           # diferencia mínima para hablar de mejora o empeoramiento


def comparar(filas_a, filas_b):
    """Modelo a (embeddings) frente a un modelo b (léxico), consulta por consulta."""
    difs = np.array([a[METRICA] - b[METRICA] for a, b in zip(filas_a, filas_b)])
    rng = np.random.default_rng(0)
    medias = rng.choice(difs, size=(2000, len(difs)), replace=True).mean(axis=1)   # bootstrap
    return {"mejora": int((difs > UMBRAL).sum()),
            "empeora": int((difs < -UMBRAL).sum()),
            "similar": int((np.abs(difs) <= UMBRAL).sum()),
            "dif_media": float(difs.mean()), "desv": float(difs.std()),
            "ic95": [float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))]}


def tabla_por_grupo(filas, etiqueta):
    """Métrica media de cada modelo por grupo de consultas (las filas de todos los modelos van alineadas)."""
    grupos = {}
    for i, f in enumerate(filas["bm25"]):
        grupos.setdefault(etiqueta(f), []).append(i)
    return {g: {"n": len(idx),
                **{m: float(np.mean([filas[m][i][METRICA] for i in idx])) for m in MODELOS}}
            for g, idx in sorted(grupos.items())}


def etiqueta_longitud(cortes):
    a, b = cortes
    def etiqueta(fila):
        n = len(fila["consulta"].split())
        if n <= a:
            return f"1. corta (hasta {a} palabras)"
        return f"2. media ({a + 1} a {b})" if n <= b else f"3. larga (más de {b})"
    return etiqueta


def etiqueta_n_rel(fila):
    n = fila["n_rel"]
    return "1. un relevante" if n == 1 else "2. dos o tres relevantes" if n <= 3 else "3. cuatro o más relevantes"


def etiqueta_numeros(fila):
    return "con números" if any(c.isdigit() for c in fila["consulta"]) else "sin números"


def ejemplos(sis, queries, qrels, filas, n=5):
    """Consultas donde los embeddings más superan a BM25 y viceversa, con el primer resultado de cada uno."""
    emb, bm = filas["embeddings"], filas["bm25"]
    orden = sorted(range(len(emb)), key=lambda i: emb[i][METRICA] - bm[i][METRICA])

    def detalle(i):
        qid = emb[i]["qid"]
        fila = {"qid": qid, "consulta": emb[i]["consulta"], "n_rel": emb[i]["n_rel"],
                "ndcg_emb": emb[i][METRICA], "ndcg_bm25": bm[i][METRICA]}
        for m in ("bm25", "embeddings"):
            res = buscar(sis, m, queries[qid], 1)
            doc = res[0][0] if res else None
            fila[f"top1_{m}"] = None if doc is None else {
                "id": doc, "relevante": doc in qrels[qid], "texto": sis["docs"][doc][:200]}
        return fila

    return {"mejor_emb": [detalle(i) for i in orden[::-1][:n]],
            "mejor_bm25": [detalle(i) for i in orden[:n]]}


def main():
    sis, queries, qrels = cargar_sistema("datos")
    filas, resumen = {}, {}
    for m in MODELOS:
        filas[m], resumen[m] = evaluar(sis, m, queries, qrels, KS)
        r = resumen[m]
        print(f"{m:11s} MAP@100={r['MAP']:.3f}  NDCG@10={r['NDCG@10']:.3f}  "
              f"R@10={r['R@10']:.3f}  {r['ms']:.1f} ms/consulta")

    largos = sorted(len(f["consulta"].split()) for f in filas["bm25"])
    cortes = (largos[len(largos) // 3], largos[2 * len(largos) // 3])

    analisis = {"metrica": METRICA, "umbral": UMBRAL,
                "emb_vs_bm25": comparar(filas["embeddings"], filas["bm25"]),
                "emb_vs_tfidf": comparar(filas["embeddings"], filas["tfidf"]),
                "por_longitud": tabla_por_grupo(filas, etiqueta_longitud(cortes)),
                "por_n_rel": tabla_por_grupo(filas, etiqueta_n_rel),
                "por_numeros": tabla_por_grupo(filas, etiqueta_numeros),
                "ejemplos": ejemplos(sis, queries, qrels, filas)}

    for nombre in ("emb_vs_bm25", "emb_vs_tfidf"):
        a = analisis[nombre]
        print(f"{nombre}: mejora en {a['mejora']}, empeora en {a['empeora']}, similar en {a['similar']} | "
              f"dif. media {a['dif_media']:+.3f} (IC95% {a['ic95'][0]:+.3f} a {a['ic95'][1]:+.3f})")
    for nombre in ("por_longitud", "por_n_rel", "por_numeros"):
        print(f"\n{METRICA} por grupo: {nombre}")
        for g, v in analisis[nombre].items():
            print(f"  {g:32s} n={v['n']:4d} " + "  ".join(f"{m}={v[m]:.3f}" for m in MODELOS))

    salida = {"ks": list(KS), "n_consultas": len(qrels), "n_docs": len(sis["docs"]),
              "modelos": {m: {"resumen": resumen[m], "filas": filas[m]} for m in MODELOS},
              "analisis": analisis}
    with open("datos/metricas.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False)
    print("\nGuardado en datos/metricas.json")


if __name__ == "__main__":
    main()