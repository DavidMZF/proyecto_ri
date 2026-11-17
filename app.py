from flask import Flask, request, render_template
from ri import *

app = Flask(__name__)
MODELOS = ["binario", "tfidf", "bm25", "embeddings"]

sis, queries, qrels = cargar_sistema("datos")
METRICAS = {m: evaluar(sis, m, queries, qrels) for m in MODELOS}

@app.route("/", methods=["GET", "POST"])
def inicio():
    consulta, resultados = "", {}
    if request.method == "POST":
        consulta = request.form["consulta"]
        n = int(request.form.get("n", 5))
        for m in request.form.getlist("modelos"):
            resultados[m] = buscar(sis, m, consulta, n)
    return render_template("index.html", consulta=consulta, resultados=resultados,
                           modelos=MODELOS, metricas=METRICAS, docs=sis["docs"])


if __name__ == "__main__":
    app.run(debug=False)