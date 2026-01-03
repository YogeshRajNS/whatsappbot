# ==========================================
# ChromaDB API Server for WhatsApp RAG Bot
# (Single Collection, No client_id)
# ==========================================

import os
from flask import Flask, request, jsonify
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ---------------- CONFIG ----------------
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- CHROMADB CLIENT ----------------
chroma_client = Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_DB_DIR
    )
)

# Dummy embedding function (embeddings already provided)
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# ---------------- COLLECTION ----------------
def get_collection():
    try:
        return chroma_client.get_collection(
            name="default",
            embedding_function=embedding_fn
        )
    except Exception:
        return chroma_client.create_collection(
            name="default",
            embedding_function=embedding_fn
        )

# ---------------- ADD DOCUMENT ----------------
@app.route("/add_doc", methods=["POST"])
def add_doc():
    data = request.get_json()

    text = data.get("text")
    vector = data.get("embedding")

    if not text or not vector:
        return jsonify({"error": "text and embedding are required"}), 400

    try:
        collection = get_collection()

        collection.add(
            documents=[text],
            embeddings=[vector],
            ids=[str(abs(hash(text)))]
        )

        chroma_client.persist()
        return jsonify({"status": "success"}), 200

    except Exception as e:
        print("Add doc error:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- QUERY DOCUMENTS ----------------
@app.route("/query_docs", methods=["POST"])
def query_docs():
    data = request.get_json()

    vector = data.get("vector")
    top_k = data.get("top_k", 3)

    if not vector:
        return jsonify({"error": "vector is required"}), 400

    try:
        collection = get_collection()

        results = collection.query(
            query_embeddings=[vector],
            n_results=top_k
        )

        docs = results["documents"][0] if results and "documents" in results else []
        return jsonify({"results": docs}), 200

    except Exception as e:
        print("Query error:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- HEALTH ----------------
@app.route("/")
def health():
    return "ChromaDB API Server Running (Single Collection)", 200

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
