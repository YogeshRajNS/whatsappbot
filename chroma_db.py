# ==========================================
# ChromaDB API Server for WhatsApp RAG Bot
# ==========================================

import os
import json
from flask import Flask, request, jsonify
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ---------------- CONFIG ----------------
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")  # local folder for Chroma
EMBEDDING_DIM = 768  # same as Gemini embedding dim

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- CHROMADB CLIENT ----------------
chroma_client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=CHROMA_DB_DIR
))

# We'll use a dummy embedding function since embeddings come from Gemini
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# ---------------- HELPERS ----------------
def get_collection(client_id):
    # Collection name per client to isolate data
    col_name = f"client_{client_id}"
    try:
        collection = chroma_client.get_collection(name=col_name)
    except Exception:
        collection = chroma_client.create_collection(name=col_name, embedding_function=embedding_fn)
    return collection

# ---------------- ADD DOCUMENT ----------------
@app.route("/add_doc", methods=["POST"])
def add_doc():
    data = request.get_json()
    client_id = data.get("client_id")
    text = data.get("text")
    vector = data.get("embedding")

    if not client_id or not text or not vector:
        return jsonify({"error": "client_id, text, and embedding are required"}), 400

    collection = get_collection(client_id)
    # Add document as individual chunk
    collection.add(
        documents=[text],
        metadatas=[{"client_id": client_id}],
        ids=[f"{client_id}_{hash(text)}"],
        embeddings=[vector]
    )
    # Persist DB
    chroma_client.persist()
    return jsonify({"status": "success"}), 200

# ---------------- QUERY DOCUMENTS ----------------
@app.route("/query_docs", methods=["POST"])
def query_docs():
    data = request.get_json()
    client_id = data.get("client_id")
    vector = data.get("vector")
    top_k = data.get("top_k", 3)

    if not client_id or not vector:
        return jsonify({"error": "client_id and vector are required"}), 400

    collection = get_collection(client_id)
    results = collection.query(
        query_embeddings=[vector],
        n_results=top_k
    )

    # Return only document texts
    docs = results["documents"][0] if results and "documents" in results else []
    return jsonify({"results": docs}), 200

# ---------------- HEALTH ----------------
@app.route("/")
def health():
    return "ChromaDB API Server Running", 200

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # different port from main bot
    app.run(host="0.0.0.0", port=port)
