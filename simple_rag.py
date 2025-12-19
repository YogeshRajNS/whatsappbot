import os
import re
import fitz
import requests

from flask import Flask, request, jsonify, Response, stream_with_context, make_response
from flask_cors import CORS

from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

# ================== CONFIG ==================

# Gemini API (Render ENV)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "salon_verify_token")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

UPLOAD_DIR = "./uploads"
CHROMA_DIR = "/tmp/chroma_store"

os.makedirs(UPLOAD_DIR, exist_ok=True)
if os.path.exists(CHROMA_DIR):
    if not os.path.isdir(CHROMA_DIR):
        raise RuntimeError(f"{CHROMA_DIR} exists but is not a directory")
else:
    os.mkdir(CHROMA_DIR)
# ================== APP ==================

app = Flask(__name__)
CORS(app)

# ================== GLOBAL MODELS (IMPORTANT) ==================

embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# ================== DOC EXTRACTOR ==================

class docExtractor:
    def __init__(self, collection_name="doc_collection"):
        self.model = get_embedding_model()
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name
        )

    def pdf_extractor(self, file_path):
        pdf_data = fitz.open(file_path)
        return [
            {"page_number": p.number + 1, "text": p.get_text()}
            for p in pdf_data
        ]

    def create_embeddings(self, pages):
        for p in pages:
            p["vector"] = self.model.encode(p["text"]).tolist()
        return pages

    def store_to_chromadb(self, pages, doc_name):
        for p in pages:
            self.collection.add(
                documents=[p["text"]],
                embeddings=[p["vector"]],
                metadatas=[{
                    "page_number": p["page_number"],
                    "doc_name": doc_name
                }],
                ids=[f"{doc_name}_page_{p['page_number']}"]
            )

    def retrieve(self, query, file_names=None, top_k=3):
        query_embedding = self.model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"doc_name": {"$in": file_names}} if file_names else None
        )

        return dict(zip(results["ids"][0], results["documents"][0]))

    def retrieve_doc_name_list(self):
        results = self.collection.get(include=["metadatas"])
        return sorted(set(m["doc_name"] for m in results["metadatas"]))

    def delete_docs(self, doc_names):
        results = self.collection.get(include=["metadatas"])
        ids = [
            doc_id for doc_id, meta in zip(results["ids"], results["metadatas"])
            if meta.get("doc_name") in doc_names
        ]
        if ids:
            self.collection.delete(ids=ids)
        return ids

# ================== GEMINI ==================

def answer_with_gemini(query, docs):
    context = "\n\n".join(docs.values())

    prompt = f"""
Answer strictly from the document.

Document:
{context}

Question:
{query}
"""

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt, stream=True)

    for chunk in response:
        yield chunk.text

# ================== WHATSAPP ==================

def send_whatsapp_message(to, text):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text[:4096]}  # WhatsApp limit safety
    }
    requests.post(url, headers=headers, json=payload, timeout=5)

# ================== ROUTES ==================

@app.route("/upload_file", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file"}), 400

    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    extractor = docExtractor("my_doc_2")
    pages = extractor.create_embeddings(
        extractor.pdf_extractor(path)
    )
    extractor.store_to_chromadb(pages, file.filename)

    os.remove(path)
    return jsonify({"message": "Uploaded successfully"})

@app.route("/query", methods=["POST"])
def query_docs():
    data = request.json
    extractor = docExtractor("my_doc_2")
    results = extractor.retrieve(data["query"], data.get("docs"))

    return Response(
        stream_with_context(answer_with_gemini(data["query"], results)),
        content_type="text/plain"
    )

# ================== WHATSAPP WEBHOOK ==================

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return make_response(request.args.get("hub.challenge"), 200)
    return "Verification failed", 403

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    try:
        msg = request.json["entry"][0]["changes"][0]["value"]["messages"][0]
        user_text = msg["text"]["body"]
        user_phone = msg["from"]
    except:
        return "ok", 200

    extractor = docExtractor("my_doc_2")
    results = extractor.retrieve(user_text)

    answer = "".join(answer_with_gemini(user_text, results))
    send_whatsapp_message(user_phone, answer)

    return "ok", 200

# ================== RENDER PORT ==================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )






