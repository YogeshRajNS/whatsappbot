import os
import json
import re
import fitz
import requests

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

# ================== CONFIG ==================

# Gemini API
with open("api.json", "r") as f:
    api = json.load(f)

genai.configure(api_key=api["api_key"])

# WhatsApp / Meta (SET THESE IN RENDER)
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "salon_verify_token")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================== APP ==================

app = Flask(__name__)
CORS(app)

# ================== DOC EXTRACTOR ==================

class docExtractor:
    def __init__(self, collection_name="doc_collection"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_store")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

    def pdf_extractor(self, file_path):
        pdf_data = fitz.open(file_path)
        pages = []
        for page in pdf_data:
            pages.append({
                "page_number": page.number + 1,
                "text": page.get_text()
            })
        return pages

    def create_embeddings(self, pages):
        for p in pages:
            p["vector"] = self.model.encode(p["text"])
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

        if file_names:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"doc_name": {"$in": file_names}},
            )
        else:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
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

# ================== GEMINI HELPERS ==================

def check_with_gemini(prompt):
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt).text

    match = re.search(r"\$\$(.*?)\$\$", response, re.DOTALL)
    if match:
        response = match.group(1)

    return re.sub(r"```", "", response).strip()

def answer_with_gemini(query, docs):
    context = "\n\n".join(docs.values())

    prompt = f"""
You are a helpful assistant.
Answer strictly from the document.

Document Content:
{context}

Question:
{query}

Rules:
- Use only document info
- No outside knowledge
"""

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt, stream=True)

    for chunk in response:
        yield chunk.text

# ================== WHATSAPP HELPERS ==================

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
        "text": {"body": text}
    }
    requests.post(url, headers=headers, json=payload)

# ================== ROUTES ==================

# ---------- FILE UPLOAD ----------
@app.route("/upload_file", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file"}), 400

    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    extractor = docExtractor("my_doc_2")
    pages = extractor.pdf_extractor(path)
    pages = extractor.create_embeddings(pages)
    extractor.store_to_chromadb(pages, file.filename)

    os.remove(path)
    return jsonify({"message": "Uploaded successfully"})

# ---------- LIST DOCS ----------
@app.route("/list_docs", methods=["GET"])
def list_docs():
    extractor = docExtractor("my_doc_2")
    return jsonify({"docs": extractor.retrieve_doc_name_list()})

# ---------- DELETE DOCS ----------
@app.route("/delete_docs", methods=["DELETE"])
def delete_docs():
    data = request.json
    extractor = docExtractor("my_doc_2")
    deleted = extractor.delete_docs(data.get("docs", []))
    return jsonify({"deleted": deleted})

# ---------- API QUERY ----------
@app.route("/query", methods=["POST"])
def query_docs():
    data = request.json
    question = data["query"]
    docs = data.get("docs", [])

    extractor = docExtractor("my_doc_2")
    results = extractor.retrieve(question, docs)

    return Response(
        stream_with_context(answer_with_gemini(question, results)),
        content_type="text/plain"
    )

# ================== WHATSAPP WEBHOOK ==================

from flask import make_response

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        # Must return plain text exactly
        return make_response(challenge, 200, {"Content-Type": "text/plain"})
    else:
        return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    data = request.json

    try:
        msg = data["entry"][0]["changes"][0]["value"]["messages"][0]
        user_text = msg["text"]["body"]
        user_phone = msg["from"]
    except:
        return "ok", 200

    extractor = docExtractor("my_doc_2")
    results = extractor.retrieve(user_text)

    answer = ""
    for chunk in answer_with_gemini(user_text, results):
        answer += chunk

    send_whatsapp_message(user_phone, answer)
    return "ok", 200

# ================== RUN ==================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

