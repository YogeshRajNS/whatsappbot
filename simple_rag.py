import os
import json
import re
import fitz
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

# ------------------ CONFIG ------------------

with open("api.json", "r") as f:
    api = json.load(f)

genai.configure(api_key=api["api_key"])

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------ APP ------------------

app = Flask(__name__)
CORS(app)

# ------------------ DOC EXTRACTOR ------------------

class docExtractor:
    def __init__(self, collection_name="doc_collection"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_store")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

    def pdf_extractor(self, file_path):
        pdf_data = fitz.open(file_path)
        data = []
        for page in pdf_data:
            data.append({
                "page_number": page.number + 1,
                "text": page.get_text()
            })
        return data

    def create_embeddings(self, pages):
        for page in pages:
            page["vector"] = self.model.encode(page["text"])
        return pages

    def store_to_chromadb(self, data, doc_name):
        for item in data:
            self.collection.add(
                documents=[item["text"]],
                embeddings=[item["vector"]],
                metadatas=[{
                    "page_number": item["page_number"],
                    "doc_name": doc_name
                }],
                ids=[f"{doc_name}_page_{item['page_number']}"]
            )

    def retrieve(self, query, file_names, top_k=3):
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

        ids = results["ids"][0]
        docs = results["documents"][0]
        return dict(zip(ids, docs))

    def retrieve_doc_name_list(self):
        results = self.collection.get(include=["metadatas"])
        doc_names = [m["doc_name"] for m in results["metadatas"] if "doc_name" in m]
        return sorted(set(doc_names))

    def delete_docs(self, doc_names):
        results = self.collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(results["ids"], results["metadatas"])
            if meta.get("doc_name") in doc_names
        ]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return ids_to_delete

# ------------------ GEMINI HELPERS ------------------

def check_with_gemini(prompt):
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt).text

    match = re.search(r"\$\$(.*?)\$\$", response, re.DOTALL)
    if match:
        response = match.group(1)

    response = re.sub(r"```", "", response).strip()
    return response

def answer_with_gemini(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs.values())

    prompt = f"""
You are a helpful assistant. Answer strictly from the document.

Document Content:
{context}

Question:
{query}

Rules:
- Use only document info
- No outside knowledge
- Markdown output
"""

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt, stream=True)

    for chunk in response:
        yield chunk.text

# ------------------ ROUTES ------------------

@app.route("/upload_file", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(file_path)

    extractor = docExtractor(collection_name="my_doc_2")
    pages = extractor.pdf_extractor(file_path)
    pages = extractor.create_embeddings(pages)
    extractor.store_to_chromadb(pages, file.filename)

    os.remove(file_path)

    return jsonify({"message": f"{file.filename} uploaded successfully"})

@app.route("/list_docs", methods=["GET"])
def list_docs():
    extractor = docExtractor(collection_name="my_doc_2")
    return jsonify({"docs": extractor.retrieve_doc_name_list()})

@app.route("/delete_docs", methods=["DELETE"])
def delete_docs():
    data = request.json
    extractor = docExtractor(collection_name="my_doc_2")
    deleted = extractor.delete_docs(data.get("docs", []))
    return jsonify({"deleted_ids": deleted})

@app.route("/query", methods=["POST"])
def query_docs():
    data = request.json

    question = data["query"]
    history = data.get("message_history", "")
    docs = data.get("docs", [])

    prompt = f"""
<message_history>
{history}
</message_history>
<user_question>
{question}
</user_question>

Return $$rephrased_question$$ or $$None$$
"""

    reformulated = check_with_gemini(prompt)
    if reformulated.lower() != "none":
        question = reformulated

    extractor = docExtractor(collection_name="my_doc_2")
    results = extractor.retrieve(question, docs)

    return Response(
        stream_with_context(answer_with_gemini(question, results)),
        content_type="text/plain"
    )

# ------------------ RUN ------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
