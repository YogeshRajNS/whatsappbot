import os
import fitz
import requests
import numpy as np

from flask import Flask, request, jsonify, Response, stream_with_context, make_response
from flask_cors import CORS

import google.generativeai as genai

# ================== CONFIG ==================

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "salon_verify_token")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================== APP ==================

app = Flask(__name__)
CORS(app)

# ================== SIMPLE VECTOR STORE ==================
# Structure:
# {
#   "doc_name": [
#       {"page": 1, "text": "...", "vector": [...]},
#       ...
#   ]
# }

VECTOR_STORE = {}

# ================== UTILS ==================

def embed_text(text: str):
    """Gemini remote embedding"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return np.array(result["embedding"], dtype=np.float32)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ================== DOC EXTRACTOR ==================

class DocExtractor:

    def pdf_extractor(self, file_path):
        pdf = fitz.open(file_path)
        return [
            {"page": p.number + 1, "text": p.get_text()}
            for p in pdf
        ]

    def index_document(self, pages, doc_name):
        vectors = []
        for p in pages:
            vectors.append({
                "page": p["page"],
                "text": p["text"],
                "vector": embed_text(p["text"])
            })
        VECTOR_STORE[doc_name] = vectors

    def retrieve(self, query, doc_names=None, top_k=3):
        query_vec = embed_text(query)
        scores = []

        for doc, pages in VECTOR_STORE.items():
            if doc_names and doc not in doc_names:
                continue
            for p in pages:
                score = cosine_sim(query_vec, p["vector"])
                scores.append((score, p["text"]))

        scores.sort(reverse=True, key=lambda x: x[0])
        return {f"doc_{i}": text for i, (_, text) in enumerate(scores[:top_k])}

# ================== GEMINI ANSWER ==================

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
        if chunk.text:
            yield chunk.text

# ================== WHATSAPP ==================

def send_whatsapp_message(to):
    url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {
            "name": "jaspers_market_plain_text_v1",
            "language": {"code": "en_US"}
        }
    }
    resp = requests.post(url, headers=headers, json=payload)
    print(resp.status_code, resp.text)


# ================== ROUTES ==================

@app.route("/")
def health():
    return "Service running", 200

@app.route("/upload_file", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file"}), 400

    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    extractor = DocExtractor()
    pages = extractor.pdf_extractor(path)
    extractor.index_document(pages, file.filename)

    os.remove(path)
    return jsonify({"message": "Uploaded successfully"})

@app.route("/query", methods=["POST"])
def query_docs():
    data = request.json
    extractor = DocExtractor()
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

        # Check if VECTOR_STORE has content
        if not VECTOR_STORE:
            send_whatsapp_message(user_phone, "No PDFs uploaded yet. Please upload a PDF first.")
            return "ok", 200

        # Retrieve relevant document content
        extractor = DocExtractor()
        results = extractor.retrieve(user_text)

        if not results:
            send_whatsapp_message(user_phone, "Sorry, I could not find anything relevant in the PDFs.")
            return "ok", 200

        # Generate full answer from Gemini (no streaming)
        context = "\n\n".join(results.values())
        prompt = f"""
Answer strictly from the document.

Document:
{context}

Question:
{user_text}
"""
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt, stream=False)  # full answer returned
        answer = response.text if hasattr(response, "text") else str(response)

        if not answer.strip():
            answer = "Sorry, I could not generate an answer."

        # Send the answer
        send_whatsapp_message(user_phone, answer)

    except Exception as e:
        print("Webhook error:", e)
        send_whatsapp_message(user_phone, "Something went wrong while processing your message.")

    return "ok", 200



# ================== LOCAL / NGROK ==================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)





