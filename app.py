# ==========================================
# WhatsApp PDF RAG Chatbot (Production Safe)
# ==========================================

import os, json, threading, time
import fitz
import numpy as np
import requests
from flask import Flask, request, make_response
import google.generativeai as genai

# ---------------- CONFIG ----------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "verify_123")

UPLOAD_DIR = "uploads"
STORE_FILE = "vector_store.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- GLOBAL STATE ----------------
VECTOR_STORE = []
EMBED_CACHE = {}
LAST_MESSAGE = {}

SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank you", "ok"}

MAX_DOC_CHARS = 1500
SIM_THRESHOLD = 0.6
RATE_LIMIT_SECONDS = 5

# ---------------- VECTOR STORE ----------------
def load_store():
    if os.path.exists(STORE_FILE):
        with open(STORE_FILE, "r") as f:
            return json.load(f)
    return []

def save_store(data):
    with open(STORE_FILE, "w") as f:
        json.dump(data, f)

VECTOR_STORE = load_store()

# ---------------- UTILS ----------------
def embed(text):
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]

    emb = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )["embedding"]

    EMBED_CACHE[text] = emb
    return emb

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def chunk_text(text, size=400):
    return [text[i:i + size] for i in range(0, len(text), size)]

# ---------------- PDF INGEST ----------------
@app.route("/upload_file", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return {"error": "No file uploaded"}, 400

    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    pdf = fitz.open(path)

    for page in pdf:
        text = page.get_text().strip()
        for chunk in chunk_text(text):
            VECTOR_STORE.append({
                "text": chunk,
                "vector": embed(chunk)
            })

    save_store(VECTOR_STORE)
    os.remove(path)

    return {"message": "PDF indexed successfully"}

# ---------------- RETRIEVAL ----------------
def retrieve(query):
    q_vec = embed(query)

    scored = [
        (cosine(q_vec, item["vector"]), item["text"])
        for item in VECTOR_STORE
    ]

    scored.sort(reverse=True)

    if not scored or scored[0][0] < SIM_THRESHOLD:
        return []

    return [scored[0][1]]  # top_k = 1

# ---------------- GEMINI ANSWER ----------------
def generate_answer(query, docs):
    try:
        docs_text = "\n".join(docs)[:MAX_DOC_CHARS]

        prompt = f"""
Answer ONLY using the document below.
If the answer is not present, say "Not available in the document".

Document:
{docs_text}

Question:
{query}
"""

        model = genai.GenerativeModel("models/gemini-2.0-flash")
        res = model.generate_content(prompt)

        if not res or not res.text:
            return None

        return res.text.strip()

    except Exception as e:
        # Log error internally (Render logs)
        print("Gemini error:", e)
        return None

# ---------------- WHATSAPP SEND ----------------
def send_whatsapp(to, text):
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text[:4096]}
    }
    requests.post(url, headers=headers, json=payload)

# ---------------- MESSAGE WORKER ----------------
def process_message(phone, text):
    now = time.time()

    # Rate limit
    if phone in LAST_MESSAGE and now - LAST_MESSAGE[phone] < RATE_LIMIT_SECONDS:
        send_whatsapp(phone, "⏳ Please wait a few seconds before asking again.")
        return
    LAST_MESSAGE[phone] = now

    # Small talk bypass
    if text.lower().strip() in SMALL_TALK:
        send_whatsapp(phone, "Hi 👋 Ask a question about the uploaded PDF.")
        return

    if not VECTOR_STORE:
        send_whatsapp(phone, "No PDF uploaded yet. Please upload a PDF first.")
        return

    docs = retrieve(text)
    if not docs:
        send_whatsapp(phone, "No relevant information found in the document.")
        return

    answer = generate_answer(text, docs)

    if not answer:
        send_whatsapp(
            phone,
            "⚠️ I'm unable to generate an answer right now. Please try again shortly."
        )
        return

    send_whatsapp(phone, answer)

# ---------------- WEBHOOK VERIFY ----------------
@app.route("/webhook", methods=["GET"])
def verify():
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return make_response(request.args.get("hub.challenge"), 200)
    return "Forbidden", 403

# ---------------- WEBHOOK RECEIVE ----------------
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    print("Incoming payload:", json.dumps(data))

    try:
        entry = data.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages")

        if messages:
            msg = messages[0]
            phone = msg.get("from")

            if msg.get("type") == "text":
                user_text = msg.get("text", {}).get("body")

                threading.Thread(
                    target=process_message,
                    args=(phone, user_text),
                    daemon=True
                ).start()

    except Exception as e:
        print("Webhook parse error:", e)

    return "ok", 200

# ---------------- HEALTH ----------------
@app.route("/")
def health():
    return "WhatsApp RAG Bot Running", 200

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
