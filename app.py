# ==========================================
# WhatsApp PDF RAG Chatbot (Weaviate Cloud)
# ==========================================

import os, threading, time, json
import fitz
import requests
from flask import Flask, request, make_response
import google.generativeai as genai
import weaviate
from weaviate.auth import AuthApiKey

# ---------------- CONFIG ----------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "verify_123")

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- WEAVIATE CLIENT ----------------
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(WEAVIATE_API_KEY)
)

# ---------------- GLOBAL STATE ----------------
EMBED_CACHE = {}
LAST_MESSAGE = {}

SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank you", "ok"}
MAX_DOC_CHARS = 1500
RATE_LIMIT_SECONDS = 5

# ---------------- SCHEMA INIT ----------------
def init_schema():
    if not client.schema.exists("PDFChunk"):
        client.schema.create_class({
            "class": "PDFChunk",
            "vectorizer": "none",
            "properties": [
                {"name": "text", "dataType": ["text"]}
            ]
        })

init_schema()

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

    with client.batch as batch:
        batch.batch_size = 50

        for page in pdf:
            text = page.get_text().strip()
            for chunk in chunk_text(text):
                batch.add_data_object(
                    data_object={"text": chunk},
                    class_name="PDFChunk",
                    vector=embed(chunk)
                )

    os.remove(path)
    return {"message": "PDF indexed successfully (Weaviate)"}

# ---------------- RETRIEVAL ----------------
def retrieve(query):
    q_vec = embed(query)

    res = client.query.get("PDFChunk", ["text"]) \
        .with_near_vector({"vector": q_vec, "certainty": 0.6}) \
        .with_limit(1) \
        .do()

    items = res.get("data", {}).get("Get", {}).get("PDFChunk", [])
    return [item["text"]] if items else []

# ---------------- GEMINI ANSWER ----------------
def generate_answer(query, docs):
    try:
        docs_text = "\n".join(docs)[:MAX_DOC_CHARS]

        prompt = f"""
Answer ONLY using the document below.
If not found, say "Not available in the document".

Document:
{docs_text}

Question:
{query}
"""

        model = genai.GenerativeModel("models/gemini-2.0-flash")
        res = model.generate_content(prompt)

        return res.text.strip() if res and res.text else None

    except Exception as e:
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

    if phone in LAST_MESSAGE and now - LAST_MESSAGE[phone] < RATE_LIMIT_SECONDS:
        send_whatsapp(phone, "⏳ Please wait a few seconds.")
        return
    LAST_MESSAGE[phone] = now

    if text.lower().strip() in SMALL_TALK:
        send_whatsapp(phone, "Hi 👋 Ask a question about the PDF.")
        return

    docs = retrieve(text)
    if not docs:
        send_whatsapp(phone, "No relevant information found in the document.")
        return

    answer = generate_answer(text, docs)
    send_whatsapp(
        phone,
        answer or "⚠️ Unable to generate an answer right now."
    )

# ---------------- WEBHOOK ----------------
@app.route("/webhook", methods=["GET"])
def verify():
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return make_response(request.args.get("hub.challenge"), 200)
    return "Forbidden", 403

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()

    try:
        entry = data.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages")

        if messages:
            msg = messages[0]
            if msg.get("type") == "text":
                threading.Thread(
                    target=process_message,
                    args=(msg["from"], msg["text"]["body"]),
                    daemon=True
                ).start()

    except Exception as e:
        print("Webhook error:", e)

    return "ok", 200

# ---------------- HEALTH ----------------
@app.route("/")
def health():
    return "WhatsApp RAG Bot Running (Weaviate)", 200

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
