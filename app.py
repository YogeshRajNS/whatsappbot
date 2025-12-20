# ==========================================
# WhatsApp PDF RAG Chatbot (Weaviate Cloud - SAFE)
# ==========================================

import os, threading, time
import fitz
import requests
from flask import Flask, request, make_response

# ===== Gemini NEW SDK =====
from google import genai

# ===== Weaviate v4 =====
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from weaviate.auth import AuthApiKey


# ---------------- CONFIG ----------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "verify_123")

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- GEMINI CLIENT ----------------
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------- WEAVIATE CLIENT (v4) ----------------
weaviate_client = WeaviateClient(
    connection_params=ConnectionParams.from_url(
        WEAVIATE_URL
    ),
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
    if not weaviate_client.collections.exists("PDFChunk"):
        weaviate_client.collections.create(
            name="PDFChunk",
            vectorizer_config=None,
            properties=[
                {"name": "text", "dataType": "text"}
            ]
        )

init_schema()

# ---------------- UTILS ----------------
def embed(text):
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]

    try:
        res = genai_client.models.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        EMBED_CACHE[text] = res["embedding"]
        return EMBED_CACHE[text]
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")

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
    collection = weaviate_client.collections.get("PDFChunk")

    with collection.batch.dynamic() as batch:
        for page in pdf:
            text = page.get_text().strip()
            for chunk in chunk_text(text):
                batch.add(
                    properties={"text": chunk},
                    vector=embed(chunk)
                )

    os.remove(path)
    return {"message": "PDF indexed successfully (Weaviate Cloud)"}

# ---------------- RETRIEVAL ----------------
def retrieve(query):
    q_vec = embed(query)
    collection = weaviate_client.collections.get("PDFChunk")

    res = collection.query.near_vector(
        near_vector=q_vec,
        limit=1,
        certainty=0.6
    )

    if not res.objects:
        return []

    return [res.objects[0].properties["text"]]

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

        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text.strip() if response and response.text else None

    except Exception as e:
        print("Gemini error:", e)
        return f"⚠️ Error generating answer: {e}"

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
    send_whatsapp(phone, answer or "⚠️ Unable to generate an answer right now.")

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
    return "WhatsApp RAG Bot Running (Weaviate Cloud v4)", 200

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
