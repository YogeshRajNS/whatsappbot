# ==========================================
# WhatsApp PDF RAG Chatbot (ChromaDB via API)
# ==========================================

import os
import threading
import time
import requests
import pdfplumber
from flask import Flask, request, make_response
from dotenv import load_dotenv

# ===== Gemini SDK =====
from google import genai

load_dotenv()

# ---------------- CONFIG ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "verify_123")

# 🔥 Chroma API (hosted separately)
CHROMA_API_URL = os.getenv("CHROMA_API_URL", "http://localhost:5001")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_DOC_CHARS = 1500
RATE_LIMIT_SECONDS = 5
SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank you", "ok"}

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- GEMINI CLIENT ----------------
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------- GLOBAL STATE ----------------
EMBED_CACHE = {}
LAST_MESSAGE = {}

# ---------------- UTILS ----------------
def embed(text):
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]

    try:
        res = genai_client.models.embed_content(
            model="models/text-embedding-004",
            contents=[text]
        )
        vector = res.embeddings[0].values
        EMBED_CACHE[text] = vector
        return vector
    except Exception as e:
        print("Embedding error:", e)
        return None


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

    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                for chunk in chunk_text(text):
                    vector = embed(chunk)
                    if not vector:
                        continue

                    payload = {
                        "text": chunk,
                        "embedding": vector
                    }

                    # 🔥 Send to ChromaDB API
                    requests.post(
                        f"{CHROMA_API_URL}/add_doc",
                        json=payload,
                        timeout=10
                    )

        return {"message": "PDF indexed successfully using ChromaDB"}

    except Exception as e:
        print("Upload error:", e)
        return {"error": str(e)}, 500

    finally:
        if os.path.exists(path):
            os.remove(path)


# ---------------- RETRIEVAL ----------------
def retrieve(query, k=3):
    vector = embed(query)
    if not vector:
        return []

    payload = {
        "vector": vector,
        "top_k": k
    }

    try:
        res = requests.post(
            f"{CHROMA_API_URL}/query_docs",
            json=payload,
            timeout=10
        )
        data = res.json()
        return data.get("results", [])
    except Exception as e:
        print("Chroma retrieval error:", e)
        return []


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
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip() if response and response.text else None
    except Exception as e:
        print("Gemini error:", e)
        return None


# ---------------- WHATSAPP SEND ----------------
def send_whatsapp(to, text):
    try:
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
        requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        print("WhatsApp send error:", e)


# ---------------- MESSAGE WORKER ----------------
def process_message(phone, text):
    now = time.time()
    if phone in LAST_MESSAGE and now - LAST_MESSAGE[phone] < RATE_LIMIT_SECONDS:
        send_whatsapp(phone, "⏳ Please wait a few seconds.")
        return
    LAST_MESSAGE[phone] = now

    if text.lower().strip() in SMALL_TALK:
        send_whatsapp(phone, "Hi 👋 How can I help you?")
        return

    docs = retrieve(text)
    if not docs:
        send_whatsapp(phone, "No relevant information found.")
        return

    answer = generate_answer(text, docs)
    send_whatsapp(phone, answer or "⚠️ Unable to generate answer.")


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
    try:
        entry = data.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages")

        if messages:
            msg = messages[0]
            phone = msg.get("from")
            if msg.get("type") == "text":
                text = msg.get("text", {}).get("body")
                threading.Thread(
                    target=process_message,
                    args=(phone, text),
                    daemon=True
                ).start()
    except Exception as e:
        print("Webhook parse error:", e)

    return "ok", 200


# ---------------- HEALTH ----------------
@app.route("/")
def health():
    return "WhatsApp RAG Bot Running (ChromaDB API)", 200


# ---------------- RUN (Railway) ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
