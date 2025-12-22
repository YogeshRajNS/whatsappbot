# ==========================================
# WhatsApp PDF RAG Chatbot (Weaviate Cloud v5+)
# ==========================================

import os, threading, time
import fitz
import requests
from flask import Flask, request, make_response
from dotenv import load_dotenv

# ===== Gemini NEW SDK =====
from google import genai

# ===== Weaviate v5+ =====
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure

load_dotenv()

# ---------------- CONFIG ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "verify_123")

WEAVIATE_URL = os.getenv("WEAVIATE_URL")  # Full URL with https://
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- GEMINI CLIENT ----------------
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------- WEAVIATE CLIENT ----------------
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=AuthApiKey(WEAVIATE_API_KEY)
)

if not weaviate_client.is_ready():
    raise RuntimeError("Weaviate client not ready")

# ---------------- GLOBAL STATE ----------------
EMBED_CACHE = {}
LAST_MESSAGE = {}
SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank you", "ok"}
MAX_DOC_CHARS = 1500
RATE_LIMIT_SECONDS = 5

# ---------------- SCHEMA INIT ----------------
def init_schema():
    try:
        pdf_collection = weaviate_client.collections.use("PDFChunk")
        print("Using existing PDFChunk collection")
    except Exception as e:
        print("Collection not found:", e)
init_schema()

# ---------------- UTILS ----------------
def embed(text):
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]

    try:
        res = genai_client.models.embed_content(
            model="models/text-embedding-004",
            contents=[text]
        )

        # ✅ IMPORTANT: extract vector values
        vector = res.embeddings[0].values

        EMBED_CACHE[text] = vector
        return vector

    except Exception as e:
        print("Embedding error:", e)
        return None


def chunk_text(text, size=400):
    return [text[i:i + size] for i in range(0, len(text), size)]

def extract_text(page):
    text = page.get_text("text").strip()
    if len(text) < 50:  # garbage protection (ID, page no, etc.)
        blocks = page.get_text("blocks")
        text = " ".join(b[4] for b in blocks if b[4].strip())
    return text

# ---------------- PDF INGEST ----------------
@app.route("/upload_file", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return {"error": "No file uploaded"}, 400

    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    pdf = None
    try:
        pdf = fitz.open(path)
        pdf_collection = weaviate_client.collections.use("PDFChunk")

        with pdf_collection.batch.fixed_size(batch_size=200) as batch:
            for page in pdf:
                text = extract_text(page).strip()
                if not text:
                    continue
                for chunk in chunk_text(text):
                    # Let the collection vectorizer handle vectorization
                    batch.add_object(properties={"text": chunk})

        return {"message": "PDF indexed successfully"}

    except Exception as e:
        print("Upload error:", e)
        return {"error": str(e)}, 500

    finally:
        if pdf:
            pdf.close()
        if os.path.exists(path):
            os.remove(path)

# ---------------- RETRIEVAL ----------------
# ---------------- RETRIEVAL ----------------
def retrieve(query):
    try:
        pdf_collection = weaviate_client.collections.use("PDFChunk")
        response = pdf_collection.query.near_text(query, limit=1)

        objs = response.objects
        if not objs:
            return []

        return [objs[0].properties["text"]]

    except Exception as e:
        print("Retrieve error:", e)
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
        send_whatsapp(phone, "Hi 👋 Ask a question about the PDF.")
        return

    docs = retrieve(text)
    if not docs:
        send_whatsapp(phone, "No relevant information found.")
        return

    answer = generate_answer(text, docs)
    send_whatsapp(phone, answer or "⚠️ Unable to generate answer.")

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
    return "WhatsApp RAG Bot Running (Weaviate v5+)", 200

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
