# ==========================================
# WhatsApp PDF RAG Chatbot (ChromaDB + Gemini)
# ==========================================

import os, threading, time
import pdfplumber
import requests
from flask import Flask, request, make_response
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ---------------- CONFIG ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- GEMINI CLIENT ----------------
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------- GLOBAL STATE ----------------
EMBED_CACHE = {}
LAST_MESSAGE = {}
SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank you", "ok"}
MAX_DOC_CHARS = 1500
RATE_LIMIT_SECONDS = 5

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

def send_whatsapp(to, text, token):
    try:
        url = f"https://graph.facebook.com/v20.0/{to}/messages"
        headers = {
            "Authorization": f"Bearer {token}",
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

# ---------------- CHROMADB UTILS ----------------
CHROMA_API_URL = os.getenv("CHROMA_API_URL")  # Your deployed ChromaDB API endpoint

def store_document(client_id, text, embedding):
    # POST to ChromaDB API
    try:
        payload = {"client_id": client_id, "text": text, "embedding": embedding}
        res = requests.post(f"{CHROMA_API_URL}/add_doc", json=payload, timeout=10)
        return res.status_code == 200
    except Exception as e:
        print("ChromaDB store error:", e)
        return False

def retrieve_documents(client_id, query_vector, top_k=3):
    try:
        payload = {"client_id": client_id, "vector": query_vector, "top_k": top_k}
        res = requests.post(f"{CHROMA_API_URL}/query_docs", json=payload, timeout=10)
        if res.status_code == 200:
            return res.json().get("results", [])
        return []
    except Exception as e:
        print("ChromaDB retrieve error:", e)
        return []

# ---------------- PDF UPLOAD ----------------
@app.route("/upload_file", methods=["POST"])
def upload_file():
    client_id = request.form.get("client_id")
    if not client_id:
        return {"error": "client_id is required"}, 400

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
                    vec = embed(chunk)
                    if vec is None:
                        continue
                    store_document(client_id, chunk, vec)
        return {"message": "PDF indexed successfully"}
    except Exception as e:
        print("Upload error:", e)
        return {"error": str(e)}, 500
    finally:
        if os.path.exists(path):
            os.remove(path)

# ---------------- MESSAGE WORKER ----------------
def process_message(client_id, customer_phone, customer_token, text):
    now = time.time()
    key = f"{client_id}-{customer_phone}"
    if key in LAST_MESSAGE and now - LAST_MESSAGE[key] < RATE_LIMIT_SECONDS:
        send_whatsapp(customer_phone, "⏳ Please wait a few seconds.", customer_token)
        return
    LAST_MESSAGE[key] = now

    if text.lower().strip() in SMALL_TALK:
        send_whatsapp(customer_phone, "Hi 👋 How can I help you?", customer_token)
        return

    vec = embed(text)
    docs = retrieve_documents(client_id, vec)
    docs_text = "\n".join(docs)[:MAX_DOC_CHARS] if docs else ""
    
    prompt = f"""
Answer ONLY using the document below.
If not found, say "Not available in the document".

Document:
{docs_text}

Question:
{text}
"""
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        answer = response.text.strip() if response and response.text else "⚠️ Unable to generate answer."
    except Exception as e:
        print("Gemini error:", e)
        answer = "⚠️ Unable to generate answer."
    
    send_whatsapp(customer_phone, answer, customer_token)

# ---------------- WEBHOOK ----------------
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    try:
        entry = data.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})

        # Client identification
        client_id = value.get("metadata", {}).get("phone_number_id")
        # WhatsApp token for this client
        client_token = os.getenv("CLIENT_WHATSAPP_TOKEN")  # OR pass dynamically via front-end

        messages = value.get("messages", [])
        for msg in messages:
            customer_phone = msg.get("from")
            text = msg.get("text", {}).get("body")
            threading.Thread(
                target=process_message,
                args=(client_id, customer_phone, client_token, text),
                daemon=True
            ).start()
    except Exception as e:
        print("Webhook parse error:", e)
    return "ok", 200

@app.route("/webhook", methods=["GET"])
def verify():
    VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "verify_123")
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return make_response(request.args.get("hub.challenge"), 200)
    return "Forbidden", 403

# ---------------- HEALTH ----------------
@app.route("/")
def health():
    return "WhatsApp RAG Bot Running (ChromaDB + Gemini AI)", 200

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
