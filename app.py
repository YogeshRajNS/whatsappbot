# ==========================================
# WhatsApp PDF RAG Chatbot (Render Safe)
# ==========================================

import os, json, threading
import fitz
import numpy as np
import requests
from flask import Flask, request, jsonify, make_response
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
    emb = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return emb["embedding"]

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---------------- PDF INGEST ----------------
@app.route("/upload_file", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return {"error": "No file"}, 400

    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    pdf = fitz.open(path)
    for page in pdf:
        text = page.get_text().strip()
        if text:
            VECTOR_STORE.append({
                "text": text,
                "vector": embed(text)
            })

    save_store(VECTOR_STORE)
    os.remove(path)

    return {"message": "PDF indexed successfully"}

# ---------------- RETRIEVAL ----------------
def retrieve(query, top_k=3):
    q_vec = embed(query)
    scored = [
        (cosine(q_vec, item["vector"]), item["text"])
        for item in VECTOR_STORE
    ]
    scored.sort(reverse=True)
    return [t for _, t in scored[:top_k]]

# ---------------- GEMINI ANSWER ----------------
def generate_answer(query, docs):
    prompt = f"""
Answer ONLY from the document content.

Document:
{chr(10).join(docs)}

Question:
{query}
"""
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    res = model.generate_content(prompt)
    return res.text.strip()

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

# ---------------- BACKGROUND WORKER ----------------
def process_message(phone, text):
    if not VECTOR_STORE:
        send_whatsapp(phone, "No PDF uploaded yet. Please upload a PDF first.")
        return

    docs = retrieve(text)
    if not docs:
        send_whatsapp(phone, "No relevant information found in the document.")
        return

    answer = generate_answer(text, docs)
    send_whatsapp(phone, answer or "Unable to generate answer.")

# ---------------- WEBHOOK ----------------
@app.route("/webhook", methods=["GET"])
def verify():
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return make_response(request.args.get("hub.challenge"), 200)
    return "Forbidden", 403

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    
    # Debug: This lets you see the payload in your Render logs
    print(f"Incoming Payload: {json.dumps(data)}")

    try:
        # Step-by-step extraction based on your specific JSON:
        entry = data.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        
        # Meta sends 'statuses' (sent/delivered) and 'messages' (incoming text).
        # We only care about 'messages'.
        messages = value.get("messages")

        if messages:
            message = messages[0]
            phone = message.get("from") # In your case: "918121676994"
            
            # Check if the message type is 'text'
            if message.get("type") == "text":
                user_text = message.get("text", {}).get("body") # In your case: "What is in the pdf"
                
                print(f"Found Message: '{user_text}' from {phone}")

                # Start your RAG/Gemini logic in a background thread
                threading.Thread(
                    target=process_message,
                    args=(phone, user_text)
                ).start()

    except Exception as e:
        print(f"Error parsing JSON: {e}")

    # ALWAYS return 200 OK to Meta immediately
    return "ok", 200

# ---------------- HEALTH ----------------
@app.route("/")
def health():
    return "WhatsApp RAG Bot Running", 200

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
