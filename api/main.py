import os
from typing import List

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
app = FastAPI(title="Windsor Knowledge API")

PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", 5432)),
    dbname=os.getenv("PGDATABASE", "windsor"),
    user=os.getenv("PGUSER", "windsor"),
    password=os.getenv("PGPASSWORD", "password"),
)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4o")
HTML_DIR    = os.getenv("HTML_DIR", "./data/html")  # where crawler writes

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Serve assets so the UI can show images (in prod, front this with auth)
assets_path = os.path.join(HTML_DIR, "assets")
if os.path.isdir(assets_path):
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

def q_conn():
    return psycopg2.connect(**PGCFG)

def embed(q: str):
    return client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search")
def search(q: str, k: int = 8):
    v = embed(q)
    with q_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT d.title, d.url, d.space_key, d.assets, c.text,
                   1 - (c.embedding <=> %s::vector) AS score
              FROM document_chunks c
              JOIN documents d ON d.id = c.doc_id
             ORDER BY c.embedding <=> %s::vector
             LIMIT %s;
        """, (v, v, k))
        rows = cur.fetchall()
    return {"results": rows}

@app.get("/chat")
def chat(q: str, k: int = 8):
    hits = search(q, k=k)["results"]
    context = "\n\n---\n\n".join([f"Title: {h['title']}\n{h['text']}" for h in hits])
    sys_prompt = (
        "You are a helpful internal assistant. Answer using the provided context only. "
        "Cite page titles if helpful. If unsure, say you don't know."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": f"Question: {q}\n\nContext:\n{context}"}
    ]
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return {"answer": resp.choices[0].message.content, "sources": hits}
