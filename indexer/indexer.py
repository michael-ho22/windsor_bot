import os, sys, time, json, hashlib
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
HTML_DIR = Path(os.getenv("HTML_DIR", "./data/html"))  # same as crawler's EXPORT_DIR
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", 5432)),
    dbname=os.getenv("PGDATABASE", "windsor"),
    user=os.getenv("PGUSER", "windsor"),
    password=os.getenv("PGPASSWORD", "password"),
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def connect():
    return psycopg2.connect(**PGCFG)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)

def chunk_text(text: str, max_tokens=800, overlap=120):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = words[i:i+max_tokens]
        chunks.append(" ".join(chunk))
        i += max_tokens - overlap if (i + max_tokens) < len(words) else len(words)
    return chunks or [text]

def embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(e.embedding, dtype=np.float32) for e in resp.data]

def load_sidecar_for(html_path: Path):
    meta_path = html_path.with_suffix(".json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def upsert_document(cur, space_key, page_id, title, url, last_modified, html_path, sha, assets):
    cur.execute("""
        INSERT INTO documents (space_key, page_id, title, url, last_modified, html_path, sha256, assets)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
        ON CONFLICT (page_id) DO UPDATE
          SET title=EXCLUDED.title,
              url=EXCLUDED.url,
              last_modified=EXCLUDED.last_modified,
              html_path=EXCLUDED.html_path,
              sha256=EXCLUDED.sha256,
              assets=EXCLUDED.assets
        RETURNING id, sha256;
    """, (space_key, page_id, title, url, last_modified, html_path, sha, json.dumps(assets)))
    return cur.fetchone()

def existing_sha_for(cur, page_id):
    cur.execute("SELECT sha256 FROM documents WHERE page_id=%s;", (page_id,))
    row = cur.fetchone()
    return row[0] if row else None

def upsert_chunks(cur, doc_id, chunks, embs):
    # wipe old chunks and insert fresh (simple & safe)
    cur.execute("DELETE FROM document_chunks WHERE doc_id=%s;", (doc_id,))
    rows = []
    for idx, (ch, emb) in enumerate(zip(chunks, embs)):
        rows.append((doc_id, idx, ch, emb.tolist(), json.dumps({})))
    execute_values(
        cur,
        "INSERT INTO document_chunks (doc_id, chunk_index, text, embedding, metadata) VALUES %s",
        rows,
        template="(%s,%s,%s,%s::vector,%s)"
    )

def index_once():
    conn = connect()
    conn.autocommit = False
    try:
        cur = conn.cursor()
        files = sorted([p for p in HTML_DIR.glob("*.html") if not p.name.startswith("TEST_")])
        for f in tqdm(files, desc="Indexing"):
            html = f.read_text(encoding="utf-8")
            text = html_to_text(html)
            sha = sha256_text(text)
            sidecar = load_sidecar_for(f)

            if sidecar:
                space_key = sidecar["space_key"]
                page_id   = sidecar["page_id"]
                title     = sidecar["title"]
                url       = sidecar["url"]
                last_modified = datetime.fromisoformat(sidecar["last_modified"].replace("Z","+00:00"))
                assets    = sidecar.get("assets", [])
            else:
                # Fallback: derive from filename + file mtime
                base = f.stem
                parts = base.split("_", 2)
                space_key = parts[0] if len(parts) > 0 else "UNKNOWN"
                page_id   = parts[1] if len(parts) > 1 else base
                title     = parts[2] if len(parts) > 2 else base
                url       = "about:blank"
                last_modified = datetime.fromtimestamp(f.stat().st_mtime)
                assets    = []

            # Skip heavy work if sha unchanged
            old_sha = existing_sha_for(cur, page_id)
            if old_sha == sha:
                # still update metadata fields if they changed (url/last_modified/assets/html_path)
                cur.execute("""
                  UPDATE documents
                     SET title=%s, url=%s, last_modified=%s, html_path=%s, assets=%s::jsonb
                   WHERE page_id=%s;
                """, (title, url, last_modified, str(f), json.dumps(assets), page_id))
                conn.commit()
                continue

            # Upsert doc and get id
            doc_id, _ = upsert_document(cur, space_key, page_id, title, url, last_modified, str(f), sha, assets)

            # Chunk & embed
            chunks = chunk_text(text)
            embs = embed_texts(chunks)
            upsert_chunks(cur, doc_id, chunks, embs)
            conn.commit()

        cur.close()
    finally:
        conn.close()

if __name__ == "__main__":
    if "--watch" in sys.argv:
        while True:
            index_once()
            time.sleep(30)
    else:
        index_once()
