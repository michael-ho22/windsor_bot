import os, sys, time, json, hashlib, random
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from openai import OpenAI, RateLimitError
from tqdm import tqdm

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from env_bootstrap import load_env

# =======================
# ENV & CONFIG
# =======================

ROOT = load_env(__file__)

HTML_DIR = Path(os.getenv("HTML_DIR", "./data/html"))  # same as crawler's EXPORT_DIR
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", 5432)),
    dbname=os.getenv("PGDATABASE", "windsor"),
    user=os.getenv("PGUSER", "windsor"),
    password=os.getenv("PGPASSWORD", "password"),
)

# Embedding call behavior
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))  # embeddings per API call
MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "6"))       # exponential backoff tries
BASE_BACKOFF = float(os.getenv("EMBED_BASE_BACKOFF", "1.0")) # seconds
BACKOFF_JITTER = float(os.getenv("EMBED_BACKOFF_JITTER", "0.35"))  # add 0..jitter seconds

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- token/cost tracking ----
TOTAL_TOKENS = 0
TOTAL_CHUNKS = 0
EMBED_COST_PER_1K = 0.00002  # USD per 1K tokens for text-embedding-3-small


# =======================
# DB & UTIL FUNCS
# =======================
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
    # simple word-based chunker (token-ish)
    words = text.split()
    chunks, i = [], 0
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    while i < len(words):
        chunks.append(" ".join(words[i:i+max_tokens]))
        i += step if i + max_tokens < len(words) else len(words)
    return chunks or [text]

def _sleep_backoff(attempt: int, retry_after_header: str | None = None):
    # If server tells us exactly how long to wait, respect it
    if retry_after_header:
        try:
            wait = float(retry_after_header)
            if wait > 0:
                time.sleep(wait)
                return
        except Exception:
            pass
    # Otherwise exponential backoff with jitter
    wait = BASE_BACKOFF * (2 ** attempt)
    wait += random.random() * BACKOFF_JITTER
    time.sleep(wait)

def embed_texts(texts):
    """
    Generate embeddings with batching + robust rate-limit handling.
    Returns list[np.ndarray] aligned to `texts`.
    """
    global TOTAL_TOKENS, TOTAL_CHUNKS
    if not texts:
        return []

    out = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i+EMBED_BATCH_SIZE]

        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                # token usage is per-request across inputs
                if getattr(resp, "usage", None) and getattr(resp.usage, "total_tokens", None) is not None:
                    TOTAL_TOKENS += resp.usage.total_tokens
                TOTAL_CHUNKS += len(batch)

                # resp.data is in same order as input
                for d in resp.data:
                    out.append(np.array(d.embedding, dtype=np.float32))
                break  # success, leave retry loop

            except RateLimitError as e:
                # Try to honor Retry-After if present
                retry_after = None
                try:
                    retry_after = getattr(getattr(e, "response", None), "headers", {}).get("Retry-After")
                except Exception:
                    pass

                if attempt >= MAX_RETRIES:
                    raise
                wait_s = retry_after if retry_after else None
                print(f"[RATE-LIMIT] batch {i//EMBED_BATCH_SIZE+1}: retrying (attempt {attempt+1}/{MAX_RETRIES})...")
                _sleep_backoff(attempt, wait_s)
                attempt += 1

            except Exception as e:
                # Network flake or transient 5xx
                if attempt >= MAX_RETRIES:
                    raise
                print(f"[WARN] embedding batch failed: {e} (attempt {attempt+1}/{MAX_RETRIES})")
                _sleep_backoff(attempt)
                attempt += 1

    return out

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
    # nuke-and-pave for simplicity/safety
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


# =======================
# MAIN INDEX PASS
# =======================
def index_once():
    conn = connect()
    conn.autocommit = False

    new_docs = 0
    updated_docs = 0
    skipped_docs = 0

    try:
        cur = conn.cursor()
        files = sorted([p for p in HTML_DIR.glob("*.html") if not p.name.startswith("TEST_")])

        for f in tqdm(files, desc="Indexing"):
            # Sidecar metadata (from crawler) if present
            sidecar = load_sidecar_for(f)
            if sidecar:
                space_key = sidecar["space_key"]
                page_id   = sidecar["page_id"]
                title     = sidecar["title"]
                url       = sidecar["url"]
                last_modified = datetime.fromisoformat(sidecar["last_modified"].replace("Z","+00:00"))
                assets    = sidecar.get("assets", [])
            else:
                # fallback from filename + mtime
                base = f.stem
                parts = base.split("_", 2)
                space_key = parts[0] if len(parts) > 0 else "UNKNOWN"
                page_id   = parts[1] if len(parts) > 1 else base
                title     = parts[2] if len(parts) > 2 else base
                url       = "about:blank"
                last_modified = datetime.fromtimestamp(f.stat().st_mtime)
                assets    = []

            # compute content hash fast (read file once)
            html_raw = f.read_text(encoding="utf-8")
            file_sha = sha256_text(html_raw)
            old_sha = existing_sha_for(cur, page_id)

            if old_sha == file_sha:
                # keep metadata fresh, no re-embed
                cur.execute("""
                  UPDATE documents
                     SET title=%s, url=%s, last_modified=%s, html_path=%s, assets=%s::jsonb
                   WHERE page_id=%s;
                """, (title, url, last_modified, str(f), json.dumps(assets), page_id))
                conn.commit()
                skipped_docs += 1
                print(f"[SKIP] {page_id} unchanged.")
                continue

            # changed/new → parse, chunk, embed
            text = html_to_text(html_raw)
            chunks = chunk_text(text)

            # embed with rate-limit aware batching
            embs = embed_texts(chunks)

            doc_id, _ = upsert_document(cur, space_key, page_id, title, url, last_modified, str(f), file_sha, assets)
            upsert_chunks(cur, doc_id, chunks, embs)
            conn.commit()

            if old_sha is None:
                new_docs += 1
                print(f"[NEW] {page_id} indexed.")
            else:
                updated_docs += 1
                print(f"[UPDATE] {page_id} re-indexed.")

        cur.close()

        # summary
        print(f"\nIndexing complete: {new_docs} new, {updated_docs} updated, {skipped_docs} skipped.")

    finally:
        conn.close()


# =======================
# CLI ENTRY
# =======================
if __name__ == "__main__":
    start = time.time()
    if "--watch" in sys.argv:
        while True:
            index_once()
            elapsed = time.time() - start
            est_cost = (TOTAL_TOKENS / 1000.0) * EMBED_COST_PER_1K
            print(f"Elapsed: {elapsed:.2f}s | Chunks: {TOTAL_CHUNKS} | Tokens: {TOTAL_TOKENS:,} | Est cost: ${est_cost:.6f}")
            time.sleep(30)
    else:
        index_once()
        elapsed = time.time() - start
        est_cost = (TOTAL_TOKENS / 1000.0) * EMBED_COST_PER_1K
        print("\n--- Indexing complete ---")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Chunks embedded this run: {TOTAL_CHUNKS:,}")
        print(f"Total tokens used: {TOTAL_TOKENS:,}")
        print(f"Estimated embedding cost: ${est_cost:.6f}")
