#!/usr/bin/env python3
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from openai import OpenAI

# -------- env & config --------
load_dotenv()

PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "windsor"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", "postgres"),
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL      = os.getenv("CHAT_MODEL",  "gpt-4o-mini")

TOP_K           = int(os.getenv("TOP_K", "6"))
MIN_SCORE       = float(os.getenv("MIN_SCORE", "0.20"))  # 0..1 (cosine similarity)
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

client = OpenAI(api_key=OPENAI_API_KEY)

# -------- helpers --------
def connect():
    return psycopg2.connect(**PGCFG)

def embed(text: str) -> List[float]:
    # basic retry/backoff on rate limits
    delay = 1.0
    for _ in range(5):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
            return resp.data[0].embedding
        except Exception as e:
            msg = str(e)
            if "rate_limit" in msg or "Rate limit" in msg:
                time.sleep(delay)
                delay = min(delay * 2, 8)
                continue
            raise
    raise RuntimeError("Embedding failed after retries")

def search(query_vec: List[float], space_filter: List[str] | None = None) -> List[dict]:
    """
    Uses pgvector cosine distance '<->' (lower = closer).
    We also return a normalized similarity score: 1 - distance.
    """
    sql = """
    SELECT
      d.page_id,
      d.title,
      d.url,
      d.space_key,
      dc.chunk_index,
      dc.text,
      (1 - (dc.embedding <-> %s::vector)) AS score
    FROM document_chunks dc
    JOIN documents d ON d.id = dc.doc_id
    {where}
    ORDER BY dc.embedding <-> %s::vector
    LIMIT %s;
    """
    params = [query_vec]
    where = ""
    if space_filter:
        where = "WHERE d.space_key = ANY(%s)"
        params.append(space_filter)
    # the distance expression appears twice (ORDER BY and score calc)
    params.append(query_vec)
    params.append(TOP_K)

    with connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql.format(where=where), params)
            rows = cur.fetchall()
    # filter by MIN_SCORE in python (optional, distance vs similarity is easy to adjust here)
    return [r for r in rows if float(r["score"]) >= MIN_SCORE]

def build_context(snippets: List[dict]) -> Tuple[str, List[dict]]:
    """
    Concatenate snippets until MAX_CONTEXT_CHARS is reached.
    Also dedupe by (page_id, chunk_index).
    """
    seen = set()
    pieces = []
    selected = []
    total = 0
    for r in snippets:
        key = (r["page_id"], r["chunk_index"])
        if key in seen:
            continue
        t = r["text"].strip()
        if not t:
            continue
        block = f"[{r['space_key']}:{r['page_id']} #{r['chunk_index']}] {r['title']}\n{t}\n"
        if total + len(block) > MAX_CONTEXT_CHARS and selected:
            break
        pieces.append(block)
        selected.append(r)
        total += len(block)
    return "\n---\n".join(pieces), selected

def answer(question: str, context: str) -> str:
    system = (
        "You are a helpful assistant answering questions about the Windsor wiki. "
        "Answer concisely, cite page titles when relevant, and say 'I’m not sure' if the answer "
        "is not in the provided context. Do not invent URLs."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Context (excerpts):\n{context}\n\n"
        "Give an accurate answer using only the context. If uncertain, say so."
    )
    # retry/backoff
    delay = 1.0
    for _ in range(5):
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role":"system", "content": system},
                    {"role":"user", "content": user},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if "rate_limit" in msg or "Rate limit" in msg:
                time.sleep(delay)
                delay = min(delay * 2, 8)
                continue
            raise

def pretty_sources(rows: List[dict]) -> str:
    lines = []
    for r in rows:
        lines.append(f"- {r['title']} (space {r['space_key']}), page {r['page_id']}, chunk #{r['chunk_index']}, score {r['score']:.3f}")
    return "\n".join(lines) if lines else "(no sources)"

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python query.py \"your question here\"")
        print("Optional env vars: TOP_K, MIN_SCORE, MAX_CONTEXT_CHARS, CHAT_MODEL, EMBED_MODEL")
        print("Space filter example: set QUERY_SPACES=WindsorSupport,NPKB")
        sys.exit(1)

    question = sys.argv[1]
    space_filter = None
    if os.getenv("QUERY_SPACES"):
        space_filter = [s.strip() for s in os.getenv("QUERY_SPACES").split(",") if s.strip()]

    print(f"\n🔎 Query: {question}")
    if space_filter:
        print(f"   (spaces: {', '.join(space_filter)})")

    qvec = embed(question)
    hits = search(qvec, space_filter)
    if not hits:
        print("\nNo matches above threshold. Try relaxing MIN_SCORE or broadening spaces.")
        sys.exit(0)

    context, used = build_context(hits)
    ans = answer(question, context)

    print("\n🧠 Answer:\n" + ans + "\n")
    print("📚 Sources:")
    print(pretty_sources(used))

if __name__ == "__main__":
    main()
