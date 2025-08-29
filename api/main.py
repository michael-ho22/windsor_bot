import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
import re
import hashlib, uuid, mimetypes
from io import BytesIO
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from fastapi import (
    Body, Depends, FastAPI, HTTPException, Query, Header, Request,
    UploadFile, File, Form
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from psycopg.types.json import Json

from openai import OpenAI, RateLimitError, APIConnectionError

# email
import smtplib, ssl
from email.message import EmailMessage

# ---- auth helpers (bcrypt + JWT) ----
import bcrypt
import jwt
from datetime import datetime, timedelta, timezone

# env bootstrap
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from env_bootstrap import load_env
ROOT = load_env(__file__)

# ===== Config =====
PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", 5432)),
    dbname=os.getenv("PGDATABASE", "windsor"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", "postgres"),
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4o")

AUTH_MODE = os.getenv("AUTH_MODE", "multi").lower()  # "multi" | "none"
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_EXPIRES_MIN = int(os.getenv("JWT_EXPIRES_MIN", "10080"))  # 7 days default
ALLOWED_EMAIL_DOMAIN = os.getenv("ALLOWED_EMAIL_DOMAIN", "windsorsolutions.com")

HTML_DIR = os.getenv("HTML_DIR", "./crawler/confluence_export")

AUTO_TITLE_MODE = os.getenv("AUTO_TITLE_MODE", "heuristic").lower()  # 'heuristic' | 'llm' | 'off'
AUTO_TITLE_MAX_WORDS = int(os.getenv("AUTO_TITLE_MAX_WORDS", "8"))
DEFAULT_SESSION_TITLE = os.getenv("DEFAULT_SESSION_TITLE", "New chat")

# NEW: uploads
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/data/uploads")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Windsor Knowledge API")

# CORS: local streamlit + docker streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve image assets (downloaded by crawler)
assets_path = os.path.join(HTML_DIR, "assets")
if os.path.isdir(assets_path):
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")


# ===== Utilities =====
@contextmanager
def q_conn():
    conn = psycopg.connect(**PGCFG)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def _retry(call, attempts: int = 5, base_sleep: float = 0.5):
    for i in range(attempts):
        try:
            return call(timeout=30)
        except (RateLimitError, APIConnectionError) as e:
            if i == attempts - 1:
                raise
            sleep = base_sleep * (2 ** i)
            print(f"[OpenAI] transient error ({type(e).__name__}); retry in {sleep:.2f}s")
            time.sleep(sleep)


def embed_once(text: str) -> List[float]:
    resp = _retry(lambda **kw: client.embeddings.create(model=EMBED_MODEL, input=text, **kw))
    return resp.data[0].embedding


# ===== Simple in-memory rate limiting (per-process) =====
from time import time as _now
_RATE: Dict[str, List[float]] = {}  # key -> timestamps (seconds)

def rate_limit(key: str, max_calls: int, per_seconds: int):
    now = _now()
    arr = _RATE.setdefault(key, [])
    cutoff = now - per_seconds
    # drop old entries
    while arr and arr[0] < cutoff:
        arr.pop(0)
    if len(arr) >= max_calls:
        raise HTTPException(status_code=429, detail="Too many requests")
    arr.append(now)


# ===== Email helper (no-ops if SMTP env not set) =====
def send_welcome_email(to_email: str):
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "0") or "0")
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    sender = os.getenv("SMTP_FROM")

    if not all([host, port, user, pwd, sender]):
        return  # SMTP not configured; skip silently

    msg = EmailMessage()
    msg["Subject"] = "Welcome to Windsor Knowledge Bot"
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(
        "You're all set!\n\n"
        "Your Windsor Knowledge Bot account has been created. "
        "You can log in with your Windsor email in the app sidebar.\n\n"
        "— Windsor Solutions"
    )

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.starttls(context=context)
        server.login(user, pwd)
        server.send_message(msg)


# ===== Auth primitives =====
def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(pw: str, h: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8"), h.encode("utf-8"))
    except Exception:
        return False


def create_jwt(uid: int, email: str, ver: int) -> str:
    payload = {
        "uid": uid,
        "email": email,
        "ver": ver,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRES_MIN),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_jwt(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])


def get_current_user_id(authorization: Optional[str] = Header(None)) -> Optional[int]:
    if AUTH_MODE == "none":
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = decode_jwt(token)
        uid = int(payload["uid"])
        ver = int(payload.get("ver", 1))
        # check token_version in DB
        with q_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT token_version FROM users WHERE id=%s AND is_active", (uid,))
            row = cur.fetchone()
            if not row or int(row["token_version"]) != ver:
                raise HTTPException(status_code=401, detail="Token invalidated")
        return uid
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid/expired token")


# ===== Health =====
@app.get("/health")
def health():
    try:
        with q_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ===== Retrieval (with user/session scope) =====
def hybrid_search_sql(
    conn,
    query: str,
    v: List[float],
    k: int,
    alpha: float,
    space_keys: Optional[List[str]],
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
):
    space_filter_vec = "AND d.space_key = ANY(%s)" if space_keys else ""
    space_filter_bm  = "AND d.space_key = ANY(%s)" if space_keys else ""
    # Scope: global docs OR user library OR this-session docs
    scope_filter = ""
    scope_params: List[Any] = []
    if user_id is not None:
        scope_filter = """
          AND (
            d.user_id IS NULL
            OR (d.user_id = %s AND d.visibility = 'library')
            OR (d.user_id = %s AND d.session_id = %s)
          )
        """
        scope_params = [user_id, user_id, session_id]

    k_each = max(20, k)

    sql = f"""
    WITH vec AS (
      SELECT c.id AS chunk_id, d.title, d.url, d.space_key, d.assets, d.page_id, c.text,
             1 - (c.embedding <=> %s::vector) AS vscore
      FROM document_chunks c
      JOIN documents d ON d.id = c.doc_id
      WHERE 1=1 {space_filter_vec} {scope_filter}
      ORDER BY c.embedding <=> %s::vector
      LIMIT %s
    ),
    bm25 AS (
      SELECT c.id AS chunk_id, d.title, d.url, d.space_key, d.assets, d.page_id, c.text,
             ts_rank(c.tsv, plainto_tsquery('english', %s)) AS tscore
      FROM document_chunks c
      JOIN documents d ON d.id = c.doc_id
      WHERE c.tsv @@ plainto_tsquery('english', %s) {space_filter_bm} {scope_filter}
      ORDER BY tscore DESC
      LIMIT %s
    ),
    unioned AS (
      SELECT COALESCE(v.chunk_id, b.chunk_id) AS chunk_id,
             COALESCE(v.title, b.title) AS title,
             COALESCE(v.url, b.url) AS url,
             COALESCE(v.space_key, b.space_key) AS space_key,
             COALESCE(v.assets, b.assets) AS assets,
             COALESCE(v.page_id, b.page_id) AS page_id,
             COALESCE(v.text, b.text) AS text,
             COALESCE(v.vscore, 0) AS vscore,
             COALESCE(b.tscore, 0) AS tscore
      FROM vec v
      FULL OUTER JOIN bm25 b ON v.chunk_id = b.chunk_id
    )
    SELECT chunk_id, title, url, space_key, assets, page_id, text,
           (%s * vscore + (1 - %s) * tscore) AS hybrid_score
    FROM unioned
    ORDER BY hybrid_score DESC
    LIMIT %s;
    """

    params: List[object] = []
    # vec
    params.append(v)
    if space_keys: params.append(space_keys)
    params.extend(scope_params)
    params.append(v)
    params.append(k_each)
    # bm25
    params.append(query)
    params.append(query)
    if space_keys: params.append(space_keys)
    params.extend(scope_params)
    params.append(k_each)
    # final
    params.append(alpha)
    params.append(alpha)
    params.append(k_each)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def mmr_rerank(items, lambda_mult=0.7, top_n=8):
    """Greedy MMR with a simple Jaccard word-set dissimilarity proxy."""
    if not items:
        return items
    for it in items:
        it["_wset"] = set((it.get("text") or "").lower().split())
    selected, remain = [], items[:]
    remain.sort(key=lambda x: x["score"], reverse=True)
    if remain:
        selected.append(remain.pop(0))

    def dissim(a, b):
        A, B = a["_wset"], b["_wset"]
        if not A or not B:
            return 1.0
        inter = len(A & B)
        union = len(A | B)
        return 1.0 - (inter / union if union else 0.0)

    while remain and len(selected) < top_n:
        best, best_val = None, -1e9
        for c in remain:
            max_div = 0.0
            for s in selected:
                max_div = max(max_div, dissim(c, s))
            val = lambda_mult * c["score"] + (1 - lambda_mult) * max_div
            if val > best_val:
                best_val, best = val, c
        selected.append(best)
        remain.remove(best)

    for it in items:
        it.pop("_wset", None)
    return selected


# ===== Search (optional user scope) =====
@app.get("/search")
def search(
    q: str = Query(...),
    k: int = Query(8, ge=1, le=50),
    spaces: Optional[str] = Query(None),
    per_doc: int = Query(2, ge=1, le=10),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    alpha: float = Query(0.6, ge=0.0, le=1.0),
    session_id: Optional[int] = Query(None),
    user_id: Optional[int] = Depends(get_current_user_id) if AUTH_MODE != "none" else None,
):
    v = embed_once(q)
    space_keys = [s.strip() for s in spaces.split(",") if s.strip()] if spaces else None
    with q_conn() as conn:
        rows = hybrid_search_sql(conn, q, v, k * 3, alpha, space_keys, user_id=user_id, session_id=session_id)

    by_doc, hits = {}, []
    for r in rows:
        score = float(r["hybrid_score"])
        if score < min_score:
            continue
        pid = r.get("page_id") or r.get("title")
        if by_doc.get(pid, 0) >= per_doc:
            continue
        by_doc[pid] = by_doc.get(pid, 0) + 1
        hits.append({
            "chunk_id": r["chunk_id"],
            "title": r["title"],
            "url": r["url"],
            "space_key": r["space_key"],
            "assets": r["assets"],
            "page_id": r["page_id"],
            "text": r["text"],
            "score": score,
        })
        if len(hits) >= k:
            break

    hits = mmr_rerank(hits, lambda_mult=0.7, top_n=k)
    return {"results": hits, "total_considered": len(rows)}


# ===== Sessions & Messages helpers =====
def ensure_session(conn, session_id: Optional[int], title: Optional[str], user_id: Optional[int]) -> int:
    with conn.cursor(row_factory=dict_row) as cur:
        if session_id:
            cur.execute("SELECT id, user_id FROM chat_sessions WHERE id=%s", (session_id,))
            row = cur.fetchone()
            if row:
                if AUTH_MODE != "none" and row["user_id"] != user_id:
                    raise HTTPException(status_code=404, detail="Session not found")
                return row["id"]
        cur.execute(
            "INSERT INTO chat_sessions(title, user_id) VALUES(%s, %s) RETURNING id",
            (title or DEFAULT_SESSION_TITLE, user_id)
        )
        return cur.fetchone()["id"]


def get_history(conn, session_id: int, limit: int = 12):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT role, content
            FROM chat_messages
            WHERE session_id=%s
            ORDER BY created_at DESC
            LIMIT %s
        """, (session_id, limit))
        rows = cur.fetchall()
    return list(reversed(rows))


def add_message(conn, session_id: int, role: str, content: str, citations: Optional[list] = None):
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO chat_messages(session_id, role, content, citations)
          VALUES (%s, %s, %s, %s)
        """, (session_id, role, content, Json(citations or [])))
        conn.commit()


def get_user_style(conn, user_id: Optional[int]) -> tuple[str, str]:
    tone, depth = "neutral", "balanced"
    if user_id is None:
        return tone, depth
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT tone, depth FROM user_settings WHERE user_id=%s", (user_id,))
        r = cur.fetchone()
        if r:
            tone, depth = r["tone"], r["depth"]
    return tone, depth


# ----- Auto title helpers -----
_TRIVIAL_OPENERS = {
    "hi","hello","hey","yo","sup","good morning","good afternoon","good evening",
    "how are you","what's up","whats up"
}

def _first_substantial_sentence(text: str) -> Optional[str]:
    if not text: return None
    t = re.sub(r"\s+", " ", text).strip()
    if not t: return None
    tl = t.lower()
    for opener in _TRIVIAL_OPENERS:
        if tl.startswith(opener):
            t = t[len(opener):].lstrip(" ,.-:;!?")
            break
    for part in re.split(r"[\.!?;\n]+", t):
        part = part.strip()
        if len(part.split()) >= 3:
            return part
    return t if len(t.split()) >= 3 else None

def _trim_words(s: str, max_words: int) -> str:
    words = s.split()
    return s if len(words) <= max_words else " ".join(words[:max_words])

def _titlecase(s: str) -> str:
    small = {"a","an","and","or","the","for","to","of","in","on","at","with","by","from"}
    out = []
    for i,w in enumerate(s.split()):
        ww = w.lower()
        out.append(ww if (i!=0 and ww in small) else ww[:1].upper()+ww[1:])
    return " ".join(out)

def _gen_title_heuristic(user_text: str, max_words: int) -> Optional[str]:
    sent = _first_substantial_sentence(user_text or "")
    if not sent: return None
    sent = re.sub(r"^(how do i|how to|can you|could you|please|why is|what is|whats|what's)\b[\s:,-]*",
                  "", sent, flags=re.IGNORECASE).strip()
    sent = _trim_words(sent, max_words)
    sent = re.sub(r'["“”\'`]+', "", sent).strip(" .,-;:!?")
    return _titlecase(sent) if len(sent) >= 3 else None

def _gen_title_llm(user_text: str, max_words: int) -> Optional[str]:
    try:
        prompt = (
            f"Make a short, specific chat title (<= {max_words} words) for this first message. "
            "No ending punctuation. No quotes.\n\nMessage:\n" + (user_text or "").strip()
        )
        resp = _retry(lambda **kw: client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role":"system","content":"You write ultra-brief, specific titles."},
                      {"role":"user","content":prompt}],
            temperature=0.2, max_tokens=16, **kw
        ))
        title = (resp.choices[0].message.content or "").strip()
        title = re.sub(r'[\"“”\'`]+', "", title).strip(" .,-;:!?")
        return _titlecase(_trim_words(title, max_words)) if title else None
    except Exception:
        return None

def generate_chat_title(user_text: str) -> Optional[str]:
    if AUTO_TITLE_MODE == "off":
        return None
    t = _gen_title_heuristic(user_text, AUTO_TITLE_MAX_WORDS)
    if t: return t
    return _gen_title_llm(user_text, AUTO_TITLE_MAX_WORDS) if AUTO_TITLE_MODE == "llm" else None


# ===== Chat (RAG) =====
@app.post("/answer")
def answer(
    payload: Dict[str, Any] = Body(...),
    request: Request = None,
    user_id: Optional[int] = Depends(get_current_user_id) if AUTH_MODE != "none" else None,
):
    # Rate limit per user (or per IP if unauth)
    rl_key = f"ans:uid:{user_id}" if user_id is not None else f"ans:ip:{request.client.host}"
    rate_limit(rl_key, max_calls=60, per_seconds=60)  # 60/min

    q = (payload.get("message") or "").strip()
    if not q:
        raise HTTPException(400, "message is required")

    k = int(payload.get("k", 8))
    alpha = float(payload.get("alpha", 0.6))
    per_doc = int(payload.get("per_doc", 2))
    min_score = float(payload.get("min_score", 0.0))
    spaces = payload.get("spaces")
    space_keys = [s.strip() for s in spaces.split(",")] if spaces else None
    memory_note = payload.get("memory_note")
    title = payload.get("title")

    with q_conn() as conn:
        # session
        session_id = ensure_session(conn, payload.get("session_id"), title, user_id)

        # optional memory
        if memory_note:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_memory(session_id, user_id, key, value)
                    VALUES (%s, %s, %s, %s)
                """, (session_id, user_id, "note", memory_note))
                conn.commit()

        # store user message
        add_message(conn, session_id, "user", q)

        # auto-title (only if current title is default/blank and client didn't set one)
        if not title:
            try:
                with conn.cursor(row_factory=dict_row) as cur2:
                    cur2.execute("SELECT title FROM chat_sessions WHERE id=%s", (session_id,))
                    row2 = cur2.fetchone()
                    current_title = (row2["title"] or "").strip() if row2 else ""
                    if (not current_title) or (current_title.lower() == DEFAULT_SESSION_TITLE.lower()):
                        new_title = generate_chat_title(q)
                        if new_title:
                            cur2.execute("UPDATE chat_sessions SET title=%s WHERE id=%s", (new_title, session_id))
                            conn.commit()
            except Exception:
                pass

        # history
        hist = get_history(conn, session_id, limit=12)

        # retrieval (scope to user/library + this session)
        v = embed_once(q)
        rows = hybrid_search_sql(conn, q, v, k * 3, alpha, space_keys, user_id=user_id, session_id=session_id)

        by_doc, hits = {}, []
        for r in rows:
            score = float(r["hybrid_score"])
            if score < min_score:
                continue
            pid = r.get("page_id") or r.get("title")
            if by_doc.get(pid, 0) >= per_doc:
                continue
            by_doc[pid] = by_doc.get(pid, 0) + 1
            hits.append({
                "chunk_id": r["chunk_id"],
                "title": r["title"],
                "url": r["url"],
                "space_key": r["space_key"],
                "assets": r["assets"],
                "page_id": r["page_id"],
                "text": r["text"],
                "score": score,
            })
            if len(hits) >= k:
                break

        hits = mmr_rerank(hits, lambda_mult=0.7, top_n=k)

        # style
        tone, depth = get_user_style(conn, user_id)

    def mk_context():
        if not hits:
            return "(no context)"
        lines = []
        for h in hits:
            title = h["title"] or h["url"] or f"doc:{h.get('page_id')}"
            cid = h.get("chunk_id")
            lines.append(f"[{title}#{cid}] {h['text']}")
        return "\n".join(lines)

    SYSTEM = (
        "You are Windsor Solutions' internal assistant.\n"
        f"- Tone: {tone}. Depth: {depth}.\n"
        "- Answer primarily from the provided CONTEXT.\n"
        "- Add bracket citations like [Title#chunk_id] right after factual sentences.\n"
        "- If unknown, say you don't know and suggest where to look."
    )

    msgs = [{"role": "system", "content": SYSTEM}]
    for m in hist[-10:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": f"Question: {q}\n\nCONTEXT:\n{mk_context()}"})

    resp = _retry(lambda **kw: client.chat.completions.create(
        model=CHAT_MODEL, messages=msgs, temperature=0.1, **kw
    ))
    answer_text = resp.choices[0].message.content

    # persist assistant message (with sources)
    with q_conn() as conn2:
        add_message(conn2, session_id, "assistant", answer_text, citations=hits)

    return {"session_id": session_id, "answer": answer_text, "sources": hits}


# ===== Sessions API (user-scoped when auth is on) =====
@app.get("/sessions")
def list_sessions(
    limit: int = Query(20, ge=1, le=200),
    user_id: Optional[int] = Depends(get_current_user_id) if AUTH_MODE != "none" else None,
):
    with q_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        if AUTH_MODE == "none":
            cur.execute("""
              SELECT id, title, created_at FROM chat_sessions
              ORDER BY created_at DESC
              LIMIT %s
            """, (limit,))
        else:
            cur.execute("""
              SELECT id, title, created_at FROM chat_sessions
              WHERE user_id = %s
              ORDER BY created_at DESC
              LIMIT %s
            """, (user_id, limit))
        return {"sessions": cur.fetchall()}


@app.get("/sessions/{session_id}")
def get_session_messages(
    session_id: int,
    limit: int = Query(100, ge=1, le=500),
    user_id: Optional[int] = Depends(get_current_user_id) if AUTH_MODE != "none" else None,
):
    with q_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        if AUTH_MODE == "none":
            cur.execute("""
              SELECT role, content, created_at
              FROM chat_messages
              WHERE session_id=%s
              ORDER BY created_at ASC
              LIMIT %s
            """, (session_id, limit))
        else:
            cur.execute("SELECT user_id FROM chat_sessions WHERE id=%s", (session_id,))
            row = cur.fetchone()
            if not row or row["user_id"] != user_id:
                raise HTTPException(404, "Session not found")
            cur.execute("""
              SELECT role, content, created_at
              FROM chat_messages
              WHERE session_id=%s
              ORDER BY created_at ASC
              LIMIT %s
            """, (session_id, limit))
        return {"messages": cur.fetchall()}


# ===== Auth endpoints =====
@app.post("/signup")
def signup(body: Dict[str, str] = Body(...), request: Request = None):
    if AUTH_MODE == "none":
        raise HTTPException(400, "Auth disabled")

    rate_limit(f"signup:{request.client.host}", max_calls=10, per_seconds=300)  # 10 / 5min

    email = (body.get("email") or "").strip().lower()
    pw = body.get("password") or ""
    if not email or not pw:
        raise HTTPException(400, "email and password are required")

    if ALLOWED_EMAIL_DOMAIN and not email.endswith("@" + ALLOWED_EMAIL_DOMAIN):
        raise HTTPException(403, f"Email must be @{ALLOWED_EMAIL_DOMAIN}")

    with q_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id FROM users WHERE email=%s", (email,))
        if cur.fetchone():
            raise HTTPException(409, "Email already registered")
        ph = hash_password(pw)
        cur.execute(
            "INSERT INTO users(email, password_hash) VALUES(%s,%s) RETURNING id",
            (email, ph)
        )
        uid = cur.fetchone()["id"]
        conn.commit()

    with q_conn() as conn2, conn2.cursor(row_factory=dict_row) as cur2:
        cur2.execute("SELECT token_version FROM users WHERE id=%s", (uid,))
        ver = int(cur2.fetchone()["token_version"])

    token = create_jwt(uid, email, ver)

    try:
        send_welcome_email(email)
    except Exception:
        pass

    return {"ok": True, "token": token, "user": {"id": uid, "email": email}}


@app.post("/login")
def login(body: Dict[str, str] = Body(...), request: Request = None):
    if AUTH_MODE == "none":
        raise HTTPException(400, "Auth disabled")

    rate_limit(f"login:{request.client.host}", max_calls=20, per_seconds=60)  # 20 / min

    email = (body.get("email") or "").strip().lower()
    pw = body.get("password") or ""
    if not email or not pw:
        raise HTTPException(400, "email and password are required")

    with q_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, password_hash, token_version FROM users WHERE email=%s AND is_active", (email,))
        row = cur.fetchone()
        if not row or not verify_password(pw, row["password_hash"]):
            raise HTTPException(401, "Invalid credentials")
        uid = int(row["id"])
        ver = int(row["token_version"])

    token = create_jwt(uid, email, ver)
    return {"ok": True, "token": token, "user": {"id": uid, "email": email}}


@app.get("/me")
def me(user_id: Optional[int] = Depends(get_current_user_id) if AUTH_MODE != "none" else None):
    if AUTH_MODE == "none":
        return {"user": None}
    with q_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, email, is_active, created_at FROM users WHERE id=%s", (user_id,))
        row = cur.fetchone()
        return {"user": row}


# ===== User settings (tone/depth) =====
@app.post("/settings")
def update_settings(
    payload: Dict[str, str] = Body(...),
    user_id: Optional[int] = Depends(get_current_user_id) if AUTH_MODE != "none" else None,
):
    if AUTH_MODE != "none" and user_id is None:
        raise HTTPException(401, "Not authenticated")

    tone  = (payload.get("tone") or "neutral").lower()
    depth = (payload.get("depth") or "balanced").lower()
    with q_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO user_settings(user_id, tone, depth)
            VALUES (%s,%s,%s)
            ON CONFLICT (user_id) DO UPDATE
              SET tone=EXCLUDED.tone, depth=EXCLUDED.depth
        """, (user_id, tone, depth))
        conn.commit()
    return {"ok": True, "tone": tone, "depth": depth}


# ===== File uploads =====
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _extract_text_bytes(filename: str, content: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith((".txt", ".md", ".csv", ".log")):
        return content.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            pdf = PdfReader(BytesIO(content))
            return "\n".join([(p.extract_text() or "") for p in pdf.pages])
        except Exception:
            return ""
    return ""  # other types: store but no text extracted

@app.post("/files/upload")
async def files_upload(
    file: UploadFile = File(...),
    session_id: Optional[int] = Form(None),
    to_library: bool = Form(False),
    user_id: Optional[int] = Depends(get_current_user_id) if AUTH_MODE != "none" else None,
):
    if AUTH_MODE != "none" and not user_id:
        raise HTTPException(401, "Not authenticated")

    data = await file.read()
    sha  = _sha256(data)
    folder = os.path.join(UPLOAD_DIR, str(user_id or 0))
    _ensure_dir(folder)
    ext = Path(file.filename).suffix or ""
    disk_name = f"{uuid.uuid4().hex}{ext}"
    disk_path = os.path.join(folder, disk_name)
    with open(disk_path, "wb") as f:
        f.write(data)

    mime = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"

    with q_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO user_files(user_id, filename, mime_type, size_bytes, sha256, disk_path, visibility)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (user_id, file.filename, mime, len(data), sha, disk_path, "library" if to_library else "private"),
        )
        ufid = cur.fetchone()["id"]
        conn.commit()

    # Link to chat if provided and not library
    if session_id and not to_library:
        with q_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_files(session_id, file_id) VALUES(%s,%s) ON CONFLICT DO NOTHING",
                (session_id, ufid),
            )
            conn.commit()

    # Index text (if any)
    text = _extract_text_bytes(file.filename, data)
    if text.strip():
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        with q_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO documents(title, url, space_key, assets, page_id, user_id, session_id, visibility)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (file.filename, None, "upload", None, ufid,
                 user_id, None if to_library else session_id,
                 "library" if to_library else "private"),
            )
            doc_id = cur.fetchone()["id"]
            for ch in chunks:
                emb = embed_once(ch)
                cur.execute(
                    """
                    INSERT INTO document_chunks(doc_id, text, embedding, tsv)
                    VALUES (%s, %s, %s, to_tsvector('english', %s))
                    """,
                    (doc_id, ch, emb, ch)
                )
            conn.commit()

    return {"ok": True, "file_id": ufid}

@app.get("/files/list")
def files_list(
    session_id: Optional[int] = Query(None),
    user_id: Optional[int] = Depends(get_current_user_id) if AUTH_MODE != "none" else None,
):
    if AUTH_MODE != "none" and not user_id:
        raise HTTPException(401, "Not authenticated")
    with q_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        if session_id:
            cur.execute(
                """
                SELECT uf.id, uf.filename, uf.mime_type, uf.size_bytes, uf.visibility
                FROM user_files uf
                LEFT JOIN chat_files cf ON cf.file_id = uf.id AND cf.session_id=%s
                WHERE uf.user_id=%s AND (uf.visibility='library' OR cf.session_id IS NOT NULL)
                ORDER BY uf.created_at DESC
                """,
                (session_id, user_id)
            )
        else:
            cur.execute(
                """
                SELECT id, filename, mime_type, size_bytes, visibility
                FROM user_files
                WHERE user_id=%s
                ORDER BY created_at DESC
                """,
                (user_id,)
            )
        return {"files": cur.fetchall()}

@app.delete("/files/{file_id}")
def files_delete(
    file_id: int,
    session_id: Optional[int] = Query(None),
    user_id: Optional[int] = Depends(get_current_user_id) if AUTH_MODE != "none" else None,
):
    if AUTH_MODE != "none" and not user_id:
        raise HTTPException(401, "Not authenticated")

    with q_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT user_id, disk_path FROM user_files WHERE id=%s", (file_id,))
        row = cur.fetchone()
        if not row or (row["user_id"] != user_id):
            raise HTTPException(404, "File not found")

        if session_id:
            cur.execute("DELETE FROM chat_files WHERE session_id=%s AND file_id=%s", (session_id, file_id))
        else:
            cur.execute("DELETE FROM chat_files WHERE file_id=%s", (file_id,))
            cur.execute("DELETE FROM user_files WHERE id=%s", (file_id,))
            try:
                os.remove(row["disk_path"])
            except Exception:
                pass
        conn.commit()

    return {"ok": True}
