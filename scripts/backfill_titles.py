import os, re, psycopg
from psycopg.rows import dict_row

# --- PG config from your .env.docker (same as the API uses) ---
PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", 5432)),
    dbname=os.getenv("PGDATABASE", "windsor"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", "postgres"),
)

DEFAULT_TITLE = os.getenv("DEFAULT_SESSION_TITLE", "New chat")
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

def mini_title(text, max_words=8):
    t = re.sub(r"\s+"," ",(text or "")).strip()
    if not t:
        return None
    # strip greetings
    t = re.sub(r"^(hi|hello|hey|yo|sup|good (morning|afternoon|evening)|how are you|what('?|’)s up)\W*",
               "", t, flags=re.I)
    # strip boilerplate
    t = re.sub(r"^(how do i|how to|can you|could you|please|why is|what is|what('?|’)s)\b[\s:,-]*",
               "", t, flags=re.I)
    # first sentence-ish
    t = re.split(r"[\.!?;\n]+", t)[0].strip()
    words = t.split()
    if not words:
        return None
    t = " ".join(words[:max_words]).strip(" .,-;:!?\"“”'`")
    return t or None

def main():
    updated = 0
    checked = 0
    with psycopg.connect(**PGCFG) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
          SELECT cs.id
          FROM chat_sessions cs
          WHERE cs.title IS NULL OR cs.title = %s
          ORDER BY cs.id ASC
        """, (DEFAULT_TITLE,))
        session_ids = [r["id"] for r in cur.fetchall()]

        for sid in session_ids:
            checked += 1
            cur.execute("""
              SELECT content
              FROM chat_messages
              WHERE session_id=%s AND role='user'
              ORDER BY created_at ASC, id ASC
              LIMIT 1
            """, (sid,))
            row = cur.fetchone()
            if not row: 
                continue
            title = mini_title(row["content"])
            if not title:
                continue
            if DRY_RUN:
                print(f"[DRY] {sid} -> {title}")
            else:
                cur.execute("UPDATE chat_sessions SET title=%s WHERE id=%s", (title, sid))
                updated += 1
        if not DRY_RUN:
            conn.commit()

    print(f"Checked: {checked}, Updated: {updated}, Skipped: {checked - updated}")

if __name__ == "__main__":
    main()
