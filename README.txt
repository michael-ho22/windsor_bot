# Windsor Knowledge Bot — README

A self-hosted Confluence-RAG chatbot.  
This stack includes:

- **crawler/** — pulls Confluence pages + assets to HTML
- **indexer/** — chunks pages, creates embeddings, stores in Postgres/pgvector
- **api/** — FastAPI server that does hybrid search + chat
- **app/** — Streamlit UI
- **db** — Postgres with pgvector (via Docker)

Everything runs with **Docker Compose**. A PowerShell helper script (`start.ps1`) brings it all up, waits for health, and sanity-checks the DB.

---

## 1) Prerequisites

- **Windows 10/11** with **Docker Desktop** (WSL2 backend enabled), or macOS/Linux with Docker
- ~4 GB free RAM for containers
- **Confluence** read-only service account:
  - `ATLASSIAN_URL` (e.g., `https://yourcompany.atlassian.net/wiki`)
  - `ATLASSIAN_EMAIL`
  - `ATLASSIAN_API_TOKEN`
- **OpenAI API Key** (`OPENAI_API_KEY`)

> Tip (Windows): Run PowerShell as Administrator the first time Docker starts.

---

## 2) What gets created at first run?

- A Postgres database named **`windsor`** with all tables & indexes (via `schemas.sql` + a bootstrap step)
- A local HTML export folder used by the crawler (images rewritten to local `crawler/confluence_export/assets/...`)
- Vector embeddings for document chunks (indexed by pgvector)

---

## 3) Configuration

There are two environment files:

- **`.env` (root)** — used when running Python locally (outside Docker).  
  Good for development.
- **`.env.docker` (root)** — used **inside** the containers.  
  **This is the one the stack uses by default.**

Minimum keys you must set in **`.env.docker`**:

```
# --- Postgres / pgvector ---
PGHOST=db
PGPORT=5432
PGDATABASE=windsor
PGUSER=postgres
PGPASSWORD=postgres

# --- OpenAI ---
OPENAI_API_KEY=sk-...

# --- Models (defaults shown) ---
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o

# --- Confluence ---
ATLASSIAN_URL=https://yourcompany.atlassian.net/wiki
ATLASSIAN_EMAIL=bot@yourcompany.com
ATLASSIAN_API_TOKEN=xxxxxxxx
ATLASSIAN_SPACES=WindsorSupport,NPKB

# --- Paths (in-container) ---
HTML_DIR=/workspace/crawler/confluence_export
```

> **Important:** `PGHOST=db` is correct for Docker (the DB service’s name).  
> Don’t set `PGHOST=localhost` in `.env.docker`.

---

## 4) Quick Start (Windows PowerShell)

**Two-liner for first run:**
```
# from repo root
.\start.ps1
# then open: http://127.0.0.1:8501
```

**Manual alternative (if you don’t want to use start.ps1):**
```
docker compose up -d
# API docs:  http://127.0.0.1:8001/docs
# App UI:    http://127.0.0.1:8501
```

The crawler runs immediately, then every ~48 hours. The indexer runs continuously in “watch” mode and re-embeds changed pages.

---

## 5) What to expect on first boot

1. **DB** starts and runs migrations from `schemas.sql`.
2. **API** builds and starts at `http://127.0.0.1:8001`.
3. **Crawler** downloads Confluence pages and assets into `crawler/confluence_export/`.
4. **Indexer** detects new/updated HTML files, chunks them, calls embeddings, and inserts rows into:
   - `documents`
   - `document_chunks`
5. **App** is available at `http://127.0.0.1:8501`.

The first run may take a bit while pages are crawled and embedded.

---

## 6) Sanity checks

**Check container status:**
```
docker compose ps
```

**API health:**
```
Invoke-WebRequest http://127.0.0.1:8001/health | Select-Object -Expand Content
# expect: {"ok":true}
```

**Tail logs (helpful during first run):**
```
docker logs -f windsor_crawler
docker logs -f windsor_indexer
docker logs -f windsor_api
```

**Confirm DB has data:**
```
docker exec -i windsor_db psql -U postgres -d windsor -c "SELECT count(*) AS documents FROM documents; SELECT count(*) AS chunks FROM document_chunks;"
```

**Open the UI:**
- App: `http://127.0.0.1:8501`
- API docs: `http://127.0.0.1:8001/docs`

---

## 7) Restart / Rebuild / Fresh start

Using **start.ps1** (recommended):

```
# normal start (brings up, waits for health, sanity-checks, opens the app)
.\start.ps1

# rebuild containers (after changing Dockerfile/compose/requirements)
.\start.ps1 -Rebuild

# fresh DB (drops volumes, re-seeds)
.\start.ps1 -Fresh

# don't auto-open the browser
.\start.ps1 -NoOpen
```

Manual equivalents:

```
docker compose up -d --build     # rebuild
docker compose down -v           # stop and wipe DB volume (fresh)
docker compose up -d             # start again
```

---

## 8) Ports & URLs

- **API:** `http://127.0.0.1:8001` (inside Docker network it’s `http://api:8000`)
- **App:** `http://127.0.0.1:8501`
- **Postgres:** `localhost:5432` (for tools like DBeaver; container name is `windsor_db`)

**DBeaver connection (host machine):**
- Host: `localhost`
- Port: `5432`
- DB: `windsor`
- User/Password: from `.env.docker` (defaults: `postgres` / `postgres`)

---

## 9) How updates happen

- **Requirements**: Each service runs `pip install -r requirements.txt` on start.
- **Crawler**: Runs once at boot, then every **48h** on a loop (configurable in compose).
- **Indexer**: Runs continually (`--watch`) and re-embeds changed HTML pages.
- **API & App**: Read your `.env.docker` in-container; they reload only on container restart.

If you change dependencies or environment values, do:
```
.\start.ps1 -Rebuild
```

---

## 10) Common issues & fixes

- **API container keeps restarting**
  - Check logs: `docker logs --tail=200 windsor_api`
  - Most common causes:
    - Missing `.env.docker` values (e.g., `OPENAI_API_KEY`)
    - DB not healthy yet (start script waits, but check `docker compose ps`)
- **No documents in UI**
  - Watch crawler/indexer logs
  - Verify Confluence creds and `ATLASSIAN_SPACES` are correct
- **DB shows tables but no rows in your local DB client**
  - Connect to the right DB (`windsor`) on port `5432`
  - Ensure you’re not pointing your client at a different Postgres instance

---

## 11) File/Folder structure (simplified)

```
windsor_bot/
  app/
    app.py
    requirements.txt
  api/
    main.py
    bootstrap.py
    requirements.txt
  crawler/
    windsorwiki_crawler.py
    confluence_export/        # HTML + assets land here
    requirements.txt
  indexer/
    indexer.py
    requirements.txt
  schemas.sql                 # DB schema & indexes (auto-run on DB init)
  docker-compose.yml
  start.ps1
  .env                        # for running local Python outside Docker
  .env.docker                 # used by containers (primary config)
  README.txt
```

---

## 12) Security notes

- Keep `.env` / `.env.docker` out of version control or use a secrets manager.
- The Confluence account should be **read-only** and scoped to the needed spaces.

---

## 13) Support / Useful one-liners

- Show services:
  ```
  docker compose ps
  ```
- Tail API logs:
  ```
  docker logs -f windsor_api
  ```
- Tail indexer logs:
  ```
  docker logs -f windsor_indexer
  ```
- In-network API check (from inside API container):
  ```
  docker exec -it windsor_api curl -s http://api:8000/health
  ```
- DB counts:
  ```
  docker exec -i windsor_db psql -U postgres -d windsor -c "SELECT count(*) documents FROM documents; SELECT count(*) chunks FROM document_chunks;"
  ```

---

Happy searching!
