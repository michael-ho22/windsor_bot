#!/usr/bin/env python3
"""
Windsor Confluence crawler:
- Auths with Atlassian Cloud and exports HTML + a JSON sidecar per page.
- Rewrites <img> tags to local asset paths and downloads attachments.
- Skips unchanged pages using server last-modified vs local file mtime.
- Optionally cleans up orphaned local files (pages deleted upstream).
- Designed to run on a schedule (e.g., every 48h via docker-compose loop).

Env (.env / .env.docker):
  ATLASSIAN_URL, ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN
  ATLASSIAN_SPACES="WindsorSupport,NPKB"
  HTML_DIR=/workspace/crawler/confluence_export
  SLEEP_BETWEEN_REQUESTS=0.5
  PAGE_LIMIT=500
  MAX_PAGES_TOTAL=100000
  FOLLOW_ALL_SPACES=false
  REQUEST_TIMEOUT=30
  RETRIES=3
  BACKOFF_BASE=2.0
  CLEAN_ORPHANS=true           # delete local files for pages no longer seen
"""

import os
import re
import json
import time
import hashlib
import requests
from urllib.parse import urljoin, urlparse
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set, Tuple, List

from atlassian import Confluence
from bs4 import BeautifulSoup

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from env_bootstrap import load_env
from tqdm import tqdm

# ------------------- CONFIG VIA .env -------------------
ROOT = load_env(__file__)

BASE_URL = os.getenv("ATLASSIAN_URL", "").rstrip("/")
EMAIL = os.getenv("ATLASSIAN_EMAIL", "")
TOKEN = os.getenv("ATLASSIAN_API_TOKEN", "")
SPACES = [s.strip() for s in os.getenv("ATLASSIAN_SPACES", "WindsorSupport,NPKB").split(",") if s.strip()]

EXPORT_DIR = Path(os.getenv("HTML_DIR", "./confluence_export"))
STATE_FILE = EXPORT_DIR / "_crawler_state.json"  # persists last-mod + seen
SLEEP_BETWEEN = float(os.getenv("SLEEP_BETWEEN_REQUESTS", "0.5"))  # seconds
PAGE_LIMIT = int(os.getenv("PAGE_LIMIT", "500"))                   # per list call
MAX_PAGES_TOTAL = int(os.getenv("MAX_PAGES_TOTAL", "100000"))      # safety guard
FOLLOW_ALL_SPACES = os.getenv("FOLLOW_ALL_SPACES", "false").lower() == "true"

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
RETRIES = int(os.getenv("RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "2.0"))

CLEAN_ORPHANS = os.getenv("CLEAN_ORPHANS", "true").lower() == "true"

# --------------- HELPERS ---------------

def log(msg: str) -> None:
    print(f"[crawler] {msg}", flush=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|\r\n]+', "_", name).strip()[:180]

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def parse_lastmod(content: dict) -> str | None:
    v = content.get("version") or {}
    when = v.get("when")
    if not when:
        h = content.get("history") or {}
        lu = h.get("lastUpdated") or {}
        when = lu.get("when")
    return when

def parse_space_key(content: dict) -> str | None:
    sp = content.get("space")
    if isinstance(sp, dict):
        return sp.get("key")
    return None

def _soup(html: str, prefer_xml=True) -> Tuple[BeautifulSoup, str]:
    if prefer_xml:
        try:
            return BeautifulSoup(html, "xml"), "xml"
        except Exception:
            pass
    return BeautifulSoup(html, "html.parser"), "html"

def extract_linked_page_ids_from_storage(html: str) -> Set[str]:
    found: Set[str] = set()
    soup_xml, _ = _soup(html, prefer_xml=True)
    for tag in soup_xml.find_all(True):
        if tag.name and tag.name.endswith("page"):
            cid = tag.attrs.get("ri:content-id") or tag.attrs.get("content-id") or tag.attrs.get("ri_content_id")
            if cid and str(cid).isdigit():
                found.add(str(cid))
    soup_html, _ = _soup(html, prefer_xml=False)
    for a in soup_html.find_all("a", href=True):
        href = a["href"]
        try:
            parsed = urlparse(href)
            path = parsed.path if parsed.scheme else href
        except Exception:
            path = href
        m = re.search(r"/pages/(\d+)", path)
        if m:
            found.add(m.group(1))
    return found

def file_is_current(filepath: Path, last_modified_iso: str) -> bool:
    if not filepath.exists():
        return False
    try:
        file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
        api_time = datetime.fromisoformat(last_modified_iso.replace("Z", "+00:00"))
        return api_time <= file_mtime
    except Exception:
        return False

def safe_sleep(seconds: float):
    if seconds > 0: time.sleep(seconds)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def guess_filename_from_url(u: str) -> str:
    try:
        path = urlparse(u).path
        name = Path(path).name
        return sanitize_filename(name) or "file.bin"
    except Exception:
        return "file.bin"

def http_get_with_retries(sess: requests.Session, url: str, **kw) -> requests.Response:
    for i in range(RETRIES):
        try:
            return sess.get(url, timeout=REQUEST_TIMEOUT, **kw)
        except Exception as e:
            if i == RETRIES - 1:
                raise
            wait = BACKOFF_BASE ** i
            log(f"[retry] GET {url} failed ({e}); retry in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")

def download_and_rewrite_images(
    html: str,
    page_id: str,
    out_dir: Path,
    base_url: str,
    email: str,
    token: str
) -> Tuple[str, List[Dict]]:
    ensure_dir(out_dir)
    assets_dir = out_dir / "assets" / str(page_id)
    ensure_dir(assets_dir)

    sess = requests.Session()
    sess.auth = (email, token)

    soup_xml, _ = _soup(html, prefer_xml=True)
    soup_html, _ = _soup(html, prefer_xml=False)

    to_download: List[str] = []

    # 1) ac:image with ri:attachment
    for ac_img in soup_xml.find_all(lambda t: t.name == "ac:image"):
        ri_att = None
        for child in ac_img.find_all(True):
            if child.name and child.name.endswith("attachment"):
                ri_att = child
                break
        if ri_att:
            fn = ri_att.get("ri:filename") or ri_att.get("filename") or ""
            if fn:
                rel_url = f"/download/attachments/{page_id}/{fn}"
                to_download.append(rel_url)

    # 2) <img src="..."> with download endpoints
    for img in soup_html.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        if "/download/attachments/" in src or src.startswith("/wiki/download/attachments/"):
            to_download.append(src)

    # Deduplicate while preserving order
    to_download = list(dict.fromkeys(to_download))

    src_to_rel: Dict[str, str] = {}
    assets_meta: List[Dict] = []

    for src in to_download:
        abs_url = urljoin(base_url + "/", src.lstrip("/"))
        try:
            r = http_get_with_retries(sess, abs_url)
            r.raise_for_status()
        except Exception as e:
            log(f"[IMG] Failed to fetch {abs_url}: {e}")
            continue

        filename = sanitize_filename(guess_filename_from_url(abs_url))
        local_path = assets_dir / filename
        try:
            local_path.write_bytes(r.content)
        except Exception as e:
            log(f"[IMG] Failed to write {local_path}: {e}")
            continue

        rel_path = os.path.relpath(local_path, start=out_dir).replace("\\", "/")
        src_to_rel[src] = rel_path

        assets_meta.append({
            "filename": filename,
            "path": rel_path,               # relative to EXPORT_DIR
            "size": len(r.content),
            "sha256": sha256_bytes(r.content),
            "source_url": abs_url
        })

    # Rewrite <img> src in the HTML soup
    if src_to_rel:
        for img in soup_html.find_all("img", src=True):
            old = img["src"]
            if old in src_to_rel:
                img["src"] = src_to_rel[old]
            else:
                # try filename match (handles absolute vs relative link variants)
                for k, v in src_to_rel.items():
                    try:
                        if guess_filename_from_url(old) == guess_filename_from_url(k):
                            img["src"] = v
                            break
                    except Exception:
                        pass

    return str(soup_html), assets_meta

def page_filename(space_key: str, page_id: str, title: str) -> str:
    safe_title = sanitize_filename(title or f"page_{page_id}")
    return f"{space_key}_{page_id}_{safe_title}.html"

def discover_local_pages() -> Dict[str, Path]:
    """
    Returns mapping of page_id -> html Path for all exported pages currently on disk.
    """
    mapping: Dict[str, Path] = {}
    for p in EXPORT_DIR.glob("*.html"):
        # Expect pattern <space>_<id>_<title>.html
        m = re.match(r"^[^_]+_(\d+)_", p.name)
        if m:
            mapping[m.group(1)] = p
    return mapping

# --------------- MAIN ---------------
def main():
    if not (BASE_URL and EMAIL and TOKEN):
        raise SystemExit("Missing ATLASSIAN_URL / ATLASSIAN_EMAIL / ATLASSIAN_API_TOKEN in .env")

    confluence = Confluence(url=BASE_URL, username=EMAIL, password=TOKEN)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()  # { "seen": {page_id: lastmod}, "since": {space_key: iso} }
    state.setdefault("seen", {})
    state.setdefault("since", {})

    # seed queue with all pages from the target spaces
    queue: deque[str] = deque()
    visited_run: Set[str] = set()

    total_seeded = 0
    for space in SPACES:
        start = 0
        while True:
            try:
                res = confluence.get_all_pages_from_space(space, limit=PAGE_LIMIT, start=start, expand=None)
            except Exception as e:
                log(f"[WARN] list pages failed for space {space}: {e}")
                break
            if not res:
                break
            for p in res:
                pid = p.get("id")
                if pid and pid not in visited_run:
                    queue.append(pid)
                    visited_run.add(pid)
                    total_seeded += 1
            if len(res) < PAGE_LIMIT:
                break
            start += PAGE_LIMIT
            safe_sleep(SLEEP_BETWEEN)
    log(f"Seeded {total_seeded} pages from spaces: {', '.join(SPACES)}")

    # BFS follow links
    saved_count = 0
    processed_count = 0

    pbar = tqdm(total=len(queue), desc="Crawling", colour="blue", leave=True)
    while queue and processed_count < MAX_PAGES_TOTAL:
        page_id = queue.popleft()
        processed_count += 1
        pbar.update(1)

        # fetch the page with needed expands to get space + lastmod + storage HTML
        content = None
        for attempt in range(RETRIES):
            try:
                content = confluence.get_page_by_id(
                    page_id,
                    expand="body.storage,version,history,space"
                )
                break
            except Exception as e:
                wait = BACKOFF_BASE ** attempt
                log(f"[WARN] get_page_by_id({page_id}) failed ({e}). retry in {wait:.1f}s")
                safe_sleep(wait)
        if not content or "body" not in content:
            continue

        title = sanitize_filename(content.get("title", f"page_{page_id}"))
        last_mod = parse_lastmod(content) or datetime.now(timezone.utc).isoformat()
        space_key = parse_space_key(content) or "UNKNOWN"

        # If we're restricting to certain spaces, skip others
        if (not FOLLOW_ALL_SPACES) and (space_key not in SPACES):
            continue

        out_name = page_filename(space_key, page_id, title)
        out_path = EXPORT_DIR / out_name

        # fetch storage HTML (again if needed)
        html_val = ((content.get("body") or {}).get("storage") or {}).get("value", "")
        if not html_val:
            try:
                detail = confluence.get_page_by_id(page_id, expand="body.storage")
                html_val = ((detail.get("body") or {}).get("storage") or {}).get("value", "")
            except Exception:
                html_val = ""
        if not html_val:
            continue

        # if current, still follow links but skip write
        if last_mod and file_is_current(out_path, last_mod):
            for linked_id in extract_linked_page_ids_from_storage(html_val):
                if linked_id not in visited_run:
                    queue.append(linked_id); visited_run.add(linked_id); pbar.total += 1
            state["seen"][page_id] = last_mod
            continue

        # download images + rewrite
        rewritten_html, assets_meta = download_and_rewrite_images(
            html=html_val,
            page_id=page_id,
            out_dir=EXPORT_DIR,
            base_url=BASE_URL,
            email=EMAIL,
            token=TOKEN
        )

        # write updated HTML
        out_path.write_text(rewritten_html, encoding="utf-8")

        # set mtime to server lastmod for freshness checks
        try:
            api_dt = datetime.fromisoformat(last_mod.replace("Z", "+00:00"))
            ts = api_dt.timestamp()
            os.utime(out_path, (ts, ts))
        except Exception:
            pass

        # sidecar
        links = (content.get("_links") or {})
        webui = links.get("webui") or f"/pages/{page_id}"
        page_url = f"{BASE_URL}{webui}"
        assets_rel = [a["path"] for a in assets_meta]

        sidecar = {
            "space_key": space_key,
            "page_id": page_id,
            "title": title,
            "url": page_url,
            "last_modified": last_mod,
            "html_path": os.path.relpath(out_path, start=EXPORT_DIR).replace("\\", "/"),
            "assets": assets_rel,
            "assets_meta": assets_meta
        }
        meta_path = out_path.with_suffix(".json")
        meta_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        state["seen"][page_id] = last_mod
        saved_count += 1

        # enqueue linked pages
        for linked_id in extract_linked_page_ids_from_storage(html_val):
            if linked_id not in visited_run:
                queue.append(linked_id); visited_run.add(linked_id); pbar.total += 1

        safe_sleep(SLEEP_BETWEEN)

    pbar.close()

    # Optional: remove local orphans (files for pages not seen this run)
    if CLEAN_ORPHANS:
        current_local = discover_local_pages()
        still_seen_ids = set(state["seen"].keys())  # pages we've ever seen (persisted)
        # We only delete files that exist on disk but weren't processed/queued this run.
        # Safer: compute "touched this run" by visited_run union of processed queue seeds.
        touched_this_run = visited_run
        deletions = 0
        for pid, path in current_local.items():
            if pid not in touched_this_run:
                try:
                    side = path.with_suffix(".json")
                    if side.exists():
                        side.unlink(missing_ok=True)
                    path.unlink(missing_ok=True)
                    deletions += 1
                except Exception as e:
                    log(f"[CLEANUP] failed to delete {path}: {e}")
        if deletions:
            log(f"[CLEANUP] removed {deletions} local files no longer present upstream.")

    save_state(state)
    log(f"[DONE] processed={processed_count}, saved/updated={saved_count}, out={EXPORT_DIR.resolve()}")
    if processed_count >= MAX_PAGES_TOTAL:
        log("[NOTE] Hit MAX_PAGES_TOTAL safety limit. Increase via .env if needed.")

if __name__ == "__main__":
    main()
