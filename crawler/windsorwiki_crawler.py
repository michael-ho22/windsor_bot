#!/usr/bin/env python3
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

from atlassian import Confluence
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

# ------------------- CONFIG VIA .env -------------------
load_dotenv()

BASE_URL = os.getenv("ATLASSIAN_URL", "").rstrip("/")
EMAIL = os.getenv("ATLASSIAN_EMAIL", "")
TOKEN = os.getenv("ATLASSIAN_API_TOKEN", "")
SPACES = [s.strip() for s in os.getenv("ATLASSIAN_SPACES", "WindsorSupport,NPKB").split(",") if s.strip()]

EXPORT_DIR = Path(os.getenv("HTML_DIR", "./confluence_export"))
STATE_FILE = EXPORT_DIR / "_crawler_state.json"  # persists last-mod + seen
SLEEP_BETWEEN = float(os.getenv("SLEEP_BETWEEN_REQUESTS", "0.5"))  # seconds
PAGE_LIMIT = int(os.getenv("PAGE_LIMIT", "500"))  # per API page list call
MAX_PAGES_TOTAL = int(os.getenv("MAX_PAGES_TOTAL", "100000"))  # safety guard
FOLLOW_ALL_SPACES = os.getenv("FOLLOW_ALL_SPACES", "false").lower() == "true"
# if FOLLOW_ALL_SPACES=false, we will only persist/follow pages whose space key is in SPACES

# --------------- HELPERS ---------------

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|\r\n]+', "_", name).strip()[:180]

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

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
    # prefer version.when, fallback to history.lastUpdated.when
    when = None
    v = content.get("version") or {}
    when = v.get("when")
    if not when:
        h = content.get("history") or {}
        lu = h.get("lastUpdated") or {}
        when = lu.get("when")
    return when

def parse_space_key(content: dict) -> str | None:
    # NOTE: space key is not always expanded by default. We request 'space' in expand where needed.
    sp = content.get("space")
    if isinstance(sp, dict):
        return sp.get("key")
    return None

def _soup(html: str, prefer_xml=True):
    """
    Try XML parser first (needs lxml). If unavailable or fails, fallback to html.parser.
    """
    if prefer_xml:
        try:
            return BeautifulSoup(html, "xml"), "xml"
        except Exception:
            pass
    # fallback
    return BeautifulSoup(html, "html.parser"), "html"

def extract_linked_page_ids_from_storage(html: str) -> set[str]:
    """
    Extract linked Confluence page IDs from storage content.
    Handles both <ac:link><ri:page ri:content-id="123"/></ac:link>
    and <a href="/spaces/KEY/pages/123/Title"> links.
    """
    found = set()

    # 1) Namespaced ri:page with ri:content-id
    soup_xml, _ = _soup(html, prefer_xml=True)
    for tag in soup_xml.find_all(True):
        if tag.name and tag.name.endswith("page"):
            cid = tag.attrs.get("ri:content-id") or tag.attrs.get("content-id") or tag.attrs.get("ri_content_id")
            if cid and str(cid).isdigit():
                found.add(str(cid))

    # 2) <a href=".../pages/<id>/..."> patterns
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
    """
    Compare file mtime vs page last-modified (ISO). Returns True if local file is up-to-date.
    """
    if not filepath.exists():
        return False
    try:
        file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
        api_time = datetime.fromisoformat(last_modified_iso.replace("Z", "+00:00"))
        return api_time <= file_mtime
    except Exception:
        # If parsing fails, be safe and refetch
        return False

def safe_sleep(seconds: float):
    if seconds > 0:
        time.sleep(seconds)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def guess_filename_from_url(u: str) -> str:
    # Strip querystring; keep last path segment as name
    try:
        path = urlparse(u).path
        name = Path(path).name
        return sanitize_filename(name) or "file.bin"
    except Exception:
        return "file.bin"

def download_and_rewrite_images(
    html: str,
    page_id: str,
    out_dir: Path,
    base_url: str,
    email: str,
    token: str
) -> tuple[str, list[dict]]:
    """
    - Finds images/attachments in Confluence storage:
      * <ac:image><ri:attachment ri:filename="..."/></ac:image>
      * <img src="/download/attachments/<pageId>/...">
    - Downloads each asset with auth.
    - Saves to: out_dir / 'assets' / page_id / <filename>
    - Rewrites <img src=> links to relative asset paths.
    - Returns (rewritten_html, assets_meta[])
    """
    ensure_dir(out_dir)
    assets_dir = out_dir / "assets" / str(page_id)
    ensure_dir(assets_dir)

    sess = requests.Session()
    sess.auth = (email, token)

    soup_xml, _ = _soup(html, prefer_xml=True)
    soup_html, _ = _soup(html, prefer_xml=False)

    to_download = []

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

    # Deduplicate
    to_download = list(dict.fromkeys(to_download))

    # Download and rewrite
    src_to_rel = {}
    assets_meta: list[dict] = []

    for src in to_download:
        abs_url = urljoin(base_url + "/", src.lstrip("/"))
        try:
            r = sess.get(abs_url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"[IMG] Failed to fetch {abs_url}: {e}")
            continue

        filename = sanitize_filename(guess_filename_from_url(abs_url))
        local_path = assets_dir / filename
        try:
            local_path.write_bytes(r.content)
        except Exception as e:
            print(f"[IMG] Failed to write {local_path}: {e}")
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

    # Return rewritten HTML using HTML soup (browser-friendly)
    return str(soup_html), assets_meta

# --------------- MAIN ---------------
def main():
    if not (BASE_URL and EMAIL and TOKEN):
        raise SystemExit("Missing ATLASSIAN_URL / ATLASSIAN_EMAIL / ATLASSIAN_API_TOKEN in .env")

    confluence = Confluence(url=BASE_URL, username=EMAIL, password=TOKEN)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()  # { "seen": {page_id: lastmod}, "since": {space_key: iso} }
    state.setdefault("seen", {})
    state.setdefault("since", {})

    # Seed queue with all pages from the target spaces (home + all list)
    queue: deque[str] = deque()
    visited_run: set[str] = set()

    total_seeded = 0
    for space in SPACES:
        start = 0
        while True:
            try:
                res = confluence.get_all_pages_from_space(space, limit=PAGE_LIMIT, start=start, expand=None)
            except Exception as e:
                print(f"[WARN] list pages failed for space {space}: {e}")
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
    print(f"[INFO] Seeded {total_seeded} pages from spaces: {', '.join(SPACES)}")

    # BFS follow links
    saved_count = 0
    processed_count = 0

    pbar = tqdm(total=len(queue), desc="Crawling", colour="blue", leave=True)
    while queue and processed_count < MAX_PAGES_TOTAL:
        page_id = queue.popleft()
        processed_count += 1
        pbar.update(1)

        # fetch the page with needed expands to get space + lastmod + storage HTML
        for attempt in range(3):
            try:
                content = confluence.get_page_by_id(
                    page_id,
                    expand="body.storage,version,history,space"
                )
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"[WARN] get_page_by_id({page_id}) failed ({e}). retry in {wait}s")
                safe_sleep(wait)
        else:
            # 3 attempts failed
            continue

        if not content or "body" not in content:
            continue

        title = sanitize_filename(content.get("title", f"page_{page_id}"))
        last_mod = parse_lastmod(content) or datetime.now(timezone.utc).isoformat()
        space_key = parse_space_key(content) or "UNKNOWN"

        # If we're restricting to certain spaces, skip others
        if (not FOLLOW_ALL_SPACES) and (space_key not in SPACES):
            continue

        # save filename
        out_name = f"{space_key}_{page_id}_{title}.html"
        out_path = EXPORT_DIR / out_name

        # fetch HTML value (storage)
        html_val = ((content.get("body") or {}).get("storage") or {}).get("value", "")
        if not html_val:
            try:
                detail = confluence.get_page_by_id(page_id, expand="body.storage")
                html_val = ((detail.get("body") or {}).get("storage") or {}).get("value", "")
            except Exception:
                html_val = ""

        if not html_val:
            # nothing to save, but still try to follow links if any
            continue

        # skip if file is current (but still discover new links)
        if last_mod and file_is_current(out_path, last_mod):
            for linked_id in extract_linked_page_ids_from_storage(html_val):
                if linked_id not in visited_run:
                    queue.append(linked_id)
                    visited_run.add(linked_id)
                    pbar.total += 1
            continue

        # Download images + rewrite HTML to local asset paths
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

        # Build sidecar metadata (URL, assets, etc.)
        links = (content.get("_links") or {})
        webui = links.get("webui") or f"/pages/{page_id}"
        page_url = f"{BASE_URL}{webui}"

        # assets list relative to EXPORT_DIR for portability
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
        linked = extract_linked_page_ids_from_storage(html_val)
        for linked_id in linked:
            if linked_id not in visited_run:
                queue.append(linked_id)
                visited_run.add(linked_id)
                pbar.total += 1  # update progress bar length dynamically

        safe_sleep(SLEEP_BETWEEN)

    pbar.close()
    save_state(state)
    print(f"[DONE] processed={processed_count}, saved/updated={saved_count}, out={EXPORT_DIR.resolve()}")
    if processed_count >= MAX_PAGES_TOTAL:
        print("[NOTE] Hit MAX_PAGES_TOTAL safety limit. Increase it if needed via .env")

if __name__ == "__main__":
    main()
