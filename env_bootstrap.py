# env_bootstrap.py (repo root)
from pathlib import Path
import os
from dotenv import load_dotenv

def load_env(this_file: str):
    """
    Load envs consistently for any entry script.
    Order:
      1) .env
      2) .env.local (overrides for your dev box)
      3) .env.docker (overrides inside containers or if RUNNING_IN_DOCKER=1)
    Returns the repo root Path.
    """
    root = Path(this_file).resolve().parents[1]

    load_dotenv(root / ".env")
    load_dotenv(root / ".env.local", override=True)

    if Path("/.dockerenv").exists() or os.getenv("RUNNING_IN_DOCKER") == "1":
        load_dotenv(root / ".env.docker", override=True)

    # Safe default so everyone shares the same export path if not set
    os.environ.setdefault("HTML_DIR", str((root / "crawler" / "confluence_export").resolve()))
    return root
