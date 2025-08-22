# api/auth.py
import os, time, hmac, base64, hashlib, json
from typing import Optional

import bcrypt
from fastapi import HTTPException, status

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ISS = os.getenv("JWT_ISSUER", "windsor-bot")
JWT_EXP_SECS = int(os.getenv("JWT_EXP_SECS", "86400"))  # 24h

# ---- Passwords ----
def hash_password(plain: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

# ---- Minimal JWT (HS256) ----
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

def _b64url_json(obj: dict) -> str:
    return _b64url(json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

def _hmac_sha256(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()

def create_jwt(sub: str, extra: Optional[dict]=None, exp_secs: Optional[int]=None) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {"iss": JWT_ISS, "sub": sub, "iat": now, "exp": now + (exp_secs or JWT_EXP_SECS)}
    if extra:
        payload.update(extra)
    head = _b64url_json(header)
    body = _b64url_json(payload)
    signing_input = f"{head}.{body}".encode("ascii")
    sig = _b64url(_hmac_sha256(JWT_SECRET.encode("utf-8"), signing_input))
    return f"{head}.{body}.{sig}"

def decode_jwt(token: str) -> dict:
    try:
        head_b64, body_b64, sig_b64 = token.split(".")
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    signing_input = f"{head_b64}.{body_b64}".encode("ascii")
    expected = _b64url(_hmac_sha256(JWT_SECRET.encode("utf-8"), signing_input))
    if not hmac.compare_digest(expected, sig_b64):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad signature")

    body_json = base64.urlsafe_b64decode(body_b64 + "==")
    payload = json.loads(body_json.decode("utf-8"))

    if payload.get("iss") != JWT_ISS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad issuer")
    if int(time.time()) >= int(payload.get("exp", 0)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")

    return payload
