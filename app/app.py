import os
import requests
from requests import HTTPError
import streamlit as st
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from env_bootstrap import load_env
ROOT = load_env(__file__)

# ---------- Config ----------
RUNNING_IN_DOCKER = os.getenv("RUNNING_IN_DOCKER") == "1"
API_URL = os.getenv("API_URL") or ("http://api:8000" if RUNNING_IN_DOCKER else "http://127.0.0.1:8001")

# Put windsor_logo.png next to this file OR set WINDSOR_LOGO to an absolute path
LOGO_PATH = os.getenv("WINDSOR_LOGO")

st.set_page_config(page_title="Windsor Knowledge Bot", layout="wide")

# ---------- Session State ----------
ss = st.session_state
ss.setdefault("token", None)
ss.setdefault("user_email", None)
ss.setdefault("session_id", None)
ss.setdefault("messages", [])
ss.setdefault("theme_mode", "dark")   # 'dark' | 'light'
ss.setdefault("show_signup", False)

# ---------- Theme (Windsor-ish) ----------
def inject_theme():
    # Windsor palette
    teal       = "#2aa49a"      # brand teal
    teal_dark  = "#14877e"      # hover/active
    teal_faint = "#e6f5f3"      # faint teal fill

    # Dark theme
    dark_bg, dark_panel, dark_text, dark_subtle = "#0f1116", "#161a22", "#e8eaed", "#9aa3af"

    # Light theme (everything light)
    light_bg, light_panel = "#f7fbfa", "#ffffff"
    light_text, light_subtle = "#2aa49a", "#5aaea7"

    mode = st.session_state.get("theme_mode", "dark")
    if mode == "dark":
        bg, panel, text, subtle = dark_bg, dark_panel, dark_text, dark_subtle
    else:
        bg, panel, text, subtle = light_bg, light_panel, light_text, light_subtle

    st.markdown(
        f"""
        <style>
        :root {{
          --bg:{bg}; --panel:{panel}; --text:{text}; --subtle:{subtle};
          --teal:{teal}; --teal-dark:{teal_dark}; --teal-faint:{teal_faint};
        }}

        /* App surfaces — nuke any lingering dark areas */
        html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stMain"],
        .main, .block-container, header, footer {{
          background: var(--bg) !important;
          color: var(--text) !important;
        }}

        /* Sidebar */
        [data-testid="stSidebar"], [data-testid="stSidebar"] .block-container {{
          background: var(--panel) !important;
          color: var(--text) !important;
        }}

        /* Typography */
        .stApp, .stMarkdown, .stText, label, p, h1, h2, h3, h4, h5, h6, span {{
          color: var(--text) !important;
        }}
        a, a:visited {{ color: var(--teal-dark) !important; }}

        /* Buttons: teal pill with white text */
        .stButton>button, .st-download-button button, .stDownloadButton>button {{
          background: var(--teal) !important;
          border: 0 !important;
          color: #fff !important;
          border-radius: 10px !important;
          padding: .45rem .9rem !important;
          box-shadow: none !important;
        }}
        .stButton>button:hover, .st-download-button button:hover, .stDownloadButton>button:hover {{
          background: var(--teal-dark) !important;
        }}
        /* Force ALL text/icons inside buttons to be white */
        .stButton>button *, .st-download-button button *, .stDownloadButton>button * {{
          color:#fff !important; fill:#fff !important;
        }}

        /* Inputs */
        input, textarea, select,
        .stTextInput>div>div>input,
        .stTextArea textarea {{
          background: var(--panel) !important;
          color: var(--text) !important;
          border: 1px solid var(--teal) !important;
          border-radius: 10px !important;
        }}
        ::placeholder {{ color: color-mix(in srgb, var(--text) 60%, transparent) !important; }}

        /* Sliders & switches */
        [data-baseweb="slider"] [role="slider"] {{ background: var(--teal) !important; }}
        [data-baseweb="slider"] div[role="progressbar"] {{ background: var(--teal-faint) !important; }}
        .stSwitch [data-testid="stThumb"] {{ background: var(--teal) !important; }}

        /* Panels */
        .windsor-panel {{
          background: var(--panel) !important;
          border: 1px solid color-mix(in srgb, var(--teal) 25%, transparent) !important;
          border-radius: 12px !important;
          padding: .75rem !important;
        }}

        /* Header logo + title row */
        .windsor-title {{ display:flex; align-items:center; gap:.6rem; }}
        .windsor-title h1 {{ margin:0; font-weight:700; color: var(--text); }}

        /* Chat input fully light */
        [data-testid="stChatInput"] {{
          background: var(--panel) !important;
          border-top: 1px solid color-mix(in srgb, var(--teal) 25%, transparent) !important;
        }}
        [data-testid="stChatInput"] textarea {{
          background: var(--panel) !important;
          color: var(--text) !important;
          border: 1px solid var(--teal) !important;
        }}

        /* Chat list buttons (sidebar) */
        .chat-link-btn button {{
          width:100%; text-align:left;
          background: transparent !important;
          border:1px solid color-mix(in srgb, var(--teal) 25%, transparent) !important;
          color: var(--text) !important;
          border-radius:10px !important;
        }}
        .chat-link-btn button:hover {{
          border-color: var(--teal) !important;
          background: var(--teal-faint) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_theme()

# ---------- Helpers ----------
def auth_headers():
    return {"Authorization": f"Bearer {ss.token}"} if ss.token else {}

def fetch_sessions(limit=30):
    try:
        r = requests.get(f"{API_URL}/sessions", params={"limit": limit}, headers=auth_headers(), timeout=20)
        r.raise_for_status()
        return r.json().get("sessions", [])
    except Exception:
        return []

def load_session_messages(session_id: int):
    try:
        r = requests.get(f"{API_URL}/sessions/{session_id}", params={"limit": 500}, headers=auth_headers(), timeout=30)
        r.raise_for_status()
        msgs = r.json().get("messages", [])
        ss.messages = [{"role": m["role"], "content": m["content"]} for m in msgs]
        ss.session_id = session_id
    except HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = getattr(e.response, "text", str(e))
        st.sidebar.error(f"Could not load session: {e.response.status_code} — {detail}")
    except Exception as e:
        st.sidebar.error(f"Could not load session: {e}")

def _do_signup(email: str, pw: str):
    r = requests.post(f"{API_URL}/signup", json={"email": email, "password": pw}, timeout=20)
    r.raise_for_status()
    data = r.json()
    ss.token = data["token"]
    ss.user_email = data["user"]["email"]

def open_signup_ui():
    if hasattr(st, "dialog"):
        @st.dialog("Create your Windsor account")
        def _modal():
            signup_form_body()
        _modal()
    elif hasattr(st, "experimental_dialog"):
        @st.experimental_dialog("Create your Windsor account")
        def _modal():
            signup_form_body()
        _modal()
    else:
        ss.show_signup = True

def signup_form_body():
    email = st.text_input("Windsor Email", key="signup_email")
    pw = st.text_input("Password", type="password", key="signup_pw")
    cpw = st.text_input("Confirm Password", type="password", key="signup_cpw")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Cancel"):
            ss.show_signup = False
            st.rerun()
    with c2:
        if st.button("Create account"):
            if not email or not pw or not cpw:
                st.warning("Please fill in all fields."); return
            if not email.lower().endswith("@windsorsolutions.com"):
                st.warning("Email must be @windsorsolutions.com"); return
            if pw != cpw:
                st.warning("Passwords do not match."); return
            try:
                _do_signup(email, pw)
                st.success("Account created! You’re logged in.")
                ss.show_signup = False
                st.rerun()
            except HTTPError as e:
                try:
                    detail = e.response.json().get("detail", e.response.text)
                except Exception:
                    detail = getattr(e.response, "text", str(e))
                st.error(f"Sign up failed: {e.response.status_code} — {detail}")
            except Exception as e:
                st.error(f"Sign up failed: {e}")

# ---------- Settings modal ----------
def render_settings_modal():
    st.markdown(f"**User:** {ss.user_email or 'Not logged in'}")

    # FIX: toggle no longer flips on open. It only changes when you click.
    current_light = (ss.theme_mode == "light")
    want_light = st.toggle("Light mode" if not current_light else "Dark mode",
                           value=current_light, key="theme_toggle")
    if want_light != current_light:
        ss.theme_mode = "light" if want_light else "dark"
        inject_theme()
        st.rerun()

    st.divider()

    # Style
    st.subheader("Style")
    tone = st.selectbox("Tone", ["neutral","friendly","concise","formal","enthusiastic"],
                        index=0, key="tone_sel_modal")
    depth = st.selectbox("Depth", ["brief","balanced","thorough"], index=1, key="depth_sel_modal")
    if st.button("Save style"):
        try:
            r = requests.post(f"{API_URL}/settings",
                              json={"tone": tone, "depth": depth},
                              headers=auth_headers(), timeout=20)
            r.raise_for_status()
            st.success("Saved!")
        except HTTPError as e:
            try:
                detail = e.response.json().get("detail", e.response.text)
            except Exception:
                detail = getattr(e.response, "text", str(e))
            st.error(f"Save failed: {e.response.status_code} — {detail}")
        except Exception as e:
            st.error(f"Save failed: {e}")

    st.divider()

    # Retrieval & Memory
    st.subheader("Retrieval & Memory")
    ss.k       = st.slider("Results (k)", 4, 20, ss.get("k", 8))
    ss.alpha   = st.slider("Vector weight α", 0.0, 1.0, ss.get("alpha", 0.6), 0.05)
    ss.per_doc = st.slider("Max results per doc", 1, 5, ss.get("per_doc", 2))
    ss.min_sc  = st.slider("Min score", 0.0, 1.0, ss.get("min_sc", 0.0), 0.05)
    ss.spaces  = st.text_input("Restrict to spaces (comma-separated)", value=ss.get("spaces",""))
    ss.memory_note = st.text_input("Add a memory note (optional)", value=ss.get("memory_note",""))

# ---------- Header (logo + title + settings gear) ----------
c1, c2 = st.columns([9,1])
with c1:
    st.markdown('<div class="windsor-title">', unsafe_allow_html=True)
    if LOGO_PATH and Path(LOGO_PATH).exists():   # ⬅️ guard
        st.image(LOGO_PATH, width=110)
    else:
        st.markdown("<div style='font-size:58px; line-height:1'>▲</div>", unsafe_allow_html=True)
    st.markdown("<h1>Windsor Knowledge Bot</h1></div>", unsafe_allow_html=True)
with c2:
    if st.button("⚙️ Settings"):
        if hasattr(st, "dialog"):
            @st.dialog("Settings")
            def _settings_modal():
                render_settings_modal()
            _settings_modal()
        elif hasattr(st, "experimental_dialog"):
            @st.experimental_dialog("Settings")
            def _settings_modal():
                render_settings_modal()
            _settings_modal()
        else:
            st.session_state["_show_inline_settings"] = True

# ---------- Sidebar ----------
with st.sidebar:
    if not ss.token:
        st.subheader("Account")
        email = st.text_input("Windsor Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sign up"): open_signup_ui()
        with c2:
            if st.button("Log in"):
                try:
                    r = requests.post(f"{API_URL}/login", json={"email": email, "password": password}, timeout=20)
                    r.raise_for_status()
                    data = r.json()
                    ss.token = data["token"]; ss.user_email = data["user"]["email"]
                    st.success("Logged in!")
                    st.rerun()
                except HTTPError as e:
                    try: detail = e.response.json().get("detail", e.response.text)
                    except Exception: detail = getattr(e.response, "text", str(e))
                    st.error(f"Login failed: {e.response.status_code} — {detail}")
                except Exception as e:
                    st.error(f"Login failed: {e}")
        if ss.show_signup and not hasattr(st, "dialog") and not hasattr(st, "experimental_dialog"):
            st.info("Create your Windsor account")
            with st.container(border=True):
                signup_form_body()
    else:
        # header row w/ user + logout
        l, r = st.columns([3,1])
        with l: st.caption(f"Logged in as **{ss.user_email}**")
        with r:
            if st.button("Log out"):
                ss.token = ss.user_email = None
                ss.session_id = None; ss.messages = []; st.rerun()

        st.markdown("### Chats")

        # Chat history
        sessions = fetch_sessions(limit=50)
        if not sessions:
            st.caption("No chats yet.")
        else:
            for s in sessions:
                sid   = s["id"]
                title = s.get("title") or f"Chat {sid}"
                key   = f"chat_{sid}"
                # style class applied by CSS above
                if st.button(f"🗨️  {title}", key=key, use_container_width=True):
                    load_session_messages(sid)

# ---------- Toolbar ----------
col1, col2 = st.columns([3,1])
with col2:
    if st.button("🆕 New Chat"):
        ss.session_id = None
        ss.messages = []
        st.rerun()
with col1:
    st.caption("Ask a question. Answers are grounded with citations.")

# ---------- Render existing ----------
for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Chat input ----------
user_msg = st.chat_input("Type your message")
if user_msg:
    if os.getenv("AUTH_MODE", "multi").lower() != "none" and not ss.token:
        st.warning("Log in first to chat.")
    else:
        ss.messages.append({"role":"user","content":user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        payload = {
            "message": user_msg,
            "session_id": ss.session_id,
            "k": int(ss.get("k", 8)),
            "alpha": float(ss.get("alpha", 0.6)),
            "per_doc": int(ss.get("per_doc", 2)),
            "min_score": float(ss.get("min_sc", 0.0)),
        }
        # NEW: include title if user just created a chat
        if ss.get("new_chat_title"): 
            payload["title"] = ss.pop("new_chat_title")
        if ss.get("spaces","").strip(): payload["spaces"] = ss.spaces
        if ss.get("memory_note","").strip(): payload["memory_note"] = ss.memory_note

        headers = auth_headers()
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    r = requests.post(f"{API_URL}/answer", json=payload, headers=headers, timeout=120)
                    r.raise_for_status()
                    data = r.json()
                    ss.session_id = data["session_id"]
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    st.markdown(answer)

                    if sources:
                        st.markdown("**Sources**")
                        for s in sources:
                            title = s["title"] or s["url"]
                            url = s["url"] or "#"
                            score = s["score"]
                            st.markdown(f"- [{title}]({url}) — score {score:.3f}")

                    ss.messages.append({"role":"assistant","content":answer})
                except HTTPError as e:
                    try:
                        detail = e.response.json().get("detail", e.response.text)
                    except Exception:
                        detail = getattr(e.response, "text", str(e))
                    st.error(f"Request failed: {e.response.status_code} — {detail}")
                except Exception as e:
                    st.error(f"Request failed: {e}")
