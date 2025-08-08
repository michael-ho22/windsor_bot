import requests, os, streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")  # set in .env if different
st.set_page_config(page_title="Windsor Knowledge Bot", layout="wide")
st.title("Windsor Knowledge Bot")

q = st.text_input("Ask a question")
k = st.slider("Results (k)", 4, 20, 8)

def asset_url_to_full(u: str):
    # u is like "assets/12345/img.png" relative to the crawler HTML_DIR.
    # API serves /assets/*, so build absolute:
    if u.startswith("assets/"):
        return f"{API_URL}/{u}"
    # if already absolute
    return u

if st.button("Search") and q:
    r = requests.get(f"{API_URL}/search", params={"q": q, "k": k}).json()
    for i, hit in enumerate(r["results"], 1):
        st.markdown(f"**{i}. {hit['title']}**  \nScore: {hit['score']:.3f}  \nSpace: `{hit['space_key']}`")
        st.write(hit["text"][:800] + ("..." if len(hit["text"])>800 else ""))
        assets = hit.get("assets") or []
        if assets:
            with st.expander("Images"):
                for a in assets[:6]:  # thumb cap
                    st.image(asset_url_to_full(a), use_column_width=False)
        st.divider()

if st.button("Chat") and q:
    r = requests.get(f"{API_URL}/chat", params={"q": q, "k": k}).json()
    st.subheader("Answer")
    st.write(r["answer"])
    st.subheader("Sources")
    for i, s in enumerate(r["sources"], 1):
        st.markdown(f"- **{s['title']}**  (Score: {s['score']:.3f})")
        assets = s.get("assets") or []
        if assets:
            with st.expander(f"Images for {s['title']}"):
                for a in assets[:6]:
                    st.image(asset_url_to_full(a), use_column_width=False)
