"""
Streamlit frontend for RAG microservice.
Query and ingest via the backend API.
"""

import os
import streamlit as st
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL.rstrip('/')}/api/v1"


def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def main():
    st.set_page_config(page_title="RAG Microservice", page_icon="📚", layout="wide")
    st.title("📚 RAG Microservice")
    st.caption(f"Backend: `{BACKEND_URL}`")

    if not check_backend():
        st.error(f"Backend not reachable at {BACKEND_URL}. Start it with: `docker compose up` or `uvicorn app.main:app` from backend/.")
        return

    tab1, tab2, tab3 = st.tabs(["🔍 Query", "📤 Ingest", "ℹ️ Info"])

    with tab1:
        st.subheader("Ask a question")
        query = st.text_area("Question", placeholder="e.g. What is RAG?", height=100)
        top_k = st.slider("Number of chunks to retrieve", 1, 20, 5)
        if st.button("Get answer", type="primary") and query.strip():
            with st.spinner("Querying..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/query",
                        json={"query": query.strip(), "top_k": top_k},
                        timeout=60,
                    )
                    r.raise_for_status()
                    data = r.json()
                    st.markdown("### Answer")
                    st.markdown(data.get("answer", "No answer."))
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander("📎 Retrieved sources", expanded=False):
                            for i, s in enumerate(sources, 1):
                                st.text_area(f"Chunk {i}", value=s, height=120, disabled=True)
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")
                except Exception as e:
                    st.error(str(e))

    with tab2:
        st.subheader("Upload documents (PDF or text)")
        uploaded = st.file_uploader("Choose files", type=["pdf", "txt"], accept_multiple_files=True)
        if st.button("Ingest", type="primary") and uploaded:
            with st.spinner("Ingesting..."):
                try:
                    files = [("files", (f.name, f.read())) for f in uploaded]
                    r = requests.post(
                        f"{API_BASE}/ingest",
                        files=files,
                        timeout=120,
                    )
                    r.raise_for_status()
                    data = r.json()
                    ingested = data.get("ingested", [])
                    errors = data.get("errors", [])
                    if ingested:
                        st.success(f"Ingested: {', '.join(ingested)}")
                    if errors:
                        for err in errors:
                            st.warning(f"{err.get('file', '?')}: {err.get('error', '')}")
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")
                except Exception as e:
                    st.error(str(e))

    with tab3:
        st.subheader("API")
        st.markdown("- **GET** `/health` — Health check")
        st.markdown("- **POST** `/api/v1/query` — RAG query (JSON: `query`, `top_k`)")
        st.markdown("- **POST** `/api/v1/ingest` — Ingest documents (form: `files`)")
        st.subheader("Run locally")
        st.code("cd backend && uvicorn app.main:app --reload --port 8000", language="bash")
        st.code("cd frontend && BACKEND_URL=http://localhost:8000 streamlit run app.py --server.port 8501", language="bash")


if __name__ == "__main__":
    main()
