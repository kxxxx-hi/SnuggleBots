import sys

# ---- Patch sqlite3 if system version is too old (for Chroma) ----
try:
    import sqlite3
    v = tuple(map(int, sqlite3.sqlite_version.split(".")))
    if v < (3, 35, 0):
        import pysqlite3 as _pysqlite3
        sys.modules["sqlite3"] = _pysqlite3
except Exception:
    import pysqlite3 as _pysqlite3
    sys.modules["sqlite3"] = _pysqlite3

"""
Streamlit Web Interface for the Proposed RAG System
"""

import streamlit as st
import time
from typing import Dict, Any
import json

# ---- Configure Streamlit (must be the first st.* call) ----
st.set_page_config(page_title="SnuggleBots", page_icon="🐾", layout="wide")

# ---- Import your RAG Manager ----
from proposed_rag_system import ProposedRAGManager

# ---- Initialize manager (adjust args if needed) ----
rag_manager = ProposedRAGManager()

# ---- Sidebar ----
st.sidebar.title("SnuggleBots Control Panel")
st.sidebar.info("Configure your chatbot here")

# ---- Main app ----
st.title("🐾 SnuggleBots: Pet Chatbot")

user_input = st.text_input("Ask me something about your pet:", "")
if st.button("Send") and user_input:
    with st.spinner("Thinking..."):
        try:
            response: Dict[str, Any] = rag_manager.query(user_input)
            st.success(response.get("answer", "No answer returned."))
            with st.expander("Debug / Full response"):
                st.json(response)
        except Exception as e:
            st.error(f"Error while processing: {e}")

# ---- Footer ----
st.caption("Powered by LangChain + Streamlit")
