# --- sqlite shim kept harmless; OK if unused ---
import sys
try:
    import sqlite3
    v = tuple(map(int, sqlite3.sqlite_version.split(".")))
    if v < (3, 35, 0):
        import pysqlite3 as _pysqlite3
        sys.modules["sqlite3"] = _pysqlite3
except Exception:
    pass

import streamlit as st
import time
from simple_chat_manager import SimpleChatManager  # <- lightweight, no Chroma

# Page config FIRST and only once
st.set_page_config(page_title="SnuggleBots", page_icon="🐾", layout="wide")

# Colors
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f5e1dc; }
[data-testid="stSidebar"], [data-testid="stSidebarContent"] { background-color: #f8d6d0; }
h1, .main-header { color: #da6274 !important; }
.stButton > button {
  background-color: #da6274 !important; color: #f5e1dc !important;
  border: none !important; border-radius: 6px !important;
}
.stButton > button:hover { filter: brightness(0.95); }
</style>
""", unsafe_allow_html=True)

# State
if "chat" not in st.session_state:
    st.session_state.chat = SimpleChatManager(system_prompt=(
        "You are SnuggleBots, a concise pet-adoption assistant. "
        "Answer clearly. If unsure, say so. No medical advice beyond general care tips."
    ))
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about pet adoption or basic pet care."}
    ]

# UI
st.markdown('<h1 class="main-header">🐾 SnuggleBots — Chat</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    st.caption("Uses OpenAI only. No indexing or initialization.")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    st.session_state.chat.set_config(model=model, temperature=temp)

# Chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t0 = time.time()
            reply = st.session_state.chat.ask(prompt)
            dt = time.time() - t0
            st.markdown(reply or "_(no response)_")
            st.caption(f"Response time: {dt:.2f}s")

    st.session_state.messages.append({"role": "assistant", "content": reply or ""})
