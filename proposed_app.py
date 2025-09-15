# If you previously pasted any sqlite shim here, you can remove it.
# The shim in vector_store.py is enough.

import sys
try:
    import sqlite3
except Exception:
    pass  # not required here; vector_store handles sqlite

"""
Streamlit Web Interface for the Proposed RAG System
"""
import streamlit as st
import time
from typing import Dict, Any
import json

from proposed_rag_system import ProposedRAGManager

# ---- Page config (must be first Streamlit call; only once) ----
st.set_page_config(page_title="SnuggleBots", page_icon="🐾", layout="wide")
# ---------------------------------------------------------------

# ---- Custom CSS (theme colors) ----
st.markdown("""
<style>
/* page background */
html, body, [data-testid="stAppViewContainer"] {
  background-color: #f5e1dc !important;
}
/* sidebar background */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
  background-color: #f8d6d0 !important;
}
/* header/title color */
h1, .main-header {
  color: #da6274 !important;
}
/* buttons */
.stButton > button {
  background-color: #da6274 !important;
  color: #f5e1dc !important;
  border: none !important;
  border-radius: 6px !important;
}
.stButton > button:hover { filter: brightness(0.95); }

/* cards */
.answer-box { background:#f8f9fa; padding:1.0rem; border-radius:.5rem; border:1px solid #dee2e6; }
.citation-box { background:#e8f5e8; padding:.75rem; border-radius:.5rem; border-left:4px solid #28a745; }
.performance-box { background:#fff3cd; padding:.75rem; border-radius:.5rem; border-left:4px solid #ffc107; }
</style>
""", unsafe_allow_html=True)
# -----------------------------------

# Session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []


def initialize_system():
    """Initialize the proposed RAG system."""
    try:
        with st.spinner("Initializing proposed RAG system..."):
            st.session_state.rag_system = ProposedRAGManager()

            # Add documents from ./documents (if present)
            result = st.session_state.rag_system.add_directory("documents")

            if result.get('success', False):
                st.session_state.system_initialized = True
                st.success(f"✅ System initialized. Processed {result.get('documents_processed', 0)} documents.")
                return True
            else:
                st.error(f"❌ Failed to initialize: {result.get('error', 'Unknown error')}")
                return False
    except Exception as e:
        st.error(f"❌ Error initializing system: {e}")
        return False


def display_system_stats():
    if st.session_state.rag_system:
        stats = st.session_state.rag_system.get_stats()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Documents", stats.get('vector_store', {}).get('document_count', 0))
        col2.metric("🔍 BM25 Index", stats.get('bm25_documents', 0))
        col3.metric("🤖 Queries", stats.get('total_queries', 0))
        col4.metric("📈 Avg Confidence", f"{stats.get('avg_confidence', 0):.3f}")


def display_query_history():
    if st.session_state.query_history:
        st.subheader("📚 Recent Queries")
        for i, q in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Query {len(st.session_state.query_history) - i}: {q['query'][:50]}..."):
                st.write(f"**Question:** {q['query']}")
                st.write(f"**Answer:** {q['answer'][:200]}...")
                st.write(f"**Confidence:** {q['confidence']:.3f}")
                st.write(f"**Response Time:** {q['response_time']:.2f}s")


def main():
    st.markdown('<h1 class="main-header">🐾 SnuggleBots — Proposed RAG System</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🔧 System Configuration")
        st.write("Architecture: BM25 + Dense + RRF + Cross-encoder + Extractive")

        if not st.session_state.system_initialized:
            if st.button("🚀 Initialize System", type="primary"):
                initialize_system()
        else:
            st.success("✅ System Ready")
            if st.button("🔄 Reset System"):
                st.session_state.rag_system = None
                st.session_state.system_initialized = False
                st.session_state.query_history = []
                st.rerun()

        st.subheader("⚙️ Query Settings")
        use_reranking = st.checkbox("Use Cross-encoder Reranking", value=True)
        rerank_threshold = st.slider("Rerank Threshold", 0.0, 1.0, 0.1, 0.05)
        max_rerank = st.slider("Max Documents to Rerank", 5, 50, 20)

        if st.session_state.system_initialized:
            st.subheader("📊 System Statistics")
            display_system_stats()

    # Main
    if not st.session_state.system_initialized:
        st.info("👈 Initialize the system using the sidebar.")
        st.subheader("🏗️ System Architecture")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Retrieval Pipeline:**
            1. BM25 Retrieval  
            2. Dense Retrieval  
            3. RRF Fusion  
            4. Cross-encoder Reranking  
            5. Extractive Generation
            """)
        with col2:
            st.markdown("""
            **Benefits:**
            - Higher accuracy  
            - Lower API cost  
            - Better citations  
            - Hybrid search  
            - Fast responses
            """)
        return

    # Query UI
    st.subheader("💬 Ask a Pet Care Question")
    question = st.text_input("Enter your question:", placeholder="Ask anything about pet care...")

    examples = [
        "What should I feed my cat?",
        "How often should I take my dog to the vet?",
        "What vaccinations does my dog need?",
        "How can I prevent obesity in my pet?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"💡 {ex}", key=f"ex_{i}"):
                question = ex
                st.rerun()

    if question:
        if st.button("🔍 Search", type="primary"):
            with st.spinner("Processing your question..."):
                start = time.time()
                resp = st.session_state.rag_system.ask(
                    question,
                    use_reranking=st.session_state.get('use_reranking', True) if False else True,
                    rerank_threshold=0.1,
                    max_rerank=20,
                )
                dt = time.time() - start

                st.session_state.query_history.append({
                    "query": question,
                    "answer": resp["answer"],
                    "confidence": resp["confidence"],
                    "response_time": dt,
                    "timestamp": time.time(),
                })

                st.subheader("🤖 Answer")
                st.markdown(f'<div class="answer-box">{resp["answer"]}</div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.metric("📊 Confidence", f"{resp['confidence']:.3f}")
                c2.metric("📚 Sources", len(resp.get('sources', [])))

                if resp.get("citations"):
                    st.subheader("📖 Citations")
                    for i, cit in enumerate(resp["citations"]):
                        st.markdown(
                            f'<div class="citation-box"><b>Source {i+1}:</b> {cit.get("source","")}<br>'
                            f'<b>Relevance:</b> {cit.get("relevance_score",0):.3f}<br>'
                            f'<b>Preview:</b> {cit.get("content_preview","")}</div>',
                            unsafe_allow_html=True,
                        )

                st.subheader("⚡ Performance")
                perf = resp.get("performance", {})
                retr = resp.get("retrieval_info", {})
                p1, p2 = st.columns(2)
                p1.markdown(
                    f'<div class="performance-box">'
                    f'<b>Total:</b> {perf.get("total_time_ms",0):.1f} ms<br>'
                    f'<b>Retrieval:</b> {perf.get("retrieval_time_ms",0):.1f} ms<br>'
                    f'<b>Fusion:</b> {perf.get("fusion_time_ms",0):.1f} ms<br>'
                    f'<b>Reranking:</b> {perf.get("rerank_time_ms",0):.1f} ms<br>'
                    f'<b>Generation:</b> {perf.get("generation_time_ms",0):.1f} ms'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                p2.markdown(
                    f'<div class="performance-box">'
                    f'<b>BM25:</b> {retr.get("bm25_results",0)} | '
                    f'<b>Dense:</b> {retr.get("dense_results",0)} | '
                    f'<b>Fused:</b> {retr.get("fused_results",0)} | '
                    f'<b>Reranked:</b> {retr.get("reranked_results",0)} | '
                    f'<b>Reranking Used:</b> {retr.get("use_reranking", True)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    display_query_history()

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#333;'>🐾 SnuggleBots — BM25 + Dense + RRF + Cross-encoder + Extractive Generation</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
