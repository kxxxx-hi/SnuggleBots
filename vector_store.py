# vector_store.py
import os
import logging
from typing import List, Tuple, Optional

# Silence Chroma telemetry noise in logs
os.environ.setdefault("CHROMA_TELEMETRY", "False")

from langchain_chroma import Chroma  # modern wrapper (replaces langchain.vectorstores.Chroma)
# Keep the old name used elsewhere but map to the new class
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings

# Lightweight loaders for a few common doc types
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document

logger = logging.getLogger("vector_store")
logger.setLevel(logging.INFO)


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.error(f"Failed to read PDF {path}: {e}")
        return ""


def _read_docx(path: str) -> str:
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.error(f"Failed to read DOCX {path}: {e}")
        return ""


def _read_html(path: str) -> str:
    try:
        html = _read_text_file(path)
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")
    except Exception as e:
        logger.error(f"Failed to read HTML {path}: {e}")
        return ""


def _load_file(path: str) -> Tuple[str, str]:
    """
    Returns (content, metadata_source)
    """
    ext = os.path.splitext(path.lower())[1]
    if ext in {".md", ".txt", ".rst"}:
        return _read_text_file(path), path
    if ext in {".pdf"}:
        return _read_pdf(path), path
    if ext in {".docx"}:
        return _read_docx(path), path
    if ext in {".html", ".htm"}:
        return _read_html(path), path
    # Fallback: try reading as text
    return _read_text_file(path), path


class VectorStore:
    """
    Minimal vector store manager using:
      - langchain_huggingface.HuggingFaceEmbeddings
      - langchain_chroma.Chroma
    This replaces deprecated LangChain classes and avoids the old schema issue
    by using a new persist directory.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "pets",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        # Use a NEW folder name so we don't reuse an old incompatible schema
        self.persist_directory = (
            persist_directory
            if persist_directory
            else os.path.join(repo_root, "chroma_db_v2")
        )

        os.makedirs(self.persist_directory, exist_ok=True)

        logger.info("Loading sentence-transformers model: %s", model_name)
        # This name matches the package; keeps compatibility with prior code that expected
        # SentenceTransformerEmbeddings but now pulls from langchain-huggingface.
        self.embeddings = SentenceTransformerEmbeddings(model_name=model_name)

        logger.info("Initializing Chroma at %s (collection=%s)", self.persist_directory, collection_name)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    # ---------- Public API used by the rest of the app ----------

    def add_directory(self, directory: str) -> dict:
        """
        Recursively ingest supported files from a directory.
        Returns a summary dict.
        """
        abs_dir = os.path.abspath(directory)
        if not os.path.isdir(abs_dir):
            return {"success": False, "error": f"Directory not found: {abs_dir}"}

        texts: List[str] = []
        metadatas: List[dict] = []

        supported_exts = {".txt", ".md", ".rst", ".pdf", ".docx", ".html", ".htm"}
        count = 0

        for root, _, files in os.walk(abs_dir):
            for name in files:
                if os.path.splitext(name.lower())[1] in supported_exts:
                    path = os.path.join(root, name)
                    content, src = _load_file(path)
                    content = (content or "").strip()
                    if not content:
                        continue
                    texts.append(content)
                    metadatas.append({"source": src})
                    count += 1

        if not texts:
            return {"success": True, "documents_processed": 0}

        try:
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            self.vectorstore.persist()
            logger.info("Ingested %d documents into Chroma", len(texts))
            return {"success": True, "documents_processed": len(texts)}
        except Exception as e:
            logger.error("Error adding texts to Chroma: %s", e)
            return {"success": False, "error": str(e)}

    def similarity_search(self, query: str, k: int = 5):
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error("similarity_search failed: %s", e)
            return []

    def as_retriever(self, k: int = 5):
        try:
            return self.vectorstore.as_retriever(search_kwargs={"k": k})
        except Exception as e:
            logger.error("as_retriever failed: %s", e)
            return None

    def get_stats(self) -> dict:
        try:
            # Chroma client introspection is limited; approximate with collection metadata
            # and persisted doc count by attempting a small search.
            n_docs = 0
            try:
                # Chroma does not expose a count directly via wrapper; do a cheap probe
                probe = self.vectorstore.similarity_search("the", k=1) or []
                # If DB is empty, this will be zero. Otherwise not reliable for exact count.
                # So we just return 0 or >=1 as a hint.
                n_docs = 0 if len(probe) == 0 else 1
            except Exception:
                n_docs = 0
            return {
                "vector_store": {"document_count": n_docs},
                "bm25_documents": 0,  # not tracked here
                "total_queries": 0,   # not tracked here
                "avg_confidence": 0.0,
            }
        except Exception as e:
            logger.error("get_stats failed: %s", e)
            return {
                "vector_store": {"document_count": 0},
                "bm25_documents": 0,
                "total_queries": 0,
                "avg_confidence": 0.0,
            }
