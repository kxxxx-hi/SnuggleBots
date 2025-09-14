# vector_store.py
import sys, os, logging, shutil
from typing import List, Dict, Any, Optional

# --- SQLite shim MUST run before importing chromadb ---
try:
    import sqlite3
    ver = tuple(map(int, sqlite3.sqlite_version.split(".")))
    if ver < (3, 35, 0):
        import pysqlite3 as _pysqlite3
        sys.modules["sqlite3"] = _pysqlite3  # monkey-patch
except Exception:
    try:
        import pysqlite3 as _pysqlite3
        sys.modules["sqlite3"] = _pysqlite3
    except Exception:
        pass

import chromadb
from chromadb.config import Settings

# LangChain bits
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "pet_docs"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("Using SentenceTransformer embeddings")

        self.collection_name = collection_name
        self._init_chroma()

    def _init_chroma(self):
        """Init Chroma; if an old schema causes errors, wipe and retry once."""
        tried_reset = False
        while True:
            try:
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                    client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
                )
                # touch DB to ensure it’s valid
                _ = self.vectorstore._collection.count()
                break
            except Exception as e:
                msg = str(e)
                logger.error("Error initializing Chroma: %s", msg)
                if (not tried_reset) and (
                    "no such column" in msg.lower()
                    or "schema" in msg.lower()
                    or "migration" in msg.lower()
                ):
                    tried_reset = True
                    shutil.rmtree(self.persist_dir, ignore_errors=True)
                    os.makedirs(self.persist_dir, exist_ok=True)
                    logger.warning("Chroma DB reset due to schema mismatch; retrying…")
                    continue
                raise

    # minimal API your app expects:
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
        return self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def similarity_search(self, query: str, k: int = 5):
        return self.vectorstore.similarity_search(query, k=k)

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        kwargs = search_kwargs or {"k": 5}
        return self.vectorstore.as_retriever(search_kwargs=kwargs)

    def persist(self):
        try:
            self.vectorstore.persist()
        except Exception:
            pass
