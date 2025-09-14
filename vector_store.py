import os
import logging
from typing import List, Dict, Any

# Silence Chroma telemetry
os.environ.setdefault("CHROMA_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages BM25 and vector-based retrieval using Chroma + HuggingFaceEmbeddings.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None

    def initialize(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] | None = None
    ) -> Dict[str, Any]:
        """
        Initialize Chroma vector store with provided texts and optional metadatas.
        """
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info("Loaded existing Chroma DB")
            else:
                self.vectorstore = Chroma.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas,
                    persist_directory=self.persist_directory
                )
                self.vectorstore.persist()
                logger.info("Created new Chroma DB with %d documents", len(texts))

            return {
                "success": True,
                "document_count": self.vectorstore._collection.count()
            }

        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            return {"success": False, "error": str(e)}

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Run a similarity search using dense embeddings.
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call initialize() first.")

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                for doc, score in results
            ]
        except Exception as e:
            logger.error(f"Error during vector similarity search: {e}")
            return []


# Backward-compat alias
VectorStore = VectorStoreManager
