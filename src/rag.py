"""RAG (Retrieval-Augmented Generation) module for the LLMOps pipeline.

Handles embedding, retrieval from ChromaDB, and prompt augmentation.
Imported and called from llm_wrapper.py — not a standalone service.
Metrics and spans integrate with the existing Prometheus + OpenTelemetry stack.
"""

import logging
import time

from opentelemetry import trace
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

RAG_RETRIEVAL_DURATION = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval latency (embed + search)",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
RAG_CHUNKS_RETRIEVED = Histogram(
    "rag_chunks_retrieved",
    "Number of chunks returned per query",
    buckets=[0, 1, 2, 3, 5, 10],
)
RAG_RETRIEVAL_ERRORS = Counter(
    "rag_retrieval_errors_total",
    "Retrieval failures (Chroma unreachable, embedding errors)",
)
RAG_CONTEXT_TOKENS_ESTIMATE = Histogram(
    "rag_context_tokens_estimate",
    "Estimated token count of injected context",
    buckets=[50, 100, 250, 500, 1000, 2000, 4000],
)

_tracer = trace.get_tracer("rag")


class RAGConfig:
    """RAG configuration loaded from environment variables."""

    def __init__(self) -> None:
        import os

        raw_enabled = os.getenv("RAG_ENABLED", "false").strip().lower()
        self.rag_enabled: bool = raw_enabled in ("true", "1", "yes")
        self.chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
        self.collection: str = os.getenv("CHROMA_COLLECTION", "documents")
        self.top_k: int = int(os.getenv("RAG_TOP_K", "5"))
        self.embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.min_similarity: float = float(os.getenv("RAG_MIN_SIMILARITY", "0.3"))


class RAGService:
    """Handles embedding, retrieval from ChromaDB, and prompt augmentation.

    Initialised once at startup. Holds a SentenceTransformer model and ChromaDB client.
    All retrieval failures are caught and logged; the wrapper request never fails because
    of RAG being unavailable.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._collection = None
        self._model = None
        self._init_model()
        self._init_chroma()

    def _init_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._config.embedding_model)
        logger.info("Loaded embedding model: %s", self._config.embedding_model)

    def _init_chroma(self) -> None:
        try:
            import chromadb

            client = chromadb.HttpClient(
                host=self._config.chroma_host,
                port=self._config.chroma_port,
            )
            self._collection = client.get_or_create_collection(
                self._config.collection,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB connected at %s:%d, collection=%s",
                self._config.chroma_host,
                self._config.chroma_port,
                self._config.collection,
            )
        except Exception as exc:
            logger.warning(
                "ChromaDB unreachable at %s:%d — RAG retrieval disabled: %s",
                self._config.chroma_host,
                self._config.chroma_port,
                exc,
            )
            self._collection = None

    def retrieve(self, query: str) -> list[dict]:
        """Embed query, search Chroma, return top_k chunks filtered by min_similarity.

        Each returned dict has keys: text, metadata, similarity.
        Returns empty list on any error (Chroma down, embedding failure, etc.).
        """
        with _tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("rag.query_length", len(query))
            span.set_attribute("rag.collection", self._config.collection)

            if self._collection is None:
                self._init_chroma()

            if self._collection is None:
                RAG_RETRIEVAL_ERRORS.inc()
                span.set_attribute("rag.chunks_returned", 0)
                span.set_attribute("rag.top_similarity", 0.0)
                return []

            try:
                start = time.monotonic()
                embedding = self._model.encode(query).tolist()
                results = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=self._config.top_k,
                    include=["documents", "metadatas", "distances"],
                )
                duration = time.monotonic() - start
                RAG_RETRIEVAL_DURATION.observe(duration)

                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]

                chunks = []
                for doc, meta, dist in zip(documents, metadatas, distances):
                    # Cosine space: distance in [0, 2]; similarity = 1 - distance
                    similarity = 1.0 - dist
                    if similarity >= self._config.min_similarity:
                        chunks.append({"text": doc, "metadata": meta, "similarity": similarity})

                top_similarity = max((c["similarity"] for c in chunks), default=0.0)
                span.set_attribute("rag.chunks_returned", len(chunks))
                span.set_attribute("rag.top_similarity", top_similarity)
                RAG_CHUNKS_RETRIEVED.observe(len(chunks))
                return chunks

            except Exception as exc:
                logger.warning("RAG retrieval error: %s", exc)
                RAG_RETRIEVAL_ERRORS.inc()
                span.set_attribute("rag.chunks_returned", 0)
                span.set_attribute("rag.top_similarity", 0.0)
                return []

    def augment_messages(self, messages: list[dict], chunks: list[dict]) -> list[dict]:
        """Prepend retrieved context as a system message before the conversation.

        Returns messages unchanged if no chunks were retrieved.
        """
        if not chunks:
            return messages

        context_parts = [f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks)]
        context_text = "\n\n".join(context_parts)
        system_content = (
            "Use the following retrieved context to answer the user's question:\n\n"
            + context_text
        )

        RAG_CONTEXT_TOKENS_ESTIMATE.observe(len(system_content) // 4)

        augmented = list(messages)
        if augmented and augmented[0].get("role") == "system":
            augmented[0] = {
                "role": "system",
                "content": system_content + "\n\n" + augmented[0]["content"],
            }
        else:
            augmented.insert(0, {"role": "system", "content": system_content})

        return augmented

    def augment_prompt(self, prompt: str, chunks: list[dict]) -> str:
        """Prepend retrieved context to a prompt string (for /api/generate).

        Returns prompt unchanged if no chunks were retrieved.
        """
        if not chunks:
            return prompt

        context_parts = [f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks)]
        context_text = "\n\n".join(context_parts)
        augmented = f"Context:\n{context_text}\n\nQuestion: {prompt}"

        RAG_CONTEXT_TOKENS_ESTIMATE.observe(len(augmented) // 4)
        return augmented
