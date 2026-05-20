"""Ingest documents into ChromaDB for RAG retrieval.

Usage:
    python scripts/ingest.py --chroma-host localhost --chroma-port 8100 \
        --collection documents --input-dir ./corpus/

Reads all .txt and .md files from input-dir, splits them into fixed-size chunks
with overlap, embeds them using the same model as rag.py (all-MiniLM-L6-v2),
and upserts into ChromaDB. Safe to re-run — upsert is idempotent.
"""

import argparse
import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest(chroma_host: str, chroma_port: int, collection_name: str, input_dir: Path) -> None:
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    files = sorted(input_dir.glob("*.txt")) + sorted(input_dir.glob("*.md"))
    if not files:
        print(f"No .txt or .md files found in {input_dir}")
        return

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    embeddings_list: list[list[float]] = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{file_path.name}:{i}:{chunk}".encode()).hexdigest()
            embedding = model.encode(chunk).tolist()
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({"source": file_path.name, "chunk_index": i})
            embeddings_list.append(embedding)
        print(f"  {file_path.name}: {len(chunks)} chunk(s)")

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings_list,
    )
    print(f"Ingested {len(ids)} chunks from {len(files)} file(s) into collection '{collection_name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--chroma-host", default="localhost", help="ChromaDB host")
    parser.add_argument("--chroma-port", type=int, default=8100, help="ChromaDB port")
    parser.add_argument("--collection", default="documents", help="Collection name")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with .txt/.md files")
    args = parser.parse_args()

    ingest(args.chroma_host, args.chroma_port, args.collection, args.input_dir)


if __name__ == "__main__":
    main()
