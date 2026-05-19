#!/usr/bin/env python3
"""
Obsidian → ChromaDB indexer.

Modes:
  --incremental  Only index files newer than last run (default)
  --full         Wipe collection and reindex everything
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import chromadb
import ollama
from langchain_community.document_loaders import ObsidianLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Environment loading ─────────────────────────────────────────────────────
def _load_env() -> dict:
    """Load .env from the same directory as this script (if it exists)."""
    env_file = Path(__file__).parent / ".env"
    env = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^([A-Za-z_]\w*)=(.*)$', line)
            if m:
                val = m.group(2).strip().strip('"').strip("'")
                env[m.group(1)] = val
    return env


_env = _load_env()
_PROJECT_DIR = _env.get("HARNESS_PROJECT_DIR", str(Path(__file__).parent))
# ─────────────────────────────────────────────────────────────────────────────


VAULT_PATH = Path(_env.get("OBSIDIAN_VAULT_PATH", str(Path.home() / "Trung's Brain")))
INGEST_DIRS = [s.strip() for s in _env.get("INGEST_DIRS", "Projects,Knowledge").split(",")]
CHROMA_PATH = Path(_env.get("CHROMA_DB_DIR", str(Path(__file__).parent / "chroma_db")))
if not CHROMA_PATH.is_absolute():
    CHROMA_PATH = Path(_PROJECT_DIR) / CHROMA_PATH
COLLECTION_NAME = _env.get("COLLECTION_NAME", "obsidian_notes")
TIMESTAMP_FILE = Path(__file__).parent / ".last_ingest"

CHUNK_SIZE = int(_env.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_env.get("CHUNK_OVERLAP", "100"))
EMBED_MODEL = _env.get("EMBED_MODEL", "nomic-embed-text")


def get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed(texts: list[str]) -> list[list[float]]:
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return response.embeddings


def load_documents(since: float | None = None) -> list:
    docs = []
    for folder in INGEST_DIRS:
        target = VAULT_PATH / folder
        if not target.exists():
            print(f"  [skip] {target} does not exist")
            continue
        loader = ObsidianLoader(str(target), collect_metadata=True)
        for doc in loader.load():
            if since is not None:
                mtime = Path(doc.metadata.get("path", "")).stat().st_mtime if doc.metadata.get("path") else 0
                if mtime <= since:
                    continue
            docs.append(doc)
    return docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_chunk_id(chunk, index: int) -> str:
    path = chunk.metadata.get("path", "unknown")
    return f"{path}::{index}"


def index_chunks(collection: chromadb.Collection, chunks: list) -> int:
    if not chunks:
        return 0

    batch_size = 50
    total = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]
        embeddings = embed(texts)

        ids = [build_chunk_id(c, i + j) for j, c in enumerate(batch)]
        metadatas = []
        for c in batch:
            meta = {k: str(v) for k, v in c.metadata.items()}
            # Ensure outcome field exists so MCP where-filter works
            meta.setdefault("outcome", "verified_success")
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total += len(batch)
        print(f"  indexed {total}/{len(chunks)} chunks", end="\r")

    print()
    return total


def run_incremental(client: chromadb.PersistentClient) -> None:
    since = None
    if TIMESTAMP_FILE.exists():
        since = float(TIMESTAMP_FILE.read_text().strip())
        print(f"Incremental: indexing files modified after {time.ctime(since)}")
    else:
        print("Incremental: no prior timestamp found — indexing all files")

    collection = get_collection(client)
    docs = load_documents(since=since)
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    indexed = index_chunks(collection, chunks)
    TIMESTAMP_FILE.write_text(str(time.time()))
    print(f"Done. Indexed {indexed} chunks. Collection size: {collection.count()}")


def run_full(client: chromadb.PersistentClient) -> None:
    print("Full re-index: wiping collection...")
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = get_collection(client)
    docs = load_documents(since=None)
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    indexed = index_chunks(collection, chunks)
    TIMESTAMP_FILE.write_text(str(time.time()))
    print(f"Done. Indexed {indexed} chunks. Collection size: {collection.count()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--incremental", action="store_true", default=True)
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()

    CHROMA_PATH.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if args.full:
        run_full(client)
    else:
        run_incremental(client)


if __name__ == "__main__":
    main()
