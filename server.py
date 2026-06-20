#!/usr/bin/env python3
"""
FastMCP server — exposes Obsidian notes to Claude Code via MCP.

Tools:
  query_notes(query)              — semantic search, top 5, verified_success only
  get_project_context(project)    — reads Projects/<name>/_Brief.md
"""

import os
import re
from pathlib import Path

import chromadb
import ollama
from fastmcp import FastMCP


# ── Environment loading ─────────────────────────────────────────────────────
def _load_env() -> dict:
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
CHROMA_PATH = Path(_env.get("CHROMA_DB_DIR", str(Path(__file__).parent / "chroma_db")))
if not CHROMA_PATH.is_absolute():
    CHROMA_PATH = Path(_PROJECT_DIR) / CHROMA_PATH
COLLECTION_NAME = _env.get("COLLECTION_NAME", "obsidian_notes")
EMBED_MODEL = _env.get("EMBED_MODEL", "nomic-embed-text")

mcp = FastMCP("harness-memory")
_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = _client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def embed(text: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=[text])
    return response.embeddings[0]


def format_results(results: dict) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return "No relevant notes found."

    parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        source = meta.get("path", "unknown")
        score = round(1 - dist, 3)
        parts.append(f"### Result {i} (score: {score})\nSource: {source}\n\n{doc}")

    return "\n\n---\n\n".join(parts)


@mcp.tool()
def query_notes(query: str) -> str:
    """Search your Obsidian knowledge base for notes relevant to the query."""
    collection = get_collection()

    if collection.count() == 0:
        return "ChromaDB collection is empty. Run ingest.py first."

    query_embedding = embed(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"outcome": "verified_success"},
    )

    return format_results(results)


@mcp.tool()
def get_project_context(project_name: str) -> str:
    """Return the _Brief.md for a project from the Obsidian vault."""
    brief = VAULT_PATH / "Projects" / project_name / "_Brief.md"

    if not brief.exists():
        # Try case-insensitive match
        projects_dir = VAULT_PATH / "Projects"
        matches = [
            p for p in projects_dir.iterdir()
            if p.is_dir() and p.name.lower() == project_name.lower()
        ]
        if matches:
            brief = matches[0] / "_Brief.md"

    if not brief.exists():
        available = [p.name for p in (VAULT_PATH / "Projects").iterdir() if p.is_dir()]
        return f"No _Brief.md found for '{project_name}'. Available projects: {', '.join(sorted(available))}"

    return brief.read_text()


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8123
    kwargs = {}
    if transport == "sse":
        kwargs["port"] = port
    mcp.run(transport=transport, **kwargs)
