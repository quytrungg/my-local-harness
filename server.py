#!/usr/bin/env python3
"""
FastMCP server — exposes Obsidian notes to Claude Code via MCP.

Tools:
  query_notes(query)              — semantic search, top 5, verified_success only
  search_knowledge(query, ...)    — broader second-brain search across vault folders
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
DEFAULT_KNOWLEDGE_FOLDERS = ("Projects", "Knowledge", "Logs")
MAX_KNOWLEDGE_RESULTS = 20
MAX_KNOWLEDGE_CANDIDATES = 100

STOP_WORDS = {
    "about", "after", "also", "and", "are", "but", "can", "does", "for",
    "from", "has", "have", "how", "into", "its", "more", "not", "the",
    "this", "that", "was", "what", "when", "where", "which", "with", "you",
}

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


def normalize_folders(folders: str | None) -> set[str]:
    default_folders = {folder.lower() for folder in DEFAULT_KNOWLEDGE_FOLDERS}
    if not folders:
        return default_folders

    normalized = set()
    for raw_folder in folders.split(","):
        folder = raw_folder.strip().strip("/")
        if not folder or "/" in folder or "\\" in folder:
            continue
        normalized.add(folder.lower())

    return normalized or default_folders


def vault_folder_for_path(path: str) -> str:
    if not path:
        return ""

    source = Path(path)
    try:
        relative = source.resolve().relative_to(VAULT_PATH.resolve())
        return relative.parts[0] if relative.parts else ""
    except ValueError:
        parts = source.parts
        return parts[0] if parts else ""


def query_terms(query: str) -> list[str]:
    terms = []
    for term in re.findall(r"[a-zA-Z0-9_][a-zA-Z0-9_.:-]*", query.lower()):
        if len(term) >= 3 and term not in STOP_WORDS:
            terms.append(term)
    return list(dict.fromkeys(terms))


def lexical_boost(query: str, terms: list[str], text: str) -> float:
    haystack = text.lower()
    boost = 0.0

    phrase = query.strip().lower()
    if len(phrase) >= 8 and phrase in haystack:
        boost += 0.12

    if terms:
        hits = sum(1 for term in terms if term in haystack)
        boost += 0.12 * (hits / len(terms))

    return boost


def format_knowledge_results(results: list[dict]) -> str:
    if not results:
        return "No relevant knowledge found in the requested folders."

    parts = []
    for index, result in enumerate(results, 1):
        score = round(result["score"], 3)
        source = result["path"]
        folder = result["folder"]
        doc = result["document"]
        parts.append(
            f"### Result {index} (score: {score}, folder: {folder})\n"
            f"Source: {source}\n\n{doc}"
        )

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
def search_knowledge(
    query: str,
    folders: str = "Projects,Knowledge,Logs",
    top_k: int = 8,
) -> str:
    """
    Search specific knowledge across indexed Obsidian folders.

    This is the broad "second brain" lookup. It searches a larger semantic
    candidate set than query_notes, filters by top-level vault folders, and
    reranks with exact phrase/term matches so specific concepts are easier to
    retrieve from Projects, Knowledge, or Logs.
    """
    collection = get_collection()
    collection_count = collection.count()

    if collection_count == 0:
        return "ChromaDB collection is empty. Run ingest.py first."

    top_k = max(1, min(top_k, MAX_KNOWLEDGE_RESULTS))
    candidate_count = min(
        max(top_k * 6, 30),
        MAX_KNOWLEDGE_CANDIDATES,
        collection_count,
    )
    folder_filter = normalize_folders(folders)
    terms = query_terms(query)
    query_embedding = embed(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    ranked = []
    for doc, meta, distance in zip(docs, metas, distances):
        source = meta.get("path", "unknown")
        folder = vault_folder_for_path(source)
        if folder.lower() not in folder_filter:
            continue

        semantic_score = 1 - distance
        score = semantic_score + lexical_boost(query, terms, f"{source}\n{doc}")
        ranked.append({
            "score": score,
            "folder": folder,
            "path": source,
            "document": doc,
        })

    ranked.sort(key=lambda result: result["score"], reverse=True)
    return format_knowledge_results(ranked[:top_k])


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
