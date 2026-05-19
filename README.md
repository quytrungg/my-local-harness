# 🧠 Harness Memory Pipeline

**A local-first RAG system** that indexes Obsidian notes into ChromaDB, exposes them to Claude Code via MCP, and auto-syncs Claude session summaries back into Obsidian — forming a closed loop of engineering knowledge.

> 📊 Slide deck: [Harness Memory Pipeline — Presentation.html](./Harness%20Memory%20Pipeline%20%E2%80%94%20Presentation.html)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         OBSIDIAN VAULT                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │  Projects/   │  │  Knowledge/ │  │    Logs/    │  │  Daily/ (skip) │  │
│  │  _Brief.md   │  │  Evergreen  │  │  auto-gen   │  │  manual notes  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘  └────────────────┘  │
└─────────┼────────────────┼────────────────┼──────────────────────────────┘
          │                │                │
          ▼                ▼                │
   ┌──────────────────────────────┐         │
   │      ingest.py               │         │
   │  ObsidianLoader → ChromaDB   │         │
   │  nomic-embed-text (Ollama)   │         │
   │  chunk: 1000 / overlap: 100  │         │
   └──────────────┬───────────────┘         │
                  │                         │
                  ▼                         │
   ┌──────────────────────────────┐         │
   │        ChromaDB              │         │
   │  Persistent vector store     │         │
   │  Collection: obsidian_notes  │         │
   │  HNSW cosine distance        │         │
   └──────────────┬───────────────┘         │
                  │ query / context         │
                  ▼                         │
   ┌──────────────────────────────┐         │
   │      server.py (FastMCP)     │         │
   │  query_notes(query) → top 5  │         │
   │  get_project_context(name)   │         │
   │  → _Brief.md                 │         │
   └──────────────┬───────────────┘         │
                  │ MCP protocol            │
                  ▼                         │
   ┌──────────────────────────────┐         │
   │      Claude Code CLI         │         │
   │  Dev sessions with RAG       │         │
   │  context from your notes     │         │
   └──────────────┬───────────────┘         │
                  │ session logs            │
                  ▼                         │
   ┌──────────────────────────────┐         │
   │    sync_sessions.py          │         │
   │  Gatekeeper filter → LLM     ├─────────┘
   │  classify → daily note       │
   └──────────────────────────────┘
```

---

## Components

### 1. `ingest.py` — Obsidian → ChromaDB Indexer

Indexes your Obsidian vault's `Projects/` and `Knowledge/` folders into a local ChromaDB vector store.

**Modes:**

| Mode | Flag | Behaviour |
| --- | --- | --- |
| Incremental | `--incremental` (default) | Skips files unchanged since last run via `.last_ingest` timestamp. Runs weekly after session sync. |
| Full re-index | `--full` | Wipes the `obsidian_notes` collection and rebuilds from scratch. Runs monthly to prevent index drift. |

**Config:**

| Setting | Value |
| --- | --- |
| Embedding model | `nomic-embed-text` (Ollama) |
| Chunk size | 1,000 characters |
| Overlap | 100 characters |
| Batch size | 50 chunks per upsert |
| Collection | `obsidian_notes` (cosine distance, HNSW) |
| Parser | LangChain `ObsidianLoader` |

Every chunk stores file path + YAML tags as metadata. A default `outcome: verified_success` is set for MCP filter compatibility.

---

### 2. `server.py` — FastMCP Server

Exposes two tools to Claude Code via the Model Context Protocol:

**`query_notes(query: str) -> str`**
- Semantic search on ChromaDB
- Returns top 5 matching chunks filtered to `outcome = verified_success`
- Converts cosine distance to a 0–1 relevance score

**`get_project_context(project_name: str) -> str`**
- Reads `Projects/<name>/_Brief.md` from the vault
- Case-insensitive project name fallback
- Lists available projects if not found

**Lifecycle:** Registered in `~/.claude/settings.json` as an MCP server. Launched at macOS login via a **launchd plist** (persistent daemon).

---

### 3. `sync_sessions.py` — The Gatekeeper

Reads Claude Code session logs from the last 7 days (`~/.claude/projects/*.jsonl`), filters through a 3-stage pipeline, and writes aggregated daily notes back to Obsidian.

#### Stage 1 — Heuristic Filter (Structural)

| Check | Trigger | Result |
| --- | --- | --- |
| No tool calls | Zero `tool_use` blocks (including `Agent` sub-agent calls) | **TRASH** |
| Loop detection | Same Bash command repeated >10x consecutively | **TRASH** |
| Git delta | `git diff --name-only` on session `cwd` is empty | **RESEARCH** |

#### Stage 2 — Outcome Signal

- **Exit codes:** Non-zero `attachment.exitCode` on the final hook → flagged as failure
- **Keywords:** Last user message contains positive signals (`"that worked"`, `"thanks"`, `"done"`, `"merged"`, etc.) → boost to KNOWLEDGE

#### Stage 3 — LLM Classifier

Sends the final 20% of the session transcript to `llama3:8b` via Ollama:

> Bug fixes, architectural decisions, non-obvious findings → **KNOWLEDGE**
> Failure, confusion, trivial chat → **TRASH**

#### Routing

| Classification | Destination | ChromaDB | Retention |
| --- | --- | --- | --- |
| **KNOWLEDGE** | `Logs/YYYY-MM-DD.md` | ✅ Indexed — high priority | Permanent |
| **RESEARCH** | `System/Staging/` | ⏸️ Skipped for now | 14-day pruning (planned) |
| **TRASH** | `~/.claude/archive/` | ❌ Not indexed | Archived |

#### Output Format

One Obsidian note per calendar day, aggregating all sessions across all repos:

```yaml
---
session_ids: ["abc-123", "def-456"]
date: "2026-05-16"
outcome: "verified_success"
repos: [employment-hero, rostering-api-service]
tags: [auto-logged]
---
```

The body is generated by `llama3:8b` via a structured summarization prompt that extracts:
- What was worked on
- Technical decisions with reasoning
- Gotchas / surprises
- Key commands and patterns
- Pending tasks

> **Note:** `utility_score` is computed at query time by ChromaDB's BM25 scorer — it is **not** stored in YAML front-matter.

---

## Automation Schedule

Three cronjobs keep the loop running:

```
30 23 * * 6   python sync_sessions.py          # Saturday 23:30
0  0  * * 0   python ingest.py --incremental   # Sunday 00:00
0  1  1 * *   python ingest.py --full          # 1st of month 01:00
```

| Time | Script | Purpose |
| --- | --- | --- |
| Saturday 23:30 | `sync_sessions.py` | Scan last 7 days of Claude sessions, filter, classify, write daily notes |
| Sunday 00:00 | `ingest.py --incremental` | Index only files modified since last run (picks up new session notes) |
| 1st of month 01:00 | `ingest.py --full` | Wipe and rebuild entire collection (prevents index drift) |

Both `nomic-embed-text` and `llama3:8b` are already pulled via Ollama.

---

## Repository Structure

```
my-local-harness/
├── ingest.py                                    # Obsidian → ChromaDB indexer
├── server.py                                    # FastMCP server (MCP tools)
├── sync_sessions.py                             # Claude session → Obsidian note pipeline
├── chroma_db/                                   # ChromaDB persistent storage (gitignored)
├── venv/                                        # Python virtual environment
├── .last_ingest                                 # Timestamp file for incremental mode
├── Pipeline Plan.md                             # Full engineering blueprint
├── CLAUDE.md                                    # Agent context file
├── README.md                                    # This file
└── Harness Memory Pipeline — Presentation.html  # Slide deck
```

### Obsidian Vault Structure

```
Trung's Brain/
├── Projects/        # Active engineering notes (one subfolder per repo)
│   └── <repo>/
│       └── _Brief.md          # High-level summary for RAG
├── Knowledge/       # Permanent evergreen technical notes
├── Logs/            # Auto-generated daily session notes (from sync_sessions.py)
├── Daily/           # Manual daily logs (excluded from ingest)
└── System/
    ├── Pipeline Plan.md
    └── Staging/     # RESEARCH tier (skipped for now)
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) with models pulled:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3:8b
  ```

### Installation

```bash
cd my-local-harness
python3 -m venv venv
source venv/bin/activate
pip install chromadb langchain-community fastmcp ollama
```

### First Run

```bash
# Full ingest to populate ChromaDB
python ingest.py --full

# Verify MCP server starts
python server.py

# Test session sync (dry run)
python sync_sessions.py
```

### MCP Server Registration

The server should be registered in `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "harness-memory": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

For persistent background operation, a launchd plist is configured to start the server at macOS login.

---

## Key Design Decisions

| Decision | Rationale |
| --- | --- |
| **ChromaDB embedded** — no Docker | Single process, simpler deploy, persistent on disk |
| **`nomic-embed-text` via Ollama** | Local embeddings, no API key, lightweight model |
| **`llama3:8b` for classification** | Fast enough for batch processing, good-enough accuracy |
| **Heuristic filter before LLM** | LLM call is expensive; cheap heuristics catch obvious trash first |
| **One note per day** | Avoids note explosion; sessions aggregate naturally |
| **`utility_score` computed at query time** | BM25 relevance varies by query; storing a static score is misleading |
| **Sub-agent `Agent` calls count as tool activity** | Prevents valid multi-agent sessions from being falsely classified as trash |

---

## Future Enhancements

- 🔗 **Graph-Aware RAG** — Prioritize chunks from notes with the most Obsidian backlinks
- 🖼️ **Image Support** — Use llava to describe schematics and diagrams for the vector DB
- 🏷️ **Auto-Tagging** — Apply `#completed` / `#todo` tags based on session outcome
- 🧹 **14-day RESEARCH Pruning** — Delete stale `System/Staging/` notes older than 14 days

---

## What NOT to do

- Do not run `ingest.py` against `Daily/`, `Logs/`, or `System/` — noise, not signal.
- Do not store `utility_score` in YAML front-matter — it's a query-time BM25 value.
- Do not skip the heuristic filter and go straight to the LLM classifier — expensive.
- Do not `git checkout` or `git switch` in this repo — it is a worktree environment.

---

Made with 🧠 by Trung — slide deck available at [Harness Memory Pipeline — Presentation.html](./Harness%20Memory%20Pipeline%20%E2%80%94%20Presentation.html)
