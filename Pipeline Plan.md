This is your comprehensive engineering blueprint for the **"Harness Memory" Pipeline**. This plan integrates your local knowledge graph (Obsidian) with semantic search (ChromaDB) and automated feedback loops (Claude Session Summaries).

---

# Project Blueprint: The Harness Memory Pipeline

**Version:** 1.1
**Target Architecture:** Local-first RAG + MCP Integration
**Date:** May 2026

## 1. System Architecture Overview

The goal is to create a bidirectional flow of information:

1. **Obsidian → ChromaDB:** Semantic indexing of your engineering notes.
2. **Claude Code → ChromaDB:** Accessing those notes via MCP during dev sessions.
3. **Claude Sessions → Obsidian:** Automated weekly cronjob to capture and summarize session logs.

---

## 2. Phase 1: The Vector Infrastructure (ChromaDB)

Using **Ollama** for embeddings and **ChromaDB** as an embedded Python library (no Docker).

### Components:

- **Vector DB:** ChromaDB (embedded Python library, persistent on disk).
- **Embedding Model:** `nomic-embed-text` (via Ollama).
- **Ingestion Script (`ingest.py`):**
  - **Parser:** Use `LangChain`'s `ObsidianLoader`.
  - **Scope:** `Projects/` and `Knowledge/` folders only.
  - **Chunking:** Recursive character splitting (Chunk: 1000, Overlap: 100).
  - **Metadata Enrichment:** Map the Obsidian file path and YAML tags to every chunk.

### Ingest Schedule (two modes):

| Mode | Trigger | When |
| --- | --- | --- |
| **Incremental** | Runs after `sync_sessions.py` completes | Every weekend (crontab) |
| **Full re-index** | Wipes and rebuilds entire collection | Monthly (crontab) |

---

## 3. Phase 2: The MCP Integration (The "Harness")

To make Claude Code "see" your notes, we need a local **Model Context Protocol** server.

### Tools to Implement:

- `query_notes(query)`: Semantic search on ChromaDB, returns top 5 relevant markdown snippets. Default filter: `outcome = 'verified_success'` only.
- `get_project_context(project_name)`: Fetches the `_Brief.md` or high-level status of a project.

### Implementation Detail:

Use the **FastMCP (Python)** framework.

```python
@mcp.tool()
def query_notes(query: str) -> str:
    results = chroma_collection.query(
        query_texts=[query],
        n_results=5,
        where={"outcome": "verified_success"}
    )
    return format_results_for_claude(results)
```

### Registration:

- Registered in Claude Code CLI config at `~/.claude/`
- Server launched at **macOS login** via a launchd plist (persistent daemon)

---

## 4. Phase 3: The "Closing Loop" (Cronjob & Summarizer)

Captures what was done in Claude Code sessions and writes it back to Obsidian.

### Session Log Format (confirmed):

- Location: `~/.claude/projects/<project-dir>/<session-id>.jsonl`
- Each line is a typed JSON object. Key types:
  - `type=user` → user messages (`message.content` is a string)
  - `type=assistant` → assistant turns; tool calls live in `message.content[]` as `{type: "tool_use", name: "Bash"|"Read"|"Agent"|...}`
  - `type=attachment` → hook results; exit codes in `attachment.exitCode` (only available here, not in Bash stdout)
- Repo association: `cwd` field on every line (e.g. `/Users/trungmai/Developer/employment-hero`)

### The Cronjob Script (`sync_sessions.py`):

Runs **every weekend** via crontab. Processes sessions modified in the last 7 days.

#### Step 1 — Heuristic Filter (Structural Check)

| Check | Logic | Result if triggered |
| --- | --- | --- |
| No tool calls | Zero `tool_use` blocks in all assistant messages (including `Agent` calls) | TRASH |
| Loop detection | Same Bash command repeated >10 times consecutively | TRASH |
| Git delta | `git diff` on session `cwd` shows zero file changes | RESEARCH |

#### Step 2 — Outcome Signal (Success Verification)

- **Exit codes:** Scan `type=attachment` lines for `attachment.exitCode != 0` on the final hook call → flag as potential failure
- **Keyword scan:** Check last user message text for positive signals: "that worked", "thanks", "perfect", "done", "merged"

#### Step 3 — LLM Classifier (Semantic Check)

Send the final 20% of the session transcript (user + assistant text only) to `llama3:8b`:

> "Determine if this technical session concluded with a solution or a useful discovery.
> - If it ended in failure, confusion, or was a trivial task: Output 'TRASH'.
> - If it contains a bug fix, an architectural decision, or a project decision: Output 'KNOWLEDGE'.
> - Response must be a single word."

#### Step 4 — Routing

| Classification | Destination | ChromaDB |
| --- | --- | --- |
| **KNOWLEDGE** | `Logs/YYYY-MM-DD.md` (aggregated per day) | Yes — high priority |
| **RESEARCH** | `System/Staging/` | Skipped for now |
| **TRASH** | `~/.claude/archive/` | No |

#### Step 5 — Output Format

One Obsidian note per calendar day, aggregating all sessions across all repos. Summarization via Ollama `llama3:8b`: extract technical decisions, commands used, pending tasks, and format with Obsidian backlinks.

Every note includes a YAML quality header:

```yaml
---
session_ids: ["abc-123", "def-456"]
date: "2026-05-16"
outcome: "verified_success"
repos: [employment-hero, rostering-api-service]
tags: [auto-logged]
---
```

> Note: `utility_score` is computed at query time by ChromaDB's BM25 scorer — it is not a stored field.

---

## 5. Phase 4: Vault Structure

All folders already exist at `/Users/trungmai/Trung's Brain/`:

```
Trung's Brain/
├── Workflows/       # Landing zone / automation scripts
├── Daily/           # Manual daily logs
├── Projects/        # Active engineering (one subfolder per repo)
│   └── <repo>/
│       └── _Brief.md   # High-level summary for RAG
├── Knowledge/       # Permanent technical notes (Evergreen)
├── Logs/            # Output from sync_sessions.py (KNOWLEDGE tier)
└── System/
    ├── Pipeline Plan.md
    └── Staging/     # RESEARCH tier (skipped for now)
```

---

## 6. Implementation Checklist

### [x] Decisions resolved
- ChromaDB: embedded library (no Docker)
- Vault path: `/Users/trungmai/Trung's Brain/`
- MCP target: Claude Code CLI (`~/.claude/`)
- MCP server lifecycle: launchd plist (launch at login)
- Ingest scope: `Projects/` + `Knowledge/` only
- Ingest schedule: incremental weekly, full re-index monthly
- Output granularity: one note per day (aggregated across sessions + repos)
- `utility_score`: BM25 score computed at query time, not stored
- Sub-agent `Agent` tool calls count as tool activity for heuristic filter
- Exit codes only reliably available in `attachment.exitCode` (hook results), not Bash stdout

### [ ] Step 1: Environment Setup
- `ollama pull nomic-embed-text` ✓ (already done)
- `ollama pull llama3:8b` ✓ (already done)
- Create Python venv in this repo
- `pip install chromadb langchain-community fastmcp ollama`

### [ ] Step 2: ingest.py
- Crawl `Projects/` and `Knowledge/` with `ObsidianLoader`
- Chunk: 1000 chars, overlap: 100
- Store file path + YAML tags as chunk metadata
- Incremental mode: only process files newer than last run timestamp
- Full mode: wipe collection and reindex everything
- Verify collection size after first run

### [ ] Step 3: server.py (FastMCP)
- `query_notes(query)` — semantic search, top 5, filter `outcome=verified_success`
- `get_project_context(project_name)` — read `Projects/<name>/_Brief.md`
- Register in `~/.claude/settings.json` as MCP server
- Write launchd plist to launch at login

### [ ] Step 4: sync_sessions.py
- Locate sessions modified in last 7 days
- Heuristic filter (tool calls, loop detection, git delta)
- Outcome signal (exit codes from attachments, keyword scan)
- LLM classifier via `llama3:8b`
- Summarize KNOWLEDGE sessions via `llama3:8b`
- Write aggregated daily note to `Logs/YYYY-MM-DD.md`
- Move TRASH sessions to `~/.claude/archive/`

### [ ] Step 5: Crontab
```
# Incremental ingest — every Sunday at midnight (after sync_sessions finishes)
0 0 * * 0 /path/to/venv/bin/python /path/to/ingest.py --incremental

# Full re-index — 1st of every month at 01:00
0 1 1 * * /path/to/venv/bin/python /path/to/ingest.py --full

# Session sync — every Saturday at 23:30
30 23 * * 6 /path/to/venv/bin/python /path/to/sync_sessions.py
```

---

## 7. Future Enhancements (Harness Engineering 2.0)

- **Graph-Aware RAG:** Prioritize chunks from notes with the most Obsidian backlinks.
- **Image Support:** Use `llava` to describe schematics/photos for the vector DB.
- **Auto-Tagging:** Cronjob applies `#completed` / `#todo` tags based on session outcome.
- **14-day RESEARCH pruning:** Second crontab to delete stale `System/Staging/` notes.

---

## 8. Intelligence Layer: Signal-to-Noise Filtering

Full spec is captured in Phase 3 above. Summary:

1. **Fetch** sessions from `~/.claude/projects/` modified in last 7 days
2. **Filter** via heuristic (tool calls, git delta, loop detection)
3. **Classify** via Ollama `llama3:8b` semantic check
4. **Format** into Markdown with YAML front-matter
5. **Write** to `Logs/YYYY-MM-DD.md` (one file per day, all repos aggregated)
