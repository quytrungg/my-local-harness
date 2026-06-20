#!/usr/bin/env python3
"""
Claude/Codex session → Obsidian note pipeline.

Reads Claude sessions from ~/.claude/projects/ and Codex sessions from
~/.codex/sessions/ for sessions modified in the last 7 days, filters them
through the gatekeeper, writes verified work to Obsidian vault Logs/, and
writes research sessions to Obsidian vault Knowledge/.
"""

import json
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import ollama


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
# ─────────────────────────────────────────────────────────────────────────────


CLAUDE_PROJECTS = Path(_env.get("CLAUDE_PROJECTS_DIR", str(Path.home() / ".claude" / "projects")))
CODEX_SESSIONS = Path(_env.get("CODEX_SESSIONS_DIR", str(Path.home() / ".codex" / "sessions")))
VAULT_ROOT = Path(_env.get("OBSIDIAN_VAULT_PATH", str(Path.home() / "Trung's Brain")))
VAULT_LOGS = Path(_env.get("OBSIDIAN_LOGS_DIR", str(VAULT_ROOT / "Logs")))
VAULT_KNOWLEDGE = Path(_env.get("OBSIDIAN_KNOWLEDGE_DIR", str(VAULT_ROOT / "Knowledge")))
VAULT_PROMPTS = Path(_env.get("OBSIDIAN_PROMPTS_DIR", str(VAULT_ROOT / "Prompts")))
VAULT_PROJECTS = Path(_env.get("OBSIDIAN_PROJECTS_DIR", str(VAULT_ROOT / "Projects")))
CLAUDE_ARCHIVE_DIR = Path(_env.get("CLAUDE_ARCHIVE_DIR", str(Path.home() / ".claude" / "archive")))
CODEX_ARCHIVE_DIR = Path(_env.get("CODEX_ARCHIVE_DIR", str(Path.home() / ".codex" / "archived_sessions")))
LOOKBACK_DAYS = int(_env.get("LOOKBACK_DAYS", "7"))
CLASSIFY_MODEL = _env.get("CLASSIFY_MODEL", "llama3:8b")

POSITIVE_KEYWORDS = {"that worked", "thanks", "thank you", "perfect", "done", "merged", "lgtm", "nice", "great"}
LOOP_THRESHOLD = 10
SHELL_TOOL_NAMES = {"bash", "exec_command", "shell"}
COMMAND_KEYS = ("command", "cmd")
RELATED_NOTE_DIRS = [
    s.strip()
    for s in _env.get("RELATED_NOTE_DIRS", "Projects,Knowledge,Prompts").split(",")
    if s.strip()
]
RELATED_NOTE_LIMIT = int(_env.get("RELATED_NOTE_LIMIT", "5"))
RELATED_NOTE_MIN_SCORE = int(_env.get("RELATED_NOTE_MIN_SCORE", "8"))
PROMPT_MESSAGE_MIN_SCORE = int(_env.get("PROMPT_MESSAGE_MIN_SCORE", "6"))
PROMPT_SESSION_MIN_SCORE = int(_env.get("PROMPT_SESSION_MIN_SCORE", "8"))
GENERATED_NOTE_PREFIXES = ("Research Sessions - ", "Session Notes - ", "Prompt Patterns - ")
TOKEN_STOPWORDS = {
    "about", "after", "agent", "also", "assistant", "because", "before", "being",
    "brief", "called", "check", "code", "codex", "command", "commands", "could",
    "done", "during", "existing", "failed", "files", "from", "have", "into",
    "issue", "jsonl", "knowledge", "logic", "notes", "output", "pending",
    "research", "repo", "session", "sessions", "should", "source", "summary",
    "technical", "that", "their", "there", "these", "thing", "this", "through",
    "tool", "tools", "unknown", "updated", "used", "user", "using", "with",
    "worked", "would", "write",
}

PROMPT_ACTION_WORDS = {
    "analyze", "check", "compare", "debug", "design", "diagnose", "explain",
    "find", "fix", "investigate", "plan", "review", "summarize", "trace",
    "validate", "verify",
}
PROMPT_FEEDBACK_PHRASES = (
    "actually", "but", "don't", "instead", "not ", "please investigate more",
    "please re-check", "rather than", "that's not", "try", "you missed",
)
PROMPT_CONTEXT_PATTERNS = (
    r"`[^`]+`",
    r"\b[A-Z]{2,10}-\d+\b",
    r"\b\w+/\w+[/\w.-]*\b",
    r"\b\w+\.(rb|py|ts|tsx|js|jsx|md|json|yml|yaml|sql)\b",
    r"\b(error|exception|failed|failing|stack trace|payload|request|response|log|query|sql)\b",
)

SESSION_SOURCES = {
    "claude": {
        "root": CLAUDE_PROJECTS,
        "archive_dir": CLAUDE_ARCHIVE_DIR,
    },
    "codex": {
        "root": CODEX_SESSIONS,
        "archive_dir": CODEX_ARCHIVE_DIR,
    },
}

CLASSIFY_PROMPT = """Determine if this technical session concluded with a solution or a useful discovery.
- If it ended in failure, confusion, or was a trivial task (general chat, simple lookups): Output 'TRASH'.
- If it contains a bug fix, an architectural decision, a project decision, or a non-obvious technical finding: Output 'KNOWLEDGE'.
Response must be a single word."""

SUMMARIZE_PROMPT = """You are summarizing a Claude Code or Codex engineering session for an Obsidian knowledge base.

Extract and format as Markdown:
1. **What was worked on** — repo, ticket, brief description
2. **Technical decisions made** — non-obvious choices and the reasoning
3. **Gotchas / surprises** — things that weren't obvious, bugs discovered
4. **Commands / patterns used** — key bash commands, code patterns worth remembering
5. **Pending tasks** — anything left unresolved

Be concise. Omit small talk, filler, raw session IDs, and session inventories.
Related [[wikilinks]] are added separately by the pipeline."""

PROMPT_PATTERNS_PROMPT = """You are distilling durable AI prompting patterns from engineering sessions.

Extract and format as Markdown:
1. **Effective prompting patterns** — the reusable user prompt shapes that led to good investigation or feedback
2. **Why they worked** — what context, constraints, evidence, or correction made them effective
3. **Reusable templates** — short prompt templates someone can adapt later
4. **Feedback moves** — useful ways the user corrected, redirected, or narrowed the agent

Do not include raw session IDs or session inventories. Prefer concise examples over long verbatim pasted prompts."""

PROJECT_NOTE_PROMPT = """You are summarizing engineering session material into a project-specific Obsidian note.

Extract and format as Markdown:
1. **Project context** — what part of this repo/system was explored
2. **Durable findings** — facts, implementation details, constraints, or behavior worth remembering for this project
3. **Patterns / commands** — commands, code paths, queries, or workflows likely to be reused
4. **Open threads** — follow-up work or unresolved questions

Keep the note project-specific. Omit raw session IDs, generic process chatter, and session inventories."""


def find_recent_sessions(lookback_days: int) -> list[tuple[str, Path]]:
    cutoff = time.time() - lookback_days * 86400
    sessions = []
    for source, config in SESSION_SOURCES.items():
        root = config["root"]
        if not root.exists():
            continue
        for jsonl in root.rglob("*.jsonl"):
            if jsonl.stat().st_mtime >= cutoff:
                sessions.append((source, jsonl))
    return sessions


def extract_text_blocks(content, allowed_types: set[str] | None = None) -> list[str]:
    texts = []
    if isinstance(content, str):
        if content.strip():
            texts.append(content.strip())
        return texts

    if not isinstance(content, list):
        return texts

    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if allowed_types and block_type not in allowed_types:
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def parse_tool_input(raw_input) -> dict:
    if isinstance(raw_input, dict):
        return raw_input
    if isinstance(raw_input, str):
        try:
            parsed = json.loads(raw_input)
        except json.JSONDecodeError:
            return {"arguments": raw_input}
        return parsed if isinstance(parsed, dict) else {"arguments": parsed}
    return {}


def is_codex_internal_user_message(text: str) -> bool:
    return text.lstrip().startswith("<environment_context>")


def append_transcript(transcript: list[tuple[str, str]], role: str, text: str) -> None:
    if text:
        transcript.append((role, text))


def build_session(
    source: str,
    path: Path,
    cwd: str | None,
    tool_calls: list[dict],
    user_messages: list[str],
    assistant_texts: list[str],
    hook_exit_codes: list[int],
    transcript: list[tuple[str, str]],
    session_id: str | None = None,
) -> dict:
    return {
        "source": source,
        "path": path,
        "session_id": session_id or path.stem,
        "cwd": cwd,
        "tool_calls": tool_calls,
        "user_messages": user_messages,
        "assistant_texts": assistant_texts,
        "hook_exit_codes": hook_exit_codes,
        "transcript": transcript,
        "archive_dir": SESSION_SOURCES[source]["archive_dir"],
        "mtime": path.stat().st_mtime,
    }


def parse_claude_session(path: Path) -> dict:
    """Extract structured data from a Claude session JSONL file."""
    tool_calls = []
    user_messages = []
    assistant_texts = []
    hook_exit_codes = []
    transcript = []
    cwd = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if cwd is None:
                cwd = obj.get("cwd")

            t = obj.get("type")

            if t == "user":
                content = obj.get("message", {}).get("content", "")
                for text in extract_text_blocks(content, {"text"}):
                    user_messages.append(text)
                    append_transcript(transcript, "User", text)

            elif t == "assistant":
                for block in obj.get("message", {}).get("content", []):
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        tool_calls.append({
                            "name": block.get("name"),
                            "input": block.get("input", {}),
                        })
                    elif block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            assistant_texts.append(text)
                            append_transcript(transcript, "Assistant", text)

            elif t == "attachment":
                att = obj.get("attachment", {})
                if att.get("type") == "hook_success" and "exitCode" in att:
                    hook_exit_codes.append(att["exitCode"])

    return build_session(
        "claude",
        path,
        cwd,
        tool_calls,
        user_messages,
        assistant_texts,
        hook_exit_codes,
        transcript,
    )


def parse_codex_session(path: Path) -> dict:
    """Extract structured data from a Codex session JSONL file."""
    tool_calls = []
    user_messages = []
    assistant_texts = []
    hook_exit_codes = []
    transcript = []
    event_transcript = []
    event_user_messages = []
    event_assistant_texts = []
    cwd = None
    session_id = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            payload = obj.get("payload") or {}
            t = obj.get("type")

            if t == "session_meta":
                session_id = payload.get("id") or session_id
                cwd = payload.get("cwd") or cwd
                continue

            if t == "turn_context":
                cwd = cwd or payload.get("cwd")
                continue

            if t == "event_msg":
                payload_type = payload.get("type")
                if payload_type == "user_message":
                    text = payload.get("message") or ""
                    if text.strip() and not is_codex_internal_user_message(text):
                        text = text.strip()
                        event_user_messages.append(text)
                        append_transcript(event_transcript, "User", text)
                elif payload_type == "agent_message":
                    text = payload.get("message") or ""
                    if text.strip():
                        text = text.strip()
                        event_assistant_texts.append(text)
                        append_transcript(event_transcript, "Assistant", text)
                continue

            if t != "response_item":
                continue

            payload_type = payload.get("type")
            if payload_type == "message":
                role = payload.get("role")
                if role == "user":
                    for text in extract_text_blocks(payload.get("content", []), {"input_text", "text"}):
                        if is_codex_internal_user_message(text):
                            continue
                        user_messages.append(text)
                        append_transcript(transcript, "User", text)
                elif role == "assistant":
                    for text in extract_text_blocks(payload.get("content", []), {"output_text", "text"}):
                        assistant_texts.append(text)
                        append_transcript(transcript, "Assistant", text)
            elif payload_type == "function_call":
                tool_calls.append({
                    "name": payload.get("name"),
                    "input": parse_tool_input(payload.get("arguments")),
                })
            elif payload_type == "function_call_output":
                output = payload.get("output")
                if isinstance(output, str):
                    match = re.search(r"Process exited with code (\d+)", output)
                    if match:
                        hook_exit_codes.append(int(match.group(1)))

    if not user_messages and event_user_messages:
        user_messages = event_user_messages
    if not assistant_texts and event_assistant_texts:
        assistant_texts = event_assistant_texts
    if not transcript and event_transcript:
        transcript = event_transcript

    return build_session(
        "codex",
        path,
        cwd,
        tool_calls,
        user_messages,
        assistant_texts,
        hook_exit_codes,
        transcript,
        session_id,
    )


def parse_session(source: str, path: Path) -> dict:
    if source == "claude":
        return parse_claude_session(path)
    if source == "codex":
        return parse_codex_session(path)
    raise ValueError(f"Unsupported session source: {source}")


def heuristic_filter(session: dict) -> str | None:
    """Returns 'TRASH', 'RESEARCH', or None (pass through)."""
    tool_calls = session["tool_calls"]

    # No tool calls at all
    if not tool_calls:
        return "TRASH"

    # Loop detection: same shell command repeated >LOOP_THRESHOLD times consecutively
    bash_commands = []
    for tool_call in tool_calls:
        command = tool_command(tool_call)
        if command:
            bash_commands.append(command)
    if bash_commands:
        max_run = 1
        current_run = 1
        for i in range(1, len(bash_commands)):
            if bash_commands[i] == bash_commands[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        if max_run > LOOP_THRESHOLD:
            return "TRASH"

    # Git delta check
    cwd = session["cwd"]
    if cwd and Path(cwd).exists():
        try:
            result = subprocess.run(
                ["git", "-C", cwd, "diff", "--name-only"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and not result.stdout.strip():
                return "RESEARCH"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return None


def tool_command(tool_call: dict) -> str | None:
    name = (tool_call.get("name") or "").lower()
    if name not in SHELL_TOOL_NAMES:
        return None

    tool_input = tool_call.get("input") or {}
    for key in COMMAND_KEYS:
        command = tool_input.get(key)
        if isinstance(command, str) and command.strip():
            return command.strip()
    return None


def outcome_signal(session: dict) -> str:
    """Returns 'positive', 'negative', or 'neutral'."""
    # Check last hook exit code
    if session["hook_exit_codes"] and session["hook_exit_codes"][-1] != 0:
        return "negative"

    # Check last user message for positive keywords
    if session["user_messages"]:
        last = session["user_messages"][-1].lower()
        if any(kw in last for kw in POSITIVE_KEYWORDS):
            return "positive"

    return "neutral"


def llm_classify(session: dict) -> str:
    """Returns 'KNOWLEDGE' or 'TRASH'."""
    full_transcript = "\n".join(session_transcript_lines(session))
    cutoff = max(1, int(len(full_transcript) * 0.8))
    tail = full_transcript[cutoff:]

    try:
        response = ollama.chat(
            model=CLASSIFY_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": tail},
            ],
        )
        result = response.message.content.strip().upper()
        return "KNOWLEDGE" if "KNOWLEDGE" in result else "TRASH"
    except Exception as e:
        print(f"  [warn] LLM classify failed: {e} — defaulting to KNOWLEDGE")
        return "KNOWLEDGE"


def session_transcript_lines(session: dict) -> list[str]:
    transcript = session.get("transcript") or []
    if transcript:
        return [
            f"{role}: {text}"
            for role, text in transcript
            if text
        ]

    lines = []
    for u, a in zip(session["user_messages"], session["assistant_texts"]):
        lines.append(f"User: {u}")
        lines.append(f"Assistant: {a}")
    return lines


def session_repo(session: dict) -> str:
    return Path(session["cwd"]).name if session.get("cwd") else "unknown"


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half].rstrip() + "\n...[truncated]...\n" + text[-half:].lstrip()


def format_sessions_for_llm(sessions: list[dict], max_chars: int = 8000) -> str:
    combined = []
    for session in sessions:
        source = session.get("source", "unknown")
        repo = session_repo(session)
        combined.append(
            f"=== Source: {source}; Repo: {repo} ===\n"
            + "\n".join(session_transcript_lines(session))
        )
    return truncate_text("\n\n".join(combined), max_chars)


def llm_generate(system_prompt: str, user_content: str) -> str:
    try:
        response = ollama.chat(
            model=CLASSIFY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return response.message.content.strip()
    except Exception as e:
        return f"[Summarization failed: {e}]"


def prompt_context_hits(text: str) -> int:
    lower = text.lower()
    return sum(1 for pattern in PROMPT_CONTEXT_PATTERNS if re.search(pattern, lower))


def prompt_message_score(text: str) -> int:
    stripped = text.strip()
    lower = stripped.lower()
    if len(stripped) < 25:
        return 0

    score = 0
    if len(stripped) >= 120:
        score += 1
    if "?" in stripped:
        score += 1
    if any(word in lower for word in PROMPT_ACTION_WORDS):
        score += 2
    if any(phrase in lower for phrase in PROMPT_FEEDBACK_PHRASES):
        score += 3
    score += min(prompt_context_hits(stripped), 4) * 2
    if stripped.count("\n") >= 2:
        score += 1
    if re.search(r"\b(start from|look at|focus on|rather than|because|given|based on)\b", lower):
        score += 1
    return score


def scored_prompt_messages(session: dict) -> list[tuple[int, str]]:
    scored = [
        (prompt_message_score(message), message.strip())
        for message in session.get("user_messages", [])
        if message.strip()
    ]
    return [
        (score, message)
        for score, message in sorted(scored, key=lambda item: -item[0])
        if score >= PROMPT_MESSAGE_MIN_SCORE
    ]


def prompt_session_score(session: dict) -> int:
    scored = scored_prompt_messages(session)
    return sum(score for score, _message in scored)


def is_prompt_worthy(session: dict) -> bool:
    return prompt_session_score(session) >= PROMPT_SESSION_MIN_SCORE


def format_prompt_patterns_input(sessions: list[dict], max_chars: int = 8000) -> str:
    entries = []
    for session in sessions:
        scored = scored_prompt_messages(session)
        if not scored:
            continue
        repo = session_repo(session)
        source = session.get("source", "unknown")
        prompt_blocks = []
        for score, message in scored[:3]:
            prompt_blocks.append(f"Score: {score}\nUser prompt:\n{truncate_text(message, 1200)}")
        entries.append(f"=== Source: {source}; Repo: {repo} ===\n" + "\n\n".join(prompt_blocks))
    return truncate_text("\n\n".join(entries), max_chars)


def text_tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower()):
        parts = [token]
        if "-" in token or "_" in token:
            parts.extend(re.split(r"[-_]+", token))
        for part in parts:
            if len(part) >= 4 and part not in TOKEN_STOPWORDS:
                tokens.add(part)
    return tokens


def session_match_text(sessions: list[dict]) -> str:
    parts = []
    for session in sessions:
        if session.get("cwd"):
            parts.append(session_repo(session))
        parts.extend(session_transcript_lines(session))
    return "\n".join(parts)


def iter_related_note_paths(current_note_path: Path | None = None):
    for folder in RELATED_NOTE_DIRS:
        root = VAULT_ROOT / folder
        if not root.exists():
            continue
        for path in root.rglob("*.md"):
            if current_note_path and path == current_note_path:
                continue
            if path.name.startswith(GENERATED_NOTE_PREFIXES):
                continue
            yield path


def obsidian_link(path: Path) -> str:
    try:
        link_path = path.relative_to(VAULT_ROOT).with_suffix("").as_posix()
    except ValueError:
        link_path = path.with_suffix("").as_posix()
    return f"[[{link_path}|{path.stem}]]"


def related_note_links(sessions: list[dict], current_note_path: Path | None = None) -> list[str]:
    session_tokens = text_tokens(session_match_text(sessions))
    if not session_tokens:
        return []

    scored = []
    for path in iter_related_note_paths(current_note_path):
        title_tokens = text_tokens(path.stem)
        parent_tokens = text_tokens(path.parent.name)

        try:
            content_tokens = text_tokens(path.read_text(errors="ignore")[:4000])
        except OSError:
            content_tokens = set()

        title_overlap = session_tokens & title_tokens
        parent_overlap = session_tokens & parent_tokens
        content_overlap = session_tokens & content_tokens
        score = len(title_overlap) * 10 + min(len(content_overlap), 16) + len(parent_overlap) * 2
        if score >= RELATED_NOTE_MIN_SCORE:
            scored.append((score, path.stem.lower(), path))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [obsidian_link(path) for _score, _title, path in scored[:RELATED_NOTE_LIMIT]]


def render_related_notes(sessions: list[dict], current_note_path: Path | None = None) -> str:
    links = related_note_links(sessions, current_note_path)
    if not links:
        return ""
    return "## Related\n\n" + "\n".join(f"- {link}" for link in links)


def append_related_section(body: str, sessions: list[dict], note_path: Path) -> str:
    related = render_related_notes(sessions, note_path)
    if not related:
        return body
    return body.rstrip() + "\n\n" + related


def append_related_to_existing_note(note_path: Path, sessions: list[dict]) -> bool:
    existing = note_path.read_text()
    if "## Related" in existing:
        return False

    related = render_related_notes(sessions, note_path)
    if not related:
        return False

    note_path.write_text(existing.rstrip() + "\n\n" + related + "\n")
    return True


def llm_summarize(sessions: list[dict]) -> str:
    """Summarize a list of sessions into a single Obsidian note body."""
    return llm_generate(SUMMARIZE_PROMPT, format_sessions_for_llm(sessions))


def session_date(session: dict) -> str:
    return datetime.fromtimestamp(session["mtime"], tz=timezone.utc).strftime("%Y-%m-%d")


def session_ids(sessions: list[dict]) -> list[str]:
    return [s["path"].stem for s in sessions]


def note_contains_session_ids(note_path: Path, ids: list[str]) -> bool:
    if not note_path.exists():
        return False
    existing = note_path.read_text()
    return all(session_id in existing for session_id in ids)


def write_daily_note(date: str, sessions: list[dict]) -> None:
    repos = sorted({Path(s["cwd"]).name for s in sessions if s["cwd"]})
    ids = session_ids(sessions)
    sources = sorted({s.get("source", "unknown") for s in sessions})

    note_path = VAULT_LOGS / f"{date}.md"
    if note_contains_session_ids(note_path, ids):
        if append_related_to_existing_note(note_path, sessions):
            print(f"  Added related links: {note_path}")
        print(f"  Skipped existing note: {note_path}")
        return

    print(f"  Summarizing {len(sessions)} session(s) for {date}...")
    body = append_related_section(llm_summarize(sessions), sessions, note_path)

    front_matter = (
        "---\n"
        f'session_ids: {json.dumps(ids)}\n'
        f'date: "{date}"\n'
        f'outcome: "verified_success"\n'
        f'repos: {json.dumps(repos)}\n'
        f'sources: {json.dumps(sources)}\n'
        f'tags: [auto-logged]\n'
        "---\n\n"
    )

    # Append if file already exists (multiple runs on same day)
    if note_path.exists():
        existing = note_path.read_text()
        note_path.write_text(existing + "\n\n---\n\n" + body)
    else:
        note_path.write_text(front_matter + f"# {date}\n\n" + body)

    print(f"  Written: {note_path}")


def write_research_note(date: str, sessions: list[dict]) -> None:
    repos = sorted({Path(s["cwd"]).name for s in sessions if s["cwd"]})
    sources = sorted({s.get("source", "unknown") for s in sessions})

    print(f"  Summarizing {len(sessions)} research session(s) for {date}...")
    note_path = VAULT_KNOWLEDGE / f"Research Sessions - {date}.md"
    body = append_related_section(llm_summarize(sessions), sessions, note_path)

    front_matter = (
        "---\n"
        f'date: "{date}"\n'
        f'outcome: "research"\n'
        f'session_count: {len(sessions)}\n'
        f'repos: {json.dumps(repos)}\n'
        f'sources: {json.dumps(sources)}\n'
        f'tags: [auto-logged, research]\n'
        "---\n\n"
    )

    note_path.write_text(
        front_matter
        + f"# Research Sessions - {date}\n\n"
        + body
    )

    print(f"  Written: {note_path}")


def write_prompt_note(date: str, sessions: list[dict]) -> None:
    prompt_input = format_prompt_patterns_input(sessions)
    if not prompt_input:
        return

    repos = sorted({session_repo(s) for s in sessions if session_repo(s) != "unknown"})
    sources = sorted({s.get("source", "unknown") for s in sessions})
    prompt_count = sum(len(scored_prompt_messages(s)) for s in sessions)
    note_path = VAULT_PROMPTS / f"Prompt Patterns - {date}.md"

    print(f"  Summarizing {prompt_count} prompt pattern(s) for {date}...")
    body = append_related_section(
        llm_generate(PROMPT_PATTERNS_PROMPT, prompt_input),
        sessions,
        note_path,
    )

    front_matter = (
        "---\n"
        f'date: "{date}"\n'
        f'outcome: "prompt_patterns"\n'
        f'prompt_count: {prompt_count}\n'
        f'repos: {json.dumps(repos)}\n'
        f'sources: {json.dumps(sources)}\n'
        f'tags: [auto-logged, prompts]\n'
        "---\n\n"
    )

    note_path.write_text(front_matter + f"# Prompt Patterns - {date}\n\n" + body)
    print(f"  Written: {note_path}")


def project_dir_for_session(session: dict) -> Path | None:
    repo = session_repo(session)
    if repo == "unknown":
        return None

    project_dir = VAULT_PROJECTS / repo
    if project_dir.is_dir():
        return project_dir
    return None


def write_project_note(date: str, project_dir: Path, sessions: list[dict]) -> None:
    sources = sorted({s.get("source", "unknown") for s in sessions})
    repo = project_dir.name
    note_path = project_dir / f"Session Notes - {date}.md"

    print(f"  Summarizing {len(sessions)} project session(s) for {repo} on {date}...")
    body = append_related_section(
        llm_generate(PROJECT_NOTE_PROMPT, format_sessions_for_llm(sessions)),
        sessions,
        note_path,
    )

    front_matter = (
        "---\n"
        f'date: "{date}"\n'
        f'outcome: "project_session_notes"\n'
        f'session_count: {len(sessions)}\n'
        f'repo: "{repo}"\n'
        f'sources: {json.dumps(sources)}\n'
        f'tags: [auto-logged, project-session]\n'
        "---\n\n"
    )

    note_path.write_text(front_matter + f"# Session Notes - {date}\n\n" + body)
    print(f"  Written: {note_path}")


def archive_session(session: dict) -> None:
    archive_dir = session["archive_dir"]
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / session["path"].name
    shutil.copy2(session["path"], dest)


def main() -> None:
    VAULT_LOGS.mkdir(parents=True, exist_ok=True)
    VAULT_KNOWLEDGE.mkdir(parents=True, exist_ok=True)
    VAULT_PROMPTS.mkdir(parents=True, exist_ok=True)

    print(f"Scanning Claude/Codex sessions modified in last {LOOKBACK_DAYS} days...")
    paths = find_recent_sessions(LOOKBACK_DAYS)
    source_counts = defaultdict(int)
    for source, _path in paths:
        source_counts[source] += 1
    found_summary = ", ".join(f"{source}={source_counts[source]}" for source in sorted(source_counts))
    print(f"Found {len(paths)} session file(s)" + (f" ({found_summary})" if found_summary else ""))

    knowledge_by_date: dict[str, list[dict]] = defaultdict(list)
    research_by_date: dict[str, list[dict]] = defaultdict(list)
    prompt_by_date: dict[str, list[dict]] = defaultdict(list)
    project_by_date: dict[tuple[str, Path], list[dict]] = defaultdict(list)
    counts = {"KNOWLEDGE": 0, "RESEARCH": 0, "TRASH": 0}

    for source, path in sorted(paths, key=lambda item: (item[0], str(item[1]))):
        print(f"\nProcessing [{source}]: {path.name}")
        session = parse_session(source, path)
        date = session_date(session)

        if is_prompt_worthy(session):
            prompt_by_date[date].append(session)

        verdict = heuristic_filter(session)
        if verdict == "TRASH":
            print(f"  → TRASH (heuristic: no tools or loop detected)")
            archive_session(session)
            counts["TRASH"] += 1
            continue
        if verdict == "RESEARCH":
            print(f"  → RESEARCH (git delta: no file changes)")
            research_by_date[date].append(session)
            project_dir = project_dir_for_session(session)
            if project_dir:
                project_by_date[(date, project_dir)].append(session)
            counts["RESEARCH"] += 1
            continue

        signal = outcome_signal(session)
        print(f"  Outcome signal: {signal}")

        classification = llm_classify(session)
        print(f"  LLM classification: {classification}")

        if classification == "TRASH":
            archive_session(session)
            counts["TRASH"] += 1
            continue

        knowledge_by_date[date].append(session)
        project_dir = project_dir_for_session(session)
        if project_dir:
            project_by_date[(date, project_dir)].append(session)
        counts["KNOWLEDGE"] += 1

    print(f"\nWriting {len(knowledge_by_date)} daily note(s)...")
    for date, sessions in sorted(knowledge_by_date.items()):
        write_daily_note(date, sessions)

    print(f"\nWriting {len(research_by_date)} research note(s) to Knowledge...")
    for date, sessions in sorted(research_by_date.items()):
        write_research_note(date, sessions)

    print(f"\nWriting {len(prompt_by_date)} prompt pattern note(s) to Prompts...")
    for date, sessions in sorted(prompt_by_date.items()):
        write_prompt_note(date, sessions)

    print(f"\nWriting {len(project_by_date)} project note(s) to Projects...")
    for (date, project_dir), sessions in sorted(project_by_date.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        write_project_note(date, project_dir, sessions)

    print(f"\nDone. KNOWLEDGE={counts['KNOWLEDGE']} RESEARCH={counts['RESEARCH']} TRASH={counts['TRASH']}")


if __name__ == "__main__":
    main()
