#!/usr/bin/env python3
"""
Claude session → Obsidian note pipeline.

Reads ~/.claude/projects/ for sessions modified in the last 7 days,
filters them through the gatekeeper, and writes aggregated daily notes
to Obsidian vault Logs/.
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
VAULT_LOGS = Path(_env.get("OBSIDIAN_LOGS_DIR", str(Path.home() / "Trung's Brain" / "Logs")))
ARCHIVE_DIR = Path(_env.get("CLAUDE_ARCHIVE_DIR", str(Path.home() / ".claude" / "archive")))
LOOKBACK_DAYS = int(_env.get("LOOKBACK_DAYS", "7"))
CLASSIFY_MODEL = _env.get("CLASSIFY_MODEL", "llama3:8b")

POSITIVE_KEYWORDS = {"that worked", "thanks", "thank you", "perfect", "done", "merged", "lgtm", "nice", "great"}
LOOP_THRESHOLD = 10

CLASSIFY_PROMPT = """Determine if this technical session concluded with a solution or a useful discovery.
- If it ended in failure, confusion, or was a trivial task (general chat, simple lookups): Output 'TRASH'.
- If it contains a bug fix, an architectural decision, a project decision, or a non-obvious technical finding: Output 'KNOWLEDGE'.
Response must be a single word."""

SUMMARIZE_PROMPT = """You are summarizing a Claude Code engineering session for an Obsidian knowledge base.

Extract and format as Markdown:
1. **What was worked on** — repo, ticket, brief description
2. **Technical decisions made** — non-obvious choices and the reasoning
3. **Gotchas / surprises** — things that weren't obvious, bugs discovered
4. **Commands / patterns used** — key bash commands, code patterns worth remembering
5. **Pending tasks** — anything left unresolved

Use [[wikilinks]] for related projects or topics. Be concise. Omit small talk and filler."""


def find_recent_sessions(lookback_days: int) -> list[Path]:
    cutoff = time.time() - lookback_days * 86400
    sessions = []
    for jsonl in CLAUDE_PROJECTS.rglob("*.jsonl"):
        if jsonl.stat().st_mtime >= cutoff:
            sessions.append(jsonl)
    return sessions


def parse_session(path: Path) -> dict:
    """Extract structured data from a session JSONL file."""
    tool_calls = []
    user_messages = []
    assistant_texts = []
    hook_exit_codes = []
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
                if isinstance(content, str) and content.strip():
                    user_messages.append(content.strip())
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            user_messages.append(block["text"].strip())

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
                        assistant_texts.append(block["text"].strip())

            elif t == "attachment":
                att = obj.get("attachment", {})
                if att.get("type") == "hook_success" and "exitCode" in att:
                    hook_exit_codes.append(att["exitCode"])

    return {
        "path": path,
        "cwd": cwd,
        "tool_calls": tool_calls,
        "user_messages": user_messages,
        "assistant_texts": assistant_texts,
        "hook_exit_codes": hook_exit_codes,
        "mtime": path.stat().st_mtime,
    }


def heuristic_filter(session: dict) -> str | None:
    """Returns 'TRASH', 'RESEARCH', or None (pass through)."""
    tool_calls = session["tool_calls"]

    # No tool calls at all
    if not tool_calls:
        return "TRASH"

    # Loop detection: same bash command repeated >LOOP_THRESHOLD times consecutively
    bash_commands = [
        tc["input"].get("command", "")
        for tc in tool_calls
        if tc["name"] == "Bash"
    ]
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
    all_text = []
    for u, a in zip(session["user_messages"], session["assistant_texts"]):
        all_text.append(f"User: {u}")
        all_text.append(f"Assistant: {a}")

    full_transcript = "\n".join(all_text)
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


def llm_summarize(sessions: list[dict]) -> str:
    """Summarize a list of sessions into a single Obsidian note body."""
    combined = []
    for s in sessions:
        repo = Path(s["cwd"]).name if s["cwd"] else "unknown"
        turns = []
        for u, a in zip(s["user_messages"], s["assistant_texts"]):
            turns.append(f"User: {u}\nAssistant: {a}")
        combined.append(f"=== Repo: {repo} ===\n" + "\n\n".join(turns))

    transcript = "\n\n".join(combined)
    # Trim to ~8000 chars to stay within context
    if len(transcript) > 8000:
        transcript = transcript[:4000] + "\n...[truncated]...\n" + transcript[-4000:]

    try:
        response = ollama.chat(
            model=CLASSIFY_MODEL,
            messages=[
                {"role": "system", "content": SUMMARIZE_PROMPT},
                {"role": "user", "content": transcript},
            ],
        )
        return response.message.content.strip()
    except Exception as e:
        return f"[Summarization failed: {e}]"


def session_date(session: dict) -> str:
    return datetime.fromtimestamp(session["mtime"], tz=timezone.utc).strftime("%Y-%m-%d")


def write_daily_note(date: str, sessions: list[dict]) -> None:
    repos = sorted({Path(s["cwd"]).name for s in sessions if s["cwd"]})
    session_ids = [s["path"].stem for s in sessions]

    print(f"  Summarizing {len(sessions)} session(s) for {date}...")
    body = llm_summarize(sessions)

    front_matter = (
        "---\n"
        f'session_ids: {json.dumps(session_ids)}\n'
        f'date: "{date}"\n'
        f'outcome: "verified_success"\n'
        f'repos: {json.dumps(repos)}\n'
        f'tags: [auto-logged]\n'
        "---\n\n"
    )

    note_path = VAULT_LOGS / f"{date}.md"

    # Append if file already exists (multiple runs on same day)
    if note_path.exists():
        existing = note_path.read_text()
        note_path.write_text(existing + "\n\n---\n\n" + body)
    else:
        note_path.write_text(front_matter + f"# {date}\n\n" + body)

    print(f"  Written: {note_path}")


def archive_session(session: dict) -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / session["path"].name
    shutil.copy2(session["path"], dest)


def main() -> None:
    VAULT_LOGS.mkdir(parents=True, exist_ok=True)

    print(f"Scanning sessions modified in last {LOOKBACK_DAYS} days...")
    paths = find_recent_sessions(LOOKBACK_DAYS)
    print(f"Found {len(paths)} session file(s)")

    knowledge_by_date: dict[str, list[dict]] = defaultdict(list)
    counts = {"KNOWLEDGE": 0, "RESEARCH": 0, "TRASH": 0}

    for path in sorted(paths):
        print(f"\nProcessing: {path.name}")
        session = parse_session(path)

        verdict = heuristic_filter(session)
        if verdict == "TRASH":
            print(f"  → TRASH (heuristic: no tools or loop detected)")
            archive_session(session)
            counts["TRASH"] += 1
            continue
        if verdict == "RESEARCH":
            print(f"  → RESEARCH (git delta: no file changes)")
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

        date = session_date(session)
        knowledge_by_date[date].append(session)
        counts["KNOWLEDGE"] += 1

    print(f"\nWriting {len(knowledge_by_date)} daily note(s)...")
    for date, sessions in sorted(knowledge_by_date.items()):
        write_daily_note(date, sessions)

    print(f"\nDone. KNOWLEDGE={counts['KNOWLEDGE']} RESEARCH={counts['RESEARCH']} TRASH={counts['TRASH']}")


if __name__ == "__main__":
    main()
