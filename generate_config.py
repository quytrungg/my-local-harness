#!/usr/bin/env python3
"""Generate .mcp.json from .env. Run after modifying .env to keep config in sync."""
import json
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ENV_FILE = SCRIPT_DIR / ".env"
MCP_FILE = SCRIPT_DIR / ".mcp.json"

if not ENV_FILE.exists():
    print(f"Error: {ENV_FILE} not found. Copy .env.example to .env first.")
    exit(1)

# Parse .env (simple key=value, skip comments/blanks)
env = {}
with open(ENV_FILE) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
        if m:
            val = m.group(2).strip().strip('"').strip("'")
            env[m.group(1)] = val

project_dir = env.get("HARNESS_PROJECT_DIR", str(SCRIPT_DIR))
venv_path = env.get("VENV_PATH", os.path.join(project_dir, "venv"))
server_script = os.path.join(project_dir, "server.py")

config = {
    "mcpServers": {
        "harness-memory": {
            "type": "stdio",
            "command": os.path.join(venv_path, "bin", "python3"),
            "args": [server_script],
        }
    }
}

MCP_FILE.write_text(json.dumps(config, indent=2) + "\n")
print(f"✓ Generated {MCP_FILE} from .env")
print(f"  command: {config['mcpServers']['harness-memory']['command']}")
