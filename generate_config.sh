#!/usr/bin/env bash
# =============================================================================
# Generate .mcp.json from .env
# Run this after modifying .env to keep .mcp.json in sync.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Error: $ENV_FILE not found. Copy .env.example to .env first."
  exit 1
fi

# Source .env (safely, line by line)
set -a
source "$ENV_FILE"
set +a

# Derive defaults for values not set .env
: "${VENV_PATH:=$HARNESS_PROJECT_DIR/venv}"
: "${SERVER_SCRIPT:=$HARNESS_PROJECT_DIR/server.py}"

cat > "$SCRIPT_DIR/.mcp.json" <<MCPEOF
{
  "mcpServers": {
    "harness-memory": {
      "type": "stdio",
      "command": "$VENV_PATH/bin/python3",
      "args": ["$SERVER_SCRIPT"]
    }
  }
}
MCPEOF

echo "✓ Generated .mcp.json from .env"
