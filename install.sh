#!/usr/bin/env sh
# defrost-ai one-go install:
#   curl -fsSL https://raw.githubusercontent.com/Signaturi4/defrost-ai/main/install.sh | sh
# Installs the `defrost` CLI (uv tool, Python 3.12), downloads and verifies the weights, and registers the
# Claude Code slash commands + MCP server for all projects. Then open Claude in a repo and run /defrost-setup.
set -eu

REPO="${DEFROST_REPO:-git+https://github.com/Signaturi4/defrost-ai}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

echo "==> installing defrost (uv tool, Python 3.12)"
uv tool uninstall kev-memory >/dev/null 2>&1 || true            # the package was called kev-memory before 1.2
uv tool install --force --reinstall --python 3.12 "defrost-ai[code,mcp,mac] @ ${REPO}"   # mac: MLX backend, Apple Silicon only (marker-gated)
export PATH="$HOME/.local/bin:$PATH"

echo "==> downloading weights (about 140 MB, sha256-checked) to ~/.cache/defrost-ai"
defrost download-weights

if command -v claude >/dev/null 2>&1; then
  echo "==> registering Claude Code slash commands + MCP server (all projects)"
  defrost claude install --user
else
  echo "Claude Code CLI not found; later run: defrost claude install --user"
fi

cat <<'EOF'

Done. In any repository:
  claude            then type   /defrost-setup
It asks how the memory should stay fresh (default: build now + on every merge/commit to main),
builds it, and from then on Claude uses memory_search for how/why questions.
EOF
