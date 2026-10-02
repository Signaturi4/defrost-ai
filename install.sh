#!/usr/bin/env sh
# defrost-ai one-go install (pinned to a release tag):
#   curl -fsSL https://raw.githubusercontent.com/Signaturi4/defrost-ai/main/install.sh | sh
# Installs the `defrost` CLI (uv tool, Python 3.12) from release tag v$DEFROST_VERSION, downloads and verifies the
# weights, and registers the Claude Code slash commands + MCP server for all projects.
#
#   DEFROST_VERSION=1.2.0   install this release (default below; kept equal to pyproject by scripts/release_check.py)
#   DEFROST_REF=main        install a branch or commit instead of a release (moving target)
#   DEFROST_REPO=/path      install from a local checkout (development)
set -eu

VERSION="${DEFROST_VERSION:-1.2.0}"
GITHUB="https://github.com/Signaturi4/defrost-ai"

fail() { printf 'defrost install: %s\n' "$*" >&2; exit 1; }

# ---- platform + uv --------------------------------------------------------------------------------------------------
os=$(uname -s); arch=$(uname -m)
case "$os" in
  Darwin|Linux) ;;
  *) fail "unsupported OS '$os' (macOS and Linux only)" ;;
esac
if [ "$os" = Darwin ] && [ "$arch" != arm64 ]; then
  echo "note: Intel Mac: no MLX backend; the PyTorch backend is used (slower reranking)." >&2
fi
command -v uv >/dev/null 2>&1 || fail "uv is required and was not found. Install it first:
    curl -LsSf https://astral.sh/uv/install.sh | sh
  (see https://docs.astral.sh/uv/getting-started/installation/), then run this script again."

# ---- what to install --------------------------------------------------------------------------------------------------
if [ -n "${DEFROST_REPO:-}" ]; then
  SRC="${DEFROST_REPO}"
  WHAT="local checkout ${DEFROST_REPO}"
elif [ -n "${DEFROST_REF:-}" ]; then
  SRC="git+${GITHUB}@${DEFROST_REF}"
  WHAT="${DEFROST_REF} (not a release)"
else
  if command -v git >/dev/null 2>&1 && ! git ls-remote --exit-code --tags "${GITHUB}" "refs/tags/v${VERSION}" >/dev/null 2>&1; then
    fail "release v${VERSION} not found at ${GITHUB}/releases.
  Pick a published version (DEFROST_VERSION=x.y.z) or the development branch (DEFROST_REF=main)."
  fi
  SRC="git+${GITHUB}@v${VERSION}"
  WHAT="release v${VERSION}"
fi

echo "==> installing defrost ${WHAT} (uv tool, Python 3.12)"
uv tool uninstall kev-memory >/dev/null 2>&1 || true            # the package was called kev-memory before 1.2
# mac extra: MLX backend; its dependencies are marker-gated to Apple Silicon, so it is a no-op elsewhere
uv tool install --force --reinstall --python 3.12 "defrost-ai[code,mac] @ ${SRC}" \
  || fail "uv could not install defrost from ${SRC} (see the uv error above)."

BIN="${UV_TOOL_BIN_DIR:-$HOME/.local/bin}"
export PATH="$BIN:$PATH"
command -v defrost >/dev/null 2>&1 || fail "installed, but 'defrost' is not on PATH; add ${BIN} to your PATH."
echo "==> $(defrost --version)"

echo "==> downloading weights (about 140 MB, sha256-checked) to ~/.cache/defrost-ai"
defrost download-weights || fail "the CLI is installed but the model weights are not (see the message above).
  Releases with weights: ${GITHUB}/releases. Retry later with: defrost download-weights"

if command -v claude >/dev/null 2>&1; then
  echo "==> registering Claude Code slash commands + MCP server (all projects)"
  defrost claude install --user || fail "could not register with Claude Code; run: defrost claude install --user"
else
  echo "Claude Code CLI not found; after installing it, run: defrost claude install --user"
fi

cat <<'EOF'

Done. Next step, in any repository:
  claude            then type   /defrost-setup
It asks 3 questions (how much to install, search mode, doc trust), builds the memory, and keeps it fresh.
EOF
