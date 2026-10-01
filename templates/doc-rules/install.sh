#!/usr/bin/env bash
# Install the doc rules into a project: copies the rules, templates and tools into <project>/docs and appends the
# short, highlighted rule block to the END of <project>/CLAUDE.md (idempotent: re-running replaces the block).
#   bash install.sh /path/to/project
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="$(cd "${1:-.}" && pwd)"
mkdir -p "$DST/docs/tools" "$DST/docs/templates"
cp "$SRC/DOC_RULES.md" "$DST/docs/DOC_RULES.md"
[ -f "$DST/docs/GLOSSARY.md" ] || sed "s/YYYY-MM-DD/$(date +%F)/" "$SRC/GLOSSARY.template.md" > "$DST/docs/GLOSSARY.md"
cp "$SRC/PAGE.template.md" "$DST/docs/templates/PAGE.template.md"
cp "$SRC/tools/doc_lint.py" "$SRC/tools/extract_facts.py" "$DST/docs/tools/"
CM="$DST/CLAUDE.md"; touch "$CM"
python3 - "$CM" "$SRC/CLAUDE.snippet.md" <<'PY'
import re, sys
cm, snip = sys.argv[1], open(sys.argv[2]).read().strip()
text = open(cm).read()
text = re.sub(r"\n*<!-- defrost-ai:doc-rules:start -->.*?<!-- defrost-ai:doc-rules:end -->\n*", "\n", text, flags=re.S)
open(cm, "w").write(text.rstrip() + ("\n\n" if text.strip() else "") + snip + "\n")
PY
echo "installed: docs/DOC_RULES.md, docs/GLOSSARY.md, docs/templates/PAGE.template.md, docs/tools/{doc_lint,extract_facts}.py"
echo "CLAUDE.md: rule block placed at the end ($(wc -l < "$SRC/CLAUDE.snippet.md") lines)"
