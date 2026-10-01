"""Doc/code conflicts are decided by a person, not by the agent.

When a doc section and the code disagree, the agent shows both sides and asks the user which is right. The answer is
recorded as one Markdown file per decision in the project's context repository
(~/.defrost-ai/<domain>.context/decisions/, see defrost_ai/context_repo.py), one git commit each, so decisions are
auditable, shareable through the repository's remote, searchable (domain `<domain>-context`), and shown on later
search hits as a `resolved:` line, so the same conflict is not asked again. Decisions recorded before the context
repository existed (~/.defrost-ai/<domain>.conflicts.jsonl) are migrated on the first write.

    decision   meaning                                         what the agent does next
    code       the code is right, the doc is outdated          updates the doc section in the same change
    doc        the doc describes the intended behaviour        reports the code as a bug; changes code only on request
    both       not a real conflict (both are fine)             nothing; the note explains why
    open       nobody knows yet                                adds an open-question note to the doc section"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

DECISIONS = {
    "code": "code is right: update the doc",
    "doc": "doc is right: the code is a bug",
    "both": "not a conflict",
    "open": "unresolved: open question",
}

QUESTION = (
    "Doc/code conflicts are the user's call, not yours. When a doc section and the code disagree (or a hit carries a "
    "`! doc/code conflict` / `! doc may be stale` line that the code confirms), do not pick a side. Show both: the doc "
    "(`path:Lstart-end`, what it says) and the code (`path:line`, what it does). Then ask with AskUserQuestion, one "
    "question per conflict (header \"Conflict\"), options: \"Code is right: update the doc\", \"Doc is right: the "
    "code is a bug\", \"Not a conflict\", \"Not sure: mark as open question\". Record the answer with "
    "`remember(kind='decision')`, then act on it: update the doc; or report the bug without changing code unless asked; "
    "or nothing; or add an open-question note to the section. Until the user answers, state both versions and present "
    "neither as fact. Hits with a `resolved:` line were already decided: follow that decision and do not ask again. "
    "Without an interactive user (e.g. `claude -p`), list them under **Doc/code conflicts: needs your decision**.")


def ledger(domain: str) -> Path:
    """Pre-context-repository log (read until migrated)."""
    return Path(os.environ.get("DEFROST_HOME", "~/.defrost-ai")).expanduser() / f"{domain}.conflicts.jsonl"


def render_decision(row: dict) -> tuple[str, str]:
    """(repo-relative path, Markdown) for one decision: readable body plus the exact record in a json block."""
    from defrost_ai.context_repo import render, slug
    stamp = row["at"].replace("-", "").replace(":", "").replace("T", "-")
    lines = f"L{row['doc_lines'][0]}-{row['doc_lines'][-1]}" if row.get("doc_lines") else ""
    body = (f"# {row['meaning']}: {row['doc_path']} {lines}\n\n"
            f"- **Doc:** `{row['doc_path']}` {lines}: {row.get('doc_says') or '(not recorded)'}\n"
            f"- **Code:** `{row.get('code_ref') or '?'}`: {row.get('code_does') or '(not recorded)'}\n"
            f"- **Decision (by the user, {row['at'][:10]}):** {row['meaning']}"
            + (f"\n- **Note:** {row['note']}" if row.get("note") else "") +
            f"\n\nFacts:\n- `{row['doc_path']}` {lines} → contradicts → `{row.get('code_ref') or 'the code'}` "
            f"(found {row['at'][:10]})\n"
            f"- `{row['doc_path']}` {lines} → user decision → {row['meaning']} ({row['at'][:10]})\n"
            "\n```json\n" + json.dumps(row, indent=1) + "\n```\n")
    return (f"decisions/{stamp}-{slug(row['doc_path'] + '-' + row['decision'])}.md",
            render(f"{row['meaning']}: {row['doc_path']} {lines}".strip(),
                   f"{row['doc_path']} {lines}: {row.get('doc_says') or ''} / code: {row.get('code_does') or ''}",
                   body))


def record(domain: str, doc_path: str, decision: str, doc_says: str = "", code_does: str = "", code_ref: str = "",
           doc_lines: list[int] | None = None, note: str = "") -> dict:
    from defrost_ai import context_repo
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)}, not {decision!r}")
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "domain": domain, "doc_path": doc_path,
           "doc_lines": doc_lines or [], "doc_says": doc_says, "code_ref": code_ref, "code_does": code_does,
           "decision": decision, "meaning": DECISIONS[decision], "note": note}
    context_repo.ensure(domain)                                            # also migrates the old jsonl once
    rel, text = render_decision(row)
    row["commit"] = context_repo.commit(domain, {rel: text}, f"decision({decision}): {doc_path} - {DECISIONS[decision]}")
    row["file"] = rel
    return row


def load(domain: str) -> list[dict]:
    """All decisions, oldest first. Read-only (never creates the repository): search calls this on every query."""
    from defrost_ai import context_repo
    rows = []
    for f in context_repo.files(domain, "decisions"):
        m = re.search(r"```json\n(.*?)\n```", f.read_text(), re.S)
        if m:
            try:
                rows.append(json.loads(m.group(1)) | {"file": f"decisions/{f.name}"})
            except ValueError:
                continue
    old = ledger(domain)
    if old.exists():
        rows += [json.loads(line) for line in old.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["at"])


def for_section(rows: list[dict], path: str, lines: list[int]) -> list[dict]:
    """Decisions about this section: same doc path and overlapping lines (or no lines recorded). Latest first."""
    a, b = lines
    out = [r for r in rows if r["doc_path"] == path and
           (not r["doc_lines"] or (r["doc_lines"][0] <= b and r["doc_lines"][-1] >= a))]
    return sorted(out, key=lambda r: r["at"], reverse=True)
