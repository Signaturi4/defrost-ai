"""Doc/code conflicts are decided by a person, not by the agent.

When a doc section and the code disagree, the agent shows both sides and asks the user which is right. The answer is
recorded here, one JSON line per decision in ~/.kev-memory/<domain>.conflicts.jsonl, so search results can show it
and the same conflict is not asked again.

    decision   meaning                                         what the agent does next
    code       the code is right, the doc is outdated          updates the doc section in the same change
    doc        the doc describes the intended behaviour        reports the code as a bug; changes code only on request
    both       not a real conflict (both are fine)             nothing; the note explains why
    open       nobody knows yet                                adds an open-question note to the doc section"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

HOME = Path(os.environ.get("KEV_MEMORY_HOME", "~/.kev-memory")).expanduser()
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
    "`memory_resolve_conflict`, then act on it: update the doc; or report the bug without changing code unless asked; "
    "or nothing; or add an open-question note to the section. Until the user answers, state both versions and present "
    "neither as fact. Hits with a `resolved:` line were already decided: follow that decision and do not ask again. "
    "Without an interactive user (e.g. `claude -p`), list them under **Doc/code conflicts: needs your decision**.")


def ledger(domain: str) -> Path:
    return HOME / f"{domain}.conflicts.jsonl"


def record(domain: str, doc_path: str, decision: str, doc_says: str = "", code_does: str = "", code_ref: str = "",
           doc_lines: list[int] | None = None, note: str = "") -> dict:
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)}, not {decision!r}")
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "domain": domain, "doc_path": doc_path,
           "doc_lines": doc_lines or [], "doc_says": doc_says, "code_ref": code_ref, "code_does": code_does,
           "decision": decision, "meaning": DECISIONS[decision], "note": note}
    HOME.mkdir(parents=True, exist_ok=True)
    with open(ledger(domain), "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load(domain: str) -> list[dict]:
    f = ledger(domain)
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def for_section(rows: list[dict], path: str, lines: list[int]) -> list[dict]:
    """Decisions about this section: same doc path and overlapping lines (or no lines recorded). Latest first."""
    a, b = lines
    out = [r for r in rows if r["doc_path"] == path and
           (not r["doc_lines"] or (r["doc_lines"][0] <= b and r["doc_lines"][-1] >= a))]
    return sorted(out, key=lambda r: r["at"], reverse=True)
