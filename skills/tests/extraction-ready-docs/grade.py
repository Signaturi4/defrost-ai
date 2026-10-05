#!/usr/bin/env python3
"""Programmatic grader for the extraction-ready-docs behaviour evals.

    python grade.py iteration-1        # writes grading.json into every run-*/ folder

Layout: <iteration>/eval-<id>-<name>/<config>/run-<k>/outputs/{repo/, final_message.md, questions.md}
Every assertion is checked by code, so reruns grade identically (consistency is measured on the runs, not the
grader)."""
import json
import re
import subprocess
import sys
from pathlib import Path

SKILL_SCRIPTS = Path(__file__).resolve().parents[1] / "extraction-ready-docs" / "scripts"
NAME_OK = re.compile(r"^(?:[A-Z][A-Z0-9]*(?:_[A-Z][A-Z0-9]*)*_\d+-)?[a-z0-9]+(?:-[a-z0-9]+)*(?:--superseded-\d{4}-\d{2}-\d{2})?$")


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""


def check(text, passed, evidence):
    return {"text": text, "passed": bool(passed), "evidence": evidence}


def frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    out = {}
    for line in (m.group(1).splitlines() if m else []):
        k, _, v = line.partition(":")
        out[k.strip()] = v.split("#")[0].strip().strip("'\"").lower()
    return out


def lint_errors(cmd):
    r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if ": ERROR:" in l]


def grade_setup(out: Path):
    repo = out / "repo"
    cm = read(repo / "CLAUDE.md")
    first_rule = min([i for i in (cm.find("<!--"), cm.lower().find("documentation")) if i >= 0] or [len(cm)])
    rules = [p for p in (repo / "docs").glob("*.md") if "rule" in p.name.lower()] if (repo / "docs").exists() else []
    return [
        check("CLAUDE.md keeps the original project instructions above the new rules",
              "# Timekit app" in cm and "pnpm test" in cm and cm.find("pnpm test") < first_rule,
              f"original text at {cm.find('pnpm test')}, rules start at {first_rule}"),
        check("The rules declare the living mode (edit in place, git keeps history)",
              re.search(r"lifecycle mode:\s*\**\s*living", cm, re.I),
              "found 'Lifecycle mode: living'" if re.search(r"lifecycle mode:\s*\**\s*living", cm, re.I)
              else "no living lifecycle mode in CLAUDE.md"),
        check("Version-suffix copies (_v2 / final / .bak) are explicitly banned",
              re.search(r"_v2|final|\.bak", cm, re.I), "searched CLAUDE.md for _v2, final, .bak"),
        check("Exactly one rule block in CLAUDE.md (no duplicates)",
              cm.count(":start -->") <= 1 and cm.lower().count("documentation rules") <= 1,
              f"{cm.count(':start -->')} marker blocks"),
        check("Full rules live in docs/, CLAUDE.md stays short (<= 45 lines)",
              rules and len(cm.splitlines()) <= 45, f"rule files {[p.name for p in rules]}, CLAUDE.md "
                                                    f"{len(cm.splitlines())} lines"),
        check("No defrost-only tooling in a general repository",
              not (repo / "docs/tools").exists() and "defrost-ai:doc-rules" not in cm,
              f"docs/tools exists: {(repo / 'docs/tools').exists()}"),
    ]


def grade_interview(out: Path):
    repo = out / "repo"
    notes = read(repo / "maya-notes.txt") or read(Path(__file__).parent / "fixtures/e2-interview/maya-notes.txt")
    new = [p for p in (repo / "docs").rglob("*.md") if "maya" in p.name.lower()]
    p = new[0] if len(new) == 1 else None
    text = read(p) if p else ""
    fm = frontmatter(text)
    stem = p.name.rsplit(".", 1)[0] if p else ""
    src = re.search(r"^#{1,6}\s+Sources\s*$(.*?)(?=^#{1,6}\s|\Z)", text, re.M | re.S)
    quotes = [q for line in text.splitlines() for q in re.findall(r"[\"“]([^\"”\n]{12,}?)[\"”]", line)
              if not line.lstrip().startswith("- ") or "→" not in line]
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731
    invented = [q for q in quotes if norm(q) not in norm(notes)]
    readme = "".join(read(d / "README.md") for d in (p.parents if p else []) if (repo / "docs") in (d, *d.parents))
    errs = lint_errors([str(SKILL_SCRIPTS / "doc_lint.py"), str(p)]) if p else ["no file"]
    rerrs = lint_errors([str(SKILL_SCRIPTS / "repo_lint.py"), str(repo), "--lifecycle", "per-document"])
    return [
        check("Exactly one interview file for Maya under docs/", len(new) == 1, [str(x.relative_to(repo)) for x in new]),
        check("File is in the research area (docs/research/...)", p and "research" in p.parts, str(p)),
        check("Name is what-who-when with the full date (interview-maya-2026-10-12)",
              p and NAME_OK.match(stem) and "2026-10-12" in stem and "interview" in stem, stem),
        check("Frontmatter marks it as a source file (source: true)", fm.get("source") in ("true", "yes"),
              f"source: {fm.get('source')}"),
        check("Frontmatter declares lifecycle immutable (raw evidence)", fm.get("lifecycle") == "immutable",
              f"lifecycle: {fm.get('lifecycle')}"),
        check("A Sources section names Maya and the interview date",
              src and "maya" in src.group(1).lower() and ("2026-10-12" in src.group(1) or "12 oct" in src.group(1).lower()),
              (src.group(1).strip()[:160] if src else "no Sources section")),
        check("Its nearest folder README lists the new file", p and p.name in readme,
              f"{p.name if p else None} in a parent README"),
        check("No invented quotes (every quote appears in the raw notes)", not invented,
              f"{len(quotes)} quotes, invented: {invented[:2]}"),
        check("doc_lint reports no ERROR for the new file", not errs, errs[:2]),
        check("repo_lint reports no ERROR for the repo", not rerrs, rerrs[:3]),
    ]


ORIGINAL = ["competitors_v2.md", "competitors_v3_FINAL.md", "Notes Monday.md", "Interview Leo 2.10.md",
            "keyword research.pdf", "pricing.md"]


def grade_messy(out: Path):
    repo = out / "repo"
    proposal = read(out / "questions.md") + "\n" + read(out / "final_message.md")
    for extra in (repo / "docs").rglob("*.md") if (repo / "docs").exists() else []:
        if re.search(r"propos|plan|clean", extra.name):
            proposal += "\n" + read(extra)
    kept = [n for n in ORIGINAL if (repo / "docs" / n).exists()]
    names = set(re.findall(r"`?([\w./-]*[a-z0-9][\w-]*\.(?:md|pdf))`?", proposal))
    proposed = {Path(n).name for n in names if not any(o.endswith(Path(n).name) for o in ORIGINAL)} - set(ORIGINAL) - {"README.md", "DOC_RULES.md", "KNOWLEDGE_RULES.md",
                                                                   "CLAUDE.md"}
    proposed -= {"final_message.md", "questions.md"}                  # the test harness's own files
    bad = [n for n in proposed if not NAME_OK.match(n.rsplit(".", 1)[0].replace("YYYY-MM-DD", "2000-01-01"))]
    low = proposal.lower()
    return [
        check("No original file was moved, renamed or deleted before the user agreed", len(kept) == len(ORIGINAL),
              f"still present: {len(kept)}/{len(ORIGINAL)}; missing {sorted(set(ORIGINAL) - set(kept))}"),
        check("A written proposal or question for the user exists", len(proposal.strip()) > 200,
              f"{len(proposal.strip())} chars across questions.md / final_message.md"),
        check("At least 3 proposed new file names, all following the naming rule", len(proposed) >= 3 and not bad,
              f"proposed {sorted(proposed)[:8]}, breaking the rule: {bad[:4]}"),
        check("Leo's interview gets a full YYYY-MM-DD date (or the missing year is asked)",
              re.search(r"interview-leo-\d{4}-\d{2}-\d{2}", proposal) or re.search(r"year", low),
              (re.search(r"interview-leo[\w-]*", proposal) or [None])[0]),
        check("The two competitor copies are merged into one current file",
              "merge" in low and "competitor" in low, "looked for 'merge' + 'competitor'"),
        check("The PDF gets a same-name .md text twin", re.search(r"text twin|keyword-research[\w-]*\.md", low),
              "looked for 'text twin' or keyword-research*.md"),
        check("The Monday decision goes to a decision register", "decision register" in low or "decision-register" in low,
              "looked for 'decision register'"),
    ]


GRADERS = {"setup-living-mode": grade_setup, "add-interview-source-file": grade_interview,
           "messy-folder-propose-first": grade_messy}


def main(iteration):
    for run in sorted(Path(iteration).glob("eval-*/*/run-*")):
        name = run.parent.parent.name.split("-", 2)[2]
        exp = GRADERS[name](run / "outputs")
        exp = [{**e, "evidence": str(e["evidence"])} for e in exp]
        n = sum(e["passed"] for e in exp)
        (run / "grading.json").write_text(json.dumps({"expectations": exp, "summary": {
            "passed": n, "failed": len(exp) - n, "total": len(exp), "pass_rate": round(n / len(exp), 2)}}, indent=1))
        print(f"{run.relative_to(iteration)}: {n}/{len(exp)}")


if __name__ == "__main__":
    main(sys.argv[1])
