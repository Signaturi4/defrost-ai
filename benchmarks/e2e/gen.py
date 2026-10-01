"""E2E benchmark questions (graphify original vs graphify + kev-memory) on three repos never used in training.

Each question is PAIRED: a gold documentation section AND gold code symbol(s) that implement the behaviour.
The generator picks the gold symbols from the component's full list of source symbols (not from kev-memory's
doc->code links), so the code gold does not favour our linker. Two styles:
  behavior  plain-words question, no identifiers (how developers ask when they don't know the API)
  named     the question names one identifier or config option and asks how it behaves / how to use it
Every candidate is then verified by an independent call that sees the question, the section and the gold symbols'
source; only questions that pass every check are kept. Sections sharing any 13-gram with our training data are
excluded, so no gold section was seen in training.

    python gen.py generate      # -> questions.raw.jsonl (resumable)
    python gen.py verify        # -> questions.jsonl"""
import json
import random
import re
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
MEM = Path.home() / ".kev-memory/e2e"
if not (MEM / "knowledge.sqlite").exists():
    MEM = Path.home() / ".kev-memory/e2e.building"
REPOS = HERE / "repos"
PER_COMPONENT = 45
SKIP = re.compile(r"(changelog|changes|contributing|code_of_conduct|license|security|authors|history|release)", re.I)
LEAKED = set(json.load(open(HERE / "leaked_grams.json")))


def grams(t):
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + 13]) for i in range(len(w) - 12)}


def claude(prompt, schema, model="sonnet"):
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json", "--tools", "", "--json-schema",
           json.dumps(schema), "--no-session-persistence", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
    for _ in range(4):
        try:
            out = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                            stdin=subprocess.DEVNULL).stdout)
            if isinstance(out, list):
                out = next(e for e in reversed(out) if e.get("type") == "result")
            return out.get("structured_output") or json.loads(out["result"])
        except Exception:
            continue
    return None


def load():
    db = sqlite3.connect(MEM / "knowledge.sqlite")
    secs = [dict(zip(["id", "component", "path", "heading", "a", "b", "text", "words"], r)) for r in db.execute(
        "SELECT id, component, path, heading_path, line_start, line_end, text, words FROM sections")]
    code = json.load(open(MEM / "code_graph.json"))["nodes"]
    syms = {}
    for n in code:
        f = n["source_file"]
        if "/tests/" in f or "/bench/" in f or "typing_tests" in f or "/docs/" in f or not f.endswith(".py"):
            continue
        if n["label"].endswith(".py"):
            continue
        syms.setdefault(n["component"], []).append(n)
    return secs, syms


def eligible(secs):
    out = []
    for s in secs:
        if s["words"] < 50 or SKIP.search(s["path"]) or not re.search(r"(docs/|README)", s["path"]):
            continue
        if grams(s["text"]) & LEAKED:
            continue
        out.append(s)
    return out


GEN_SCHEMA = {"type": "object", "additionalProperties": False,
              "required": ["usable", "question", "answer", "evidence", "gold_symbols"],
              "properties": {"usable": {"type": "boolean"}, "question": {"type": "string"},
                             "answer": {"type": "string"}, "evidence": {"type": "string"},
                             "gold_symbols": {"type": "array", "items": {"type": "string"}}}}
GEN_PROMPT = """You are writing a benchmark question for a code+documentation search system over the Python project
"{component}". Below is ONE documentation section and the list of source-code symbols of the project (label | file).

Write ONE question a developer working with this project would realistically ask, such that:
- the documentation section answers it (the answer must be stated in the section), and
- one to three symbols from the list implement the behaviour the question is about (the code a developer would open
  to see how it works). Choose them from the list only, copying the label exactly.
Style: {style}
Also give a short reference answer (1-3 sentences, from the section) and one verbatim evidence sentence copied from
the section. Set usable=false if the section is navigation, boilerplate, or no symbol in the list implements it.

STYLE behavior = plain words, NO identifiers, function/class/option names or code in the question; describe the
behaviour or goal. STYLE named = the question names exactly one identifier or configuration option that appears in the
section, and asks how it behaves or how to use it. Do not copy distinctive multi-word phrases from the section.

## Section ({path}, lines {a}-{b}) {heading}
{text}

## Source symbols of {component}
{symbols}"""

VER_SCHEMA = {"type": "object", "additionalProperties": False,
              "required": ["section_answers", "symbols_implement", "copies_section", "ambiguous", "notes"],
              "properties": {"section_answers": {"type": "boolean"}, "symbols_implement": {"type": "boolean"},
                             "copies_section": {"type": "boolean"}, "ambiguous": {"type": "boolean"},
                             "notes": {"type": "string"}}}
VER_PROMPT = """You are auditing one benchmark item for a code+documentation search system. Be strict.

Question: {question}
Reference answer: {answer}

## Gold documentation section ({path})
{text}

## Gold code symbols (source)
{code}

Answer:
- section_answers: does the section itself contain the information needed to answer the question correctly?
- symbols_implement: is at least one of the gold symbols genuinely the code that implements the behaviour asked about?
- copies_section: does the question copy a distinctive multi-word phrase (4+ words) from the section verbatim?
- ambiguous: is the question so generic that many unrelated sections of this project's docs would answer it equally?
- notes: one short sentence."""


def symbol_source(n, max_lines=60):
    f = REPOS / n["source_file"]
    try:
        lines = f.read_text(errors="ignore").splitlines()
    except OSError:
        return ""
    m = re.match(r"L(\d+)", n.get("source_location") or "")
    start = int(m.group(1)) - 1 if m else 0
    return f"# {n['label']} ({n['source_file']}:{start + 1})\n" + "\n".join(lines[start:start + max_lines])


def generate():
    secs, syms = load()
    el = eligible(secs)
    rng = random.Random(0)
    by = {}
    for s in el:
        by.setdefault(s["component"], []).append(s)
    picked = []
    for comp, ss in sorted(by.items()):
        rng.shuffle(ss)
        picked += ss[:PER_COMPONENT]
    print({c: len(v) for c, v in by.items()}, "eligible; picked", len(picked), flush=True)
    out_f = HERE / "questions.raw.jsonl"
    done = {json.loads(l)["section_id"] for l in open(out_f)} if out_f.exists() else set()
    todo = [(i, s) for i, s in enumerate(picked) if s["id"] not in done]

    def run(item):
        i, s = item
        style = "behavior" if i % 2 == 0 else "named"
        symbols = "\n".join(f"{n['label']} | {n['source_file']}" for n in syms[s["component"]])
        r = claude(GEN_PROMPT.format(component=s["component"], style=style, path=s["path"], a=s["a"], b=s["b"],
                                     heading=s["heading"], text=s["text"][:6000], symbols=symbols), GEN_SCHEMA)
        if not r:
            return None
        labels = {n["label"]: n for n in syms[s["component"]]}
        gold = [labels[g] for g in r["gold_symbols"] if g in labels]
        return {"section_id": s["id"], "component": s["component"], "style": style, "usable": r["usable"],
                "question": r["question"].strip(), "answer": r["answer"].strip(), "evidence": r["evidence"],
                "gold_doc": {"path": s["path"], "lines": [s["a"], s["b"]], "section_id": s["id"]},
                "gold_code": [{"id": n["id"], "label": n["label"], "source_file": n["source_file"],
                               "source_location": n.get("source_location")} for n in gold],
                "invalid_symbols": [g for g in r["gold_symbols"] if g not in labels]}

    with ThreadPoolExecutor(8) as ex, open(out_f, "a") as f:
        for k, row in enumerate(ex.map(run, todo)):
            if row:
                f.write(json.dumps(row) + "\n"); f.flush()
            if k % 10 == 0:
                print(f"  {k}/{len(todo)}", flush=True)


def verify():
    secs, syms = load()
    text = {s["id"]: s for s in secs}
    rows = [json.loads(l) for l in open(HERE / "questions.raw.jsonl")]
    nodes = {n["id"]: n for v in syms.values() for n in v}
    for r in rows:                                   # the generator often copies "label | file": resolve by both
        by_lf = {(n["label"], n["source_file"]): n for n in syms[r["component"]]}
        by_l = {}
        for n in syms[r["component"]]:
            by_l.setdefault(n["label"], []).append(n)
        for g in r.pop("invalid_symbols", []):
            lab, _, f = (x.strip() for x in g.partition(" | "))
            hit = [by_lf[(lab, f)]] if (lab, f) in by_lf else by_l.get(lab, [])
            for n in hit[:1]:
                if all(c["id"] != n["id"] for c in r["gold_code"]):
                    r["gold_code"].append({"id": n["id"], "label": n["label"], "source_file": n["source_file"],
                                           "source_location": n.get("source_location")})
    rows = [r for r in rows if r["usable"] and r["gold_code"]]

    def run(r):
        s = text[r["section_id"]]
        code = "\n\n".join(symbol_source(nodes[g["id"]]) for g in r["gold_code"])
        v = claude(VER_PROMPT.format(question=r["question"], answer=r["answer"], path=s["path"], text=s["text"][:6000],
                                     code=code), VER_SCHEMA)
        return r | {"verify": v}

    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(run, rows))
    keep = [r for r in out if r["verify"] and r["verify"]["section_answers"] and r["verify"]["symbols_implement"]
            and not r["verify"]["copies_section"] and not r["verify"]["ambiguous"]
            and (r["style"] == "named" or not re.search(r"`|\w+\(\)|\b\w+_\w+\b|\b[a-z]+[A-Z]\w*", r["question"]))]
    for i, r in enumerate(keep):
        r["id"] = f"e2e-{i:03d}"
    (HERE / "questions.verified_all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    (HERE / "questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    from collections import Counter
    print(f"raw usable {len(rows)} -> kept {len(keep)}", Counter((r["component"], r["style"]) for r in keep))


if __name__ == "__main__":
    {"generate": generate, "verify": verify}[sys.argv[1]]()
