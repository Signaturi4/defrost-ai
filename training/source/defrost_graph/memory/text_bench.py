"""Section-retrieval benchmark for the text KB: "which part of the docs answers this developer question?"

Build (once per corpus, then frozen):
  1. sample sections, stratified by component: >= 60 words, not a changelog, not mostly code
  2. a writer model (claude -p, sonnet) turns each section into a developer question that the section answers,
     paraphrased: it must not copy distinctive phrases or the heading (so plain keyword match is not free)
  3. a different verifier model (haiku) sees only the question + section and must confirm the section answers it
     (round-trip filter); rejected items are dropped, not rewritten
  4. gold = the source span + every near-duplicate span in the corpus (word 5-shingle Jaccard >= 0.8), so a system
     that returns a verbatim copy of the section from another file is not punished
  5. split by hash of the section id: dev (1/3, allowed for tuning) and test (2/3, only for the final report)

Gold is (path, line range), not section ids, so any chunking can be scored: a result hits a gold span when it is in
the same file and covers >= 50% of the span's lines or the span covers >= 50% of the result's.

    uv run python -m defrost_graph.memory.text_bench --workspace defrost_graph/memory/workspaces/heldout.json \\
        --n 150 --out defrost_graph/memory/evals/text_heldout_v1.jsonl"""
import argparse
import hashlib
import json
import random
import re
import sqlite3
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from defrost_graph.memory.workspace import load

SKIP_PATH = re.compile(r"(CHANGES|CHANGELOG|HISTORY|RELEASE|AUTHORS|LICENSE|CONTRIBUTORS)", re.I)
GEN_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["usable", "question", "answer"],
              "properties": {"usable": {"type": "boolean"}, "question": {"type": "string"},
                             "answer": {"type": "string"}}}
VER_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answers_it"],
              "properties": {"answers_it": {"type": "boolean"}}}

GEN = """You write evaluation questions for a documentation search engine used by developers.
Below is one section of a project's documentation. Write ONE question a developer working on this project would
realistically ask, whose answer is contained in this section.
Rules:
- Paraphrase. Do not copy the section heading, and do not reuse distinctive multi-word phrases from the text.
  Ordinary technical terms and identifiers a developer would know (e.g. a function or service name) are fine.
- The question must be specific enough that this section (not the project in general) answers it.
- One sentence, no preamble. Also give the short answer (1-2 sentences) from the section.
- Set usable=false if the section has no real content to ask about (a table of contents, only links, boilerplate).

SECTION ({path}, {heading}):
{text}"""

VER = """Does the documentation excerpt below contain the answer to the question? Answer true only if a developer
reading just this excerpt would get the answer.

QUESTION: {question}

EXCERPT:
{text}"""


def code_share(text):
    lines = text.splitlines()
    inside, n = False, 0
    for ln in lines:
        if ln.strip().startswith("```"):
            inside = not inside
            n += 1
        elif inside or ln.startswith("    ") or ln.lstrip().startswith(">>>"):
            n += 1
    return n / max(1, len(lines))


def shingles(text, k=5):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def claude(prompt, schema, model):
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json", "--tools", "",
           "--json-schema", json.dumps(schema), "--no-session-persistence", "--strict-mcp-config",
           "--mcp-config", '{"mcpServers":{}}']
    for _ in range(2):
        try:
            out = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout)
            if isinstance(out, list):
                out = next(e for e in reversed(out) if e.get("type") == "result")
            return out.get("structured_output") or json.loads(out["result"])
        except (json.JSONDecodeError, KeyError, TypeError, StopIteration, subprocess.TimeoutExpired):
            continue
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--writer", default="sonnet")
    ap.add_argument("--verifier", default="haiku")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    ws = load(args.workspace)
    db = sqlite3.connect(ws["out"] / "text_kb.sqlite")
    rows = db.execute("SELECT id, component, path, heading_path, line_start, line_end, text, words FROM sections").fetchall()
    cols = ["id", "component", "path", "heading", "a", "b", "text", "words"]
    secs = [dict(zip(cols, r)) for r in rows]

    pool = defaultdict(list)
    for s in secs:
        if s["words"] >= 60 and not SKIP_PATH.search(s["path"]) and code_share(s["text"]) < 0.5:
            pool[s["component"]].append(s)
    rng = random.Random(args.seed)
    for v in pool.values():
        rng.shuffle(v)
    # round-robin over components (stratified), oversample 1.4x for the verifier's rejections
    want, picked = int(args.n * 1.4), []
    while len(picked) < want and any(pool.values()):
        for c in sorted(pool):
            if pool[c] and len(picked) < want:
                picked.append(pool[c].pop())

    def build(s):
        g = claude(GEN.format(path=s["path"], heading=s["heading"], text=s["text"][:6000]), GEN_SCHEMA, args.writer)
        if not g or not g["usable"] or not g["question"].strip():
            return None
        v = claude(VER.format(question=g["question"], text=s["text"][:6000]), VER_SCHEMA, args.verifier)
        if not v or not v["answers_it"]:
            return None
        return {**s, "question": g["question"].strip(), "answer": g["answer"].strip()}

    with ThreadPoolExecutor(args.workers) as ex:
        items = [x for x in ex.map(build, picked) if x]
    print(f"sampled {len(picked)}, kept {len(items)} after writer + verifier")

    sh = {s["id"]: shingles(s["text"]) for s in secs if s["words"] >= 30}
    code = {n["id"]: n.get("source_file") for n in json.loads((ws["out"] / "code_kb.json").read_text())["nodes"]}
    linked = defaultdict(set)                     # section -> code files it names (exact doc->code links)
    for sid, nid in db.execute("SELECT section_id, node_id FROM links WHERE confidence='EXTRACTED'"):
        if code.get(nid):
            linked[sid].add(code[nid])
    out = []
    for it in items[:args.n]:
        a = sh.get(it["id"]) or shingles(it["text"])
        gold = [{"path": it["path"], "lines": [it["a"], it["b"]]}]
        for s in secs:
            b = sh.get(s["id"])
            if s["id"] != it["id"] and b and len(a & b) / len(a | b) >= 0.8:
                gold.append({"path": s["path"], "lines": [s["a"], s["b"]]})
        split = "dev" if int(hashlib.sha256(it["id"].encode()).hexdigest(), 16) % 3 == 0 else "test"
        out.append({"id": f"{ws['name']}-{len(out):03d}", "split": split, "component": it["component"],
                    "question": it["question"], "answer": it["answer"], "source": it["id"], "gold": gold,
                    "gold_code": sorted(linked.get(it["id"], []))})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(o) + "\n" for o in out)
    Path(args.out).write_text(body)
    by = defaultdict(int)
    for o in out:
        by[(o["split"], o["component"])] += 1
    print(f"{len(out)} questions -> {args.out}  sha256 {hashlib.sha256(body.encode()).hexdigest()}")
    print(dict(sorted(by.items())), "with near-dup gold:", sum(len(o["gold"]) > 1 for o in out),
          "with linked code:", sum(bool(o["gold_code"]) for o in out))


if __name__ == "__main__":
    main()
