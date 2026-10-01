"""Agentic end-to-end eval: Claude Sonnet answers each question as an agent inside the repo.

Arms (same model, prompt, turn cap and file tools: Read, Grep, Glob). Each arm runs in its own copy of the repo:
  none      file tools only
  graphify  stock graphify 0.4.32 as a user installs it: full semantic graph (/graphify skill), `graphify claude
            install` (CLAUDE.md rules + PreToolUse hook), `graphify` CLI allowed via Bash, MCP server
  defrost       the graphify fork installed the same way over the same graph; the only differences are the memory tools
            (CLI `graphify memory search`, MCP `memory_search`) and the one CLAUDE.md rule that points to them
Scored: AnswerAccuracy (RAGAS NVIDIA, Haiku judge) vs the reference answer, code_named (answer names a gold symbol),
cost (USD), turns, wall time.

    STOCK_GRAPHIFY_PY=... FORK_GRAPHIFY_PY=... python benchmarks/e2e/agent_eval.py [N_QUESTIONS]"""
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path.home() / "e2e-bench"
CACHE = HERE / "agent_cache.json"
STOCK_PY = os.environ.get("STOCK_GRAPHIFY_PY", "python")      # a venv with stock graphifyy==0.4.32 from PyPI
FORK_PY = os.environ.get("FORK_GRAPHIFY_PY", "python")       # a venv with graphify + integrations/graphify patch
PROMPT = """A developer asks a question about the Python project in the current directory. Use the available tools to
find the answer in the project's documentation and code. Answer concisely (at most 4 sentences). Then, on a last
line starting with "Code:", name the function(s) or class(es) that implement this behaviour.

Question: {q}"""
ARMS = ("none", "graphify", "defrost")


def mcp_config(arm, comp):
    graph = str(HERE / "arms" / arm / comp / "graphify-out/graph.json")
    if arm == "none":
        return '{"mcpServers":{}}', ""
    py = STOCK_PY if arm == "graphify" else FORK_PY
    env = {"DEFROST_DOMAINS": "e2e"} if arm == "defrost" else {}
    cfg = {"mcpServers": {"graphify": {"command": py, "args": ["-m", "graphify.serve", graph], "env": env}}}
    return json.dumps(cfg), ",mcp__graphify"


def run_agent(q, arm):
    cfg, extra = mcp_config(arm, q["component"])
    cwd = HERE / "arms" / arm / q["component"]               # each arm: own copy + its own `graphify claude install`
    tools = "Read,Grep,Glob" if arm == "none" else "Read,Grep,Glob,Bash"
    allowed = "Read,Grep,Glob" if arm == "none" else "Read,Grep,Glob,Bash(graphify:*)" + extra
    env = dict(os.environ, PATH=str(Path(STOCK_PY if arm == "graphify" else FORK_PY).parent) + ":" + os.environ["PATH"],
               DEFROST_DOMAINS="e2e")
    cmd = ["claude", "-p", PROMPT.format(q=q["question"]), "--model", "sonnet", "--output-format", "stream-json",
           "--verbose", "--max-turns", "20", "--strict-mcp-config", "--mcp-config", cfg, "--no-session-persistence",
           "--allowedTools", allowed, "--tools", tools]
    t0 = time.time()
    for _ in range(3):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=cwd, stdin=subprocess.DEVNULL, env=env)
            events = [json.loads(l) for l in r.stdout.splitlines() if l.strip().startswith("{")]
            out = next(e for e in reversed(events) if e.get("type") == "result")
            tools = Counter(c["name"] for e in events if e.get("type") == "assistant"
                            for c in e["message"].get("content", []) if c.get("type") == "tool_use")
            return {"response": out.get("result"), "cost": out.get("total_cost_usd"), "turns": out.get("num_turns"),
                    "seconds": round(time.time() - t0, 1), "is_error": out.get("is_error"), "tools": dict(tools)}
        except Exception as e:
            err = str(e)
            time.sleep(10)
    return {"response": None, "error": err}


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    n_q = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    qs = [json.loads(l) for l in open(HERE / "questions.jsonl")]
    rng = random.Random(1)
    by = {}
    for q in qs:
        by.setdefault((q["component"], q["style"]), []).append(q)
    pick = []
    for k in sorted(by):
        rng.shuffle(by[k])
    while len(pick) < n_q:                                          # round-robin over (repo, style) strata
        for k in sorted(by):
            if by[k] and len(pick) < n_q:
                pick.append(by[k].pop())
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    jobs = [(q, a) for q in pick for a in ARMS if f"{q['id']}|{a}" not in cache or not cache[f"{q['id']}|{a}"].get("response")]
    print(f"{len(pick)} questions x {len(ARMS)} arms; {len(jobs)} runs to go", flush=True)
    with ThreadPoolExecutor(4) as ex:
        for n, ((q, a), res) in enumerate(zip(jobs, ex.map(lambda j: run_agent(*j), jobs)), 1):
            cache[f"{q['id']}|{a}"] = res
            CACHE.write_text(json.dumps(cache, indent=0))
            print(f"  {n}/{len(jobs)} {q['id']} {a} ${res.get('cost')} turns {res.get('turns')}", flush=True)
    # judge accuracy
    sys.path.insert(0, str(Path(__file__).parent))
    from llm import claude_llm
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.metrics import AnswerAccuracy
    todo = [(q, a) for q in pick for a in ARMS if cache[f"{q['id']}|{a}"].get("response")
            and cache[f"{q['id']}|{a}"].get("nv_accuracy") is None]
    if todo:
        ds = EvaluationDataset.from_list([{"user_input": q["question"], "reference": q["answer"],
                                           "response": cache[f"{q['id']}|{a}"]["response"]} for q, a in todo])
        df = evaluate(ds, metrics=[AnswerAccuracy(llm=claude_llm("haiku"))], show_progress=False,
                      run_config=RunConfig(max_workers=8, timeout=900)).to_pandas()
        for (q, a), (_, row) in zip(todo, df.iterrows()):
            v = row.get("nv_accuracy")
            cache[f"{q['id']}|{a}"]["nv_accuracy"] = None if v is None or v != v else float(v)
        CACHE.write_text(json.dumps(cache, indent=0))
    print("accuracy tuples", Counter(cache[f"{q['id']}|{a}"].get("nv_accuracy") for q in pick for a in ARMS))
    table = {}
    for a in ARMS:
        rows = [(q, cache[f"{q['id']}|{a}"]) for q in pick]
        ok = [(q, r) for q, r in rows if r.get("nv_accuracy") is not None]
        named = []
        for q, r in rows:
            line = (r.get("response") or "").split("Code:")[-1]
            names = [g["label"].strip(".").replace("()", "").split(".")[-1] for g in q["gold_code"]]
            named.append(float(any(re.search(rf"\b{re.escape(n)}\b", line) for n in names)))
        tool_use = Counter()
        for _, r in rows:
            tool_use.update(r.get("tools") or {})
        table[a] = {"n": len(rows), "tool_calls": dict(tool_use), "accuracy": boot([r["nv_accuracy"] for _, r in ok]), "code_named": boot(named),
                    "cost_usd": boot([r.get("cost") or 0 for _, r in rows]), "turns": boot([r.get("turns") or 0 for _, r in rows]),
                    "seconds": boot([r.get("seconds") or 0 for _, r in rows])}
    paired = {}
    for a in ("none", "defrost"):
        d = [cache[f"{q['id']}|{a}"]["nv_accuracy"] - cache[f"{q['id']}|graphify"]["nv_accuracy"] for q in pick
             if cache[f"{q['id']}|{a}"].get("nv_accuracy") is not None and cache[f"{q['id']}|graphify"].get("nv_accuracy") is not None]
        c = [(cache[f"{q['id']}|{a}"].get("cost") or 0) - (cache[f"{q['id']}|graphify"].get("cost") or 0) for q in pick]
        paired[a] = {"accuracy_delta": boot(d), "cost_delta": boot(c)}
    (HERE / "agent_report.json").write_text(json.dumps({"questions": [q["id"] for q in pick], "table": table,
                                                        "vs_graphify": paired}, indent=1))
    for a, t in table.items():
        print(f"{a:9s} acc {t['accuracy'][0]:.3f}  code_named {t['code_named'][0]:.3f}  cost ${t['cost_usd'][0]:.3f}"
              f"  turns {t['turns'][0]:.1f}  {t['seconds'][0]:.0f}s")
    for a, p in paired.items():
        print(f"  {a} vs graphify: acc {p['accuracy_delta'][0]:+.3f} [{p['accuracy_delta'][1]:+.3f}, {p['accuracy_delta'][2]:+.3f}]"
              f"  cost {p['cost_delta'][0]:+.3f} [{p['cost_delta'][1]:+.3f}, {p['cost_delta'][2]:+.3f}]")


if __name__ == "__main__":
    main()
