"""Agent QA benchmark: the same project questions through Claude Code headless, per arm and repeat, scored by a blind
LLM judge. An arm is a git ref of the project plus what the agent gets: its CLAUDE.md, the defrost memory (MCP server
and/or prompt hook), skills, and an optional dev build of defrost.

    python3 bench/agent_qa/aqa.py CONFIG setup              # worktrees under cfg.base (neutral letters), answer key stripped
    python3 bench/agent_qa/aqa.py CONFIG memory             # build a defrost memory per defrost arm; no triggers installed
    python3 bench/agent_qa/aqa.py CONFIG hooks              # record what the prompt hook injects per question (free)
    python3 bench/agent_qa/aqa.py CONFIG estimate           # runs planned and a cost bound; spends nothing
    python3 bench/agent_qa/aqa.py CONFIG run --yes [--only Q01 Q02] [--arms A B] [--repeats N]
    python3 bench/agent_qa/aqa.py CONFIG judge --yes        # blind judge, a different model from the answering one
    python3 bench/agent_qa/aqa.py CONFIG sample             # 10 blinded answers to calibration.csv for hand scoring
    python3 bench/agent_qa/aqa.py CONFIG agree              # hand vs judge agreement per field
    python3 bench/agent_qa/aqa.py CONFIG report             # report.md: metrics per arm, per question, paired sign test
    python3 bench/agent_qa/aqa.py CONFIG teardown           # remove the worktrees

The config and the questions stay with the project (they hold its facts); see config.example.json and
questions.example.json. Outputs go to cfg.out. Nothing spends money without --yes.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import signal
import statistics as st
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULTS = {"base": "/tmp/aqa", "model": "claude-sonnet-5", "effort": "medium", "judge_model": "claude-opus-5-5",
            "repeats": 3, "max_turns": 30, "max_usd_per_run": 1.5, "est_usd_per_run": 0.06, "strip": [],
            "instruction": "Answer the user's question about the project in the current directory. Use the files in "
                           "it. Be concise, cite the files you relied on, and say so if sources disagree or you are "
                           "unsure."}
READ_TOOLS = "Read,Grep,Glob,Bash(ls:*),Bash(cat:*),Bash(grep:*),Bash(find:*),Bash(wc:*),Bash(head:*),Bash(sed:*)"
DENIED = "Edit,Write,NotebookEdit,WebFetch,WebSearch,Task,Agent,Bash(git:*)"   # git can read other arms' branches
FIELDS = ("correctness", "flags_conflict", "uses_stale_fact", "hallucination", "cites_source")


def load_cfg(path):
    cfg = DEFAULTS | json.loads(Path(path).read_text())
    cfg["_dir"] = Path(path).resolve().parent
    cfg["out"] = Path(cfg.get("out") or cfg["_dir"] / "out")
    cfg["questions_file"] = (cfg["_dir"] / cfg["questions"]).resolve()
    return cfg


def questions(cfg):
    return json.loads(cfg["questions_file"].read_text())["questions"]


def paths(cfg):
    return json.loads((cfg["out"] / "arms.json").read_text())["paths"]


def arm_env(cfg, arm):
    spec, env = cfg["arms"][arm], dict(os.environ)
    if spec.get("defrost_bin"):                         # a dev build: its `defrost` comes first on PATH
        env["PATH"] = str(Path(spec["defrost_bin"]).expanduser()) + os.pathsep + env["PATH"]
    if spec.get("defrost"):
        env["DEFROST_DOMAIN"] = spec["defrost"]["domain"]
    env.update(spec.get("env", {}))
    return env


# ---- setup -----------------------------------------------------------------------------------------------------------
def setup(cfg, _):
    cfg["out"].mkdir(parents=True, exist_ok=True)
    base = Path(cfg["base"]); base.mkdir(parents=True, exist_ok=True)
    letters = random.Random(11).sample("abcdefghijklmnopqrstuvwxyz", len(cfg["arms"]))
    out, refs = {}, {}
    for (arm, spec), letter in zip(cfg["arms"].items(), letters):
        ref = subprocess.run(["git", "-C", cfg["repo"], "rev-parse", "--short", spec["ref"]], capture_output=True,
                             text=True, check=True).stdout.strip()
        path = Path(spec["path"]) if spec.get("path") else base / letter   # arms may share a worktree
        if not path.exists():
            subprocess.run(["git", "-C", cfg["repo"], "worktree", "add", "--detach", str(path), ref], check=True)
        for pattern in cfg["strip"]:                    # the answer key and anything that reveals it: never in an arm
            for hit in path.glob(pattern):
                shutil.rmtree(hit) if hit.is_dir() else hit.unlink()
        if not spec.get("claude_md", True):
            (path / "CLAUDE.md").unlink(missing_ok=True)
        out[arm], refs[arm] = str(path), ref
    (cfg["out"] / "arms.json").write_text(json.dumps({"paths": out, "refs": refs}, indent=1))
    print(json.dumps(out, indent=1))


def memory(cfg, _):
    """Build each defrost arm's memory from its own worktree, as `defrost setup` does minus every trigger: worktrees
    share the repository's git hooks, so a normal setup would overwrite the real project's refresh hook."""
    for arm, spec in cfg["arms"].items():
        if not spec.get("defrost"):
            continue
        d, env = spec["defrost"], arm_env(cfg, arm)
        python = Path(shutil.which("defrost", path=env["PATH"])).read_text().splitlines()[0][2:]
        path = paths(cfg)[arm]
        code = ("import json; from pathlib import Path; from defrost_ai import trust; "
                "from defrost_ai.project_setup import setup\n"
                f"level = {d.get('doc_trust')!r} or trust.suggest(Path({path!r}))[0]\n"
                f"r = setup({path!r}, {d['domain']!r}, 'now', doc_trust=level, memory_dir=None, "
                f"prompt_context={bool(d.get('hook', True))})\n"
                "print(json.dumps({k: r.get(k) for k in ('domain', 'doc_trust', 'triggers', 'build')}, default=str))")
        print(f"[{arm}] building memory {d['domain']}")
        subprocess.run([python, "-c", code], check=True, env=env)


def hooks(cfg, _):
    """What the prompt hook injects for each question, per defrost arm (deterministic search; costs nothing)."""
    for arm, spec in cfg["arms"].items():
        if not (spec.get("defrost") and spec["defrost"].get("hook", True)):
            continue
        out = cfg["out"] / "hooks" / arm.replace("/", "_")
        out.mkdir(parents=True, exist_ok=True)
        for q in questions(cfg):
            payload = json.dumps({"prompt": q["q"], "cwd": paths(cfg)[arm], "hook_event_name": "UserPromptSubmit"})
            p = subprocess.run(["defrost", "hook", "prompt"], input=payload, capture_output=True, text=True,
                               env=arm_env(cfg, arm), cwd=paths(cfg)[arm], timeout=120)
            (out / f"{q['id']}.txt").write_text(p.stdout)
        print(f"[{arm}] hook output -> {out}")


# ---- run ---------------------------------------------------------------------------------------------------------------
def plan(cfg, args):
    qs = [q for q in questions(cfg) if not args.only or q["id"] in args.only]
    arms = [a for a in cfg["arms"] if not args.arms or a in args.arms]
    jobs = [(a, q, r) for a in arms for q in qs for r in range(args.repeats or cfg["repeats"])]
    random.Random(7).shuffle(jobs)                      # interleave arms so load and time of day spread evenly
    return jobs


def run_one(cfg, arm, q, rep):
    log = cfg["out"] / "runs" / arm.replace("/", "_") / f"{q['id']}_r{rep}.jsonl"
    if log.exists() and log.stat().st_size:
        return None
    log.parent.mkdir(parents=True, exist_ok=True)
    spec = cfg["arms"][arm]
    cmd = ["claude", "-p", q["q"], "--append-system-prompt", spec.get("instruction", cfg["instruction"]),
           "--output-format", "stream-json", "--verbose", "--model", spec.get("model", cfg["model"]),
           "--effort", cfg["effort"], "--max-turns", str(cfg["max_turns"]),
           "--max-budget-usd", str(cfg["max_usd_per_run"]), "--strict-mcp-config", "--no-session-persistence"]
    allowed, denied = READ_TOOLS, DENIED
    d = spec.get("defrost")
    mcp = {"mcpServers": {}}
    if d and d.get("mcp", True):
        mcp["mcpServers"]["defrost"] = {"command": "defrost", "args": ["mcp"], "env": {"DEFROST_DOMAIN": d["domain"]}}
        allowed += ",mcp__defrost__search,mcp__defrost__docs_for"
        denied += ",mcp__defrost__remember,mcp__defrost__refresh"   # the memory stays as built
    cmd += ["--mcp-config", json.dumps(mcp)]
    if not (d and d.get("hook", True)):
        cmd += ["--settings", json.dumps({"disableAllHooks": True})]
    if spec.get("skills"):
        allowed += ",Skill"
    else:
        denied += ",Skill"
        cmd += ["--disable-slash-commands"]
    cmd += ["--allowedTools", allowed, "--disallowedTools", denied]
    t0 = time.time()
    # Own process group: on timeout the whole group (claude, its MCP server, hooks) is killed, so a child holding the
    # output pipe cannot keep the batch waiting past the limit.
    proc = subprocess.Popen(cmd, cwd=paths(cfg)[arm], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=arm_env(cfg, arm), stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=900)
    except subprocess.TimeoutExpired:                   # a hung session: record it and go on; a rerun retries it
        os.killpg(proc.pid, signal.SIGKILL) if hasattr(os, "killpg") else proc.kill()
        proc.communicate()
        log.with_suffix(".stderr").write_text("timeout after 900 s\n")
        return time.time() - t0
    p = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    log.write_text(p.stdout)
    if p.returncode:
        log.with_suffix(".stderr").write_text(p.stderr)
    return time.time() - t0


def run(cfg, args):
    jobs = plan(cfg, args)
    print(f"{len(jobs)} runs planned; estimate ${len(jobs) * cfg['est_usd_per_run']:.0f}, "
          f"hard cap ${len(jobs) * cfg['max_usd_per_run']:.0f}")
    if args.cmd == "estimate" or not args.yes:
        return print("Not run: add --yes once the budget is confirmed.") if args.cmd != "estimate" else None
    for i, (arm, q, rep) in enumerate(jobs, 1):
        dt = run_one(cfg, arm, q, rep)
        print(f"[{i}/{len(jobs)}] {arm} {q['id']} r{rep} " + ("skipped (logged)" if dt is None else f"{dt:.0f}s"),
              flush=True)
    collect(cfg)


PATH_RE = re.compile(r"(?:^|[\s'\"(])((?:\.?[\w.-]+/)+[\w.-]+\.\w+)")


def parse(log: Path, root: str) -> dict:
    """One row per run: answer, usage, cost, latency, tool calls and distinct files read."""
    roots = (root.rstrip("/") + "/", "/private" + root.rstrip("/") + "/")
    rel = lambda f: next((f[len(r):] for r in roots if f.startswith(r)), f)
    row = {"tools": {}, "files": set(), "answer": "", "error": None}
    for line in log.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for c in ev.get("message", {}).get("content", []):
                if c.get("type") != "tool_use":
                    continue
                row["tools"][c["name"]] = row["tools"].get(c["name"], 0) + 1
                inp = c.get("input", {})
                if c["name"] == "Read" and inp.get("file_path"):
                    row["files"].add(rel(inp["file_path"]))
                elif c["name"] == "Bash" and re.match(r"\s*(cat|head|sed|wc)\b", inp.get("command", "")):
                    row["files"].update(rel(m) for m in PATH_RE.findall(inp["command"]))
        elif ev.get("type") == "result":
            u = ev.get("usage", {})
            row.update(answer=ev.get("result", ""), duration_ms=ev.get("duration_ms"), cost_usd=ev.get("total_cost_usd"),
                       turns=ev.get("num_turns"), input_tokens=u.get("input_tokens", 0),
                       output_tokens=u.get("output_tokens", 0), cache_read=u.get("cache_read_input_tokens", 0),
                       cache_write=u.get("cache_creation_input_tokens", 0),
                       error=ev.get("subtype") if ev.get("is_error") else None)
    row["tool_calls"] = sum(row["tools"].values())
    row["files"] = sorted(row["files"])
    return row


def collect(cfg, _=None):
    rows = []
    for log in sorted((cfg["out"] / "runs").glob("*/*.jsonl")):
        arm = log.parent.name.replace("_", "/", 1)
        if arm not in cfg["arms"]:
            continue
        qid, rep = log.stem.split("_r")
        rows.append({"arm": arm, "qid": qid, "rep": int(rep), **parse(log, paths(cfg)[arm])})
    (cfg["out"] / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} runs -> {cfg['out'] / 'rows.jsonl'}")


# ---- judge -----------------------------------------------------------------------------------------------------------
JUDGE = """You grade an answer to a question about a software project. The ground truth was taken from the code
and git history, not from the docs; the docs may be stale. Doc file names can differ between project snapshots of
the same docs; treat either naming as valid.

Question: {q}
Ground truth: {truth}
Evidence: {evidence}
Key points (ALL needed for correctness 2; some of them = 1; none or a wrong main claim = 0): {key}
Optional points (never required; do not lower the score when missing): {optional}
Known stale claims (wrong today): {stale}
Extra true details beyond the ground truth are not errors. Only call something a hallucination if it is false per
the ground truth and evidence, not merely absent from them.

Answer to grade:
<<<
{answer}
>>>

Return only a JSON object:
{{"correctness": 0|1|2,        // 2 = matches the ground truth in substance, 1 = partly right or missing a key point, 0 = wrong or no answer
 "flags_conflict": true|false, // says that sources disagree, or that a doc is stale / differs from the code
 "uses_stale_fact": true|false,// states a known stale claim as current
 "hallucination": true|false,  // names a file, route, setting, decision or number that does not exist per the truth
 "cites_source": true|false,   // names at least one file it relied on
 "why": "<one sentence>"}}"""


def ask_judge(cfg, q, row):
    prompt = JUDGE.format(q=q["q"], truth=q["truth"], evidence="; ".join(q.get("evidence", [])),
                          key="; ".join(q.get("key_points", [])) or "(the ground truth as a whole)",
                          optional="; ".join(q.get("optional_points", [])) or "none",
                          stale="; ".join(q.get("stale", [])) or "none", answer=row["answer"] or "(no answer)")
    with tempfile.TemporaryDirectory() as empty:        # nothing to read: the judge only grades
        p = subprocess.run(["claude", "-p", prompt, "--model", cfg["judge_model"], "--output-format", "json",
                            "--max-turns", "1", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                            "--settings", json.dumps({"disableAllHooks": True}), "--disallowedTools",
                            "Bash,Read,Grep,Glob,Edit,Write,WebFetch,WebSearch,Skill,Task,Agent",
                            "--no-session-persistence"], cwd=empty, capture_output=True, text=True, timeout=300)
    out = json.loads(p.stdout)
    if isinstance(out, list):                           # newer CLIs print the event list; the result is the last event
        out = next(e for e in reversed(out) if e.get("type") == "result")
    text = out["result"].strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    verdict = json.loads(text[text.index("{"): text.rindex("}") + 1])
    verdict["judge_cost_usd"] = out.get("total_cost_usd")
    return verdict


def load_rows(cfg):
    return [json.loads(line) for line in (cfg["out"] / "rows.jsonl").read_text().splitlines()]


def load_scores(cfg):
    f = cfg["out"] / "scores.jsonl"
    return {(s["arm"], s["qid"], s["rep"]): s for s in map(json.loads, f.read_text().splitlines())} if f.exists() else {}


def judge(cfg, args):
    qs = {q["id"]: q for q in questions(cfg)}
    done = load_scores(cfg)
    todo = [r for r in load_rows(cfg) if (r["arm"], r["qid"], r["rep"]) not in done]
    print(f"{len(todo)} answers to judge (about ${len(todo) * 0.035:.0f})")
    if not args.yes:
        return print("Not run: add --yes once the budget is confirmed.")
    random.Random(3).shuffle(todo)                      # judge order is not grouped by arm
    with (cfg["out"] / "scores.jsonl").open("a") as f:
        for r in todo:
            v = ask_judge(cfg, qs[r["qid"]], r)
            f.write(json.dumps({"arm": r["arm"], "qid": r["qid"], "rep": r["rep"], **v}) + "\n")
            f.flush()


def sample(cfg, _):
    qs = {q["id"]: q for q in questions(cfg)}
    with (cfg["out"] / "calibration.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "question", "truth", "answer", *FIELDS])
        for r in random.Random(5).sample(load_rows(cfg), 10):
            w.writerow([f"{r['arm']}|{r['qid']}|{r['rep']}", qs[r["qid"]]["q"], qs[r["qid"]]["truth"], r["answer"],
                        *[""] * len(FIELDS)])
    print(f"Fill the last {len(FIELDS)} columns by hand in {cfg['out'] / 'calibration.csv'}, hiding the key column")


def agree(cfg, _):
    judged = {"|".join(map(str, k)): s for k, s in load_scores(cfg).items()}
    rows = [r for r in csv.DictReader((cfg["out"] / "calibration.csv").open()) if r["correctness"] != ""]
    rep = {"n": len(rows)}
    for field in FIELDS:
        pairs = [(str(r[field]).strip().lower(), str(judged[r["key"]][field]).lower()) for r in rows if r["key"] in judged]
        rep[field] = round(sum(a == b for a, b in pairs) / len(pairs), 2) if pairs else None
    (cfg["out"] / "agreement.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# ---- report ----------------------------------------------------------------------------------------------------------
def report(cfg, args):
    qs = {q["id"]: q for q in questions(cfg)}
    skip = set(args.exclude or [])
    rows = {(r["arm"], r["qid"], r["rep"]): r for r in load_rows(cfg) if r["qid"] not in skip}
    scores = load_scores(cfg)
    arms = [a for a in cfg["arms"] if any(k[0] == a for k in rows)]
    items = {a: [(rows[k], scores.get(k)) for k in rows if k[0] == a] for a in arms}
    js = lambda a, kinds=None: [s for r, s in items[a] if s and (not kinds or qs[r["qid"]]["kind"] in kinds)]
    mean = lambda xs: st.mean(xs) if xs else None
    pctl = lambda xs, q: sorted(xs)[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None
    metrics = [
        ("runs", lambda a: len(items[a]), "{:.0f}"),
        ("correctness, mean (0-2)", lambda a: mean([s["correctness"] for s in js(a)]), "{:.2f}"),
        ("fully correct, %", lambda a: mean([100 * (s["correctness"] == 2) for s in js(a)]), "{:.0f}"),
        ("wrong (0), %", lambda a: mean([100 * (s["correctness"] == 0) for s in js(a)]), "{:.0f}"),
        ("conflict flagged on conflict traps, %", lambda a: mean([100 * s["flags_conflict"] for s in js(a, {"conflict-trap"})]), "{:.0f}"),
        ("stale fact stated as current, %", lambda a: mean([100 * s["uses_stale_fact"] for s in js(a, {"stale-trap"})]), "{:.0f}"),
        ("hallucination, %", lambda a: mean([100 * s["hallucination"] for s in js(a)]), "{:.0f}"),
        ("cites a source, %", lambda a: mean([100 * s["cites_source"] for s in js(a)]), "{:.0f}"),
        ("answered in 1 turn, %", lambda a: mean([100 * (r.get("turns") == 1) for r, _ in items[a]]), "{:.0f}"),
        ("turns, median", lambda a: st.median([r.get("turns") or 0 for r, _ in items[a]]), "{:.0f}"),
        ("tool calls, median", lambda a: st.median([r["tool_calls"] for r, _ in items[a]]), "{:.0f}"),
        ("latency median, s", lambda a: st.median([r["duration_ms"] / 1000 for r, _ in items[a] if r.get("duration_ms")]), "{:.0f}"),
        ("latency p90, s", lambda a: pctl([r["duration_ms"] / 1000 for r, _ in items[a] if r.get("duration_ms")], 0.9), "{:.0f}"),
        ("cost per run, mean USD", lambda a: mean([r.get("cost_usd") or 0 for r, _ in items[a]]), "{:.3f}"),
        ("cost total, USD", lambda a: sum(r.get("cost_usd") or 0 for r, _ in items[a]), "{:.2f}"),
    ]
    lines = [f"# Agent QA report ({time.strftime('%Y-%m-%d')})", "",
             f"Excluded questions: {', '.join(sorted(skip)) or 'none'}", "",
             "| metric | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for name, fn, fmt in metrics:
        vals = [fn(a) for a in arms]
        lines.append(f"| {name} | " + " | ".join("–" if v is None else fmt.format(v) for v in vals) + " |")
    if len(arms) >= 2:                                  # paired sign test: each arm against the first
        lines += ["", "| paired against " + arms[0] + " | better | worse | same | sign test p |", "|---|---|---|---|---|"]
        for a in arms[1:]:
            w = l = t = 0
            for (arm, qid, rep), s in scores.items():
                b = scores.get((a, qid, rep))
                if arm != arms[0] or qid in skip or not b:
                    continue
                w += b["correctness"] > s["correctness"]; l += b["correctness"] < s["correctness"]
                t += b["correctness"] == s["correctness"]
            n = w + l
            p = min(1.0, 2 * sum(math.comb(n, i) for i in range(max(w, l), n + 1)) / 2 ** n) if n else 1.0
            lines.append(f"| {a} | {w} | {l} | {t} | {p:.3f} |")
    lines += ["", "| question | kind | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    for qid in sorted({k[1] for k in rows}):
        cells = []
        for a in arms:
            c = [scores[k]["correctness"] for k in rows if k[0] == a and k[1] == qid and k in scores]
            cells.append(f"{st.mean(c):.1f} ({max(c) - min(c)})" if c else "–")
        lines.append(f"| {qid} | {qs[qid]['kind']} | " + " | ".join(cells) + " |")
    (cfg["out"] / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def teardown(cfg, _):
    for path in paths(cfg).values():
        subprocess.run(["git", "-C", cfg["repo"], "worktree", "remove", "--force", path])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("cmd", choices=["setup", "memory", "hooks", "estimate", "run", "collect", "judge", "sample", "agree",
                                    "report", "teardown"])
    ap.add_argument("--yes", action="store_true", help="confirm spending; without it nothing is run")
    ap.add_argument("--only", nargs="*", help="question ids")
    ap.add_argument("--arms", nargs="*", help="arm names")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--exclude", nargs="*", help="question ids left out of the report (e.g. a wrong ground truth)")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    fn = {"setup": setup, "memory": memory, "hooks": hooks, "estimate": run, "run": run, "collect": collect,
          "judge": judge, "sample": sample, "agree": agree, "report": report, "teardown": teardown}[args.cmd]
    return fn(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
