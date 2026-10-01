"""Shepherd integration: memory updates and memory-grounded agent work as reviewable, revertible Shepherd runs.

Shepherd (shepherd-agents/shepherd) runs agent tasks in an OS sandbox and keeps their results as retained outputs:
nothing reaches your files until you `shepherd run select` it. This module uses that for two things:

  refresh DOMAIN      rebuild the domain's memory (incremental), then record the snapshot (sources + commits,
                      counts, changed documents) in memory/<domain>.snapshot.json as a retained output.
                      select = accept the new memory; discard (via `reject`) = restore the previous build.
  ask "QUESTION"      retrieve cited context from the memory (fast policy) and give it to a sandboxed Claude agent,
                      which writes a cited answer to answers/<slug>.md as a retained output.

Run from a Shepherd workspace (`shepherd init`); needs `pip install shepherd-ai` and a running or startable
defrost service (see defrost_ai.service.client).

    python -m defrost_ai.integrations.shepherd refresh heldout
    python -m defrost_ai.integrations.shepherd reject <run_ref> heldout
    python -m defrost_ai.integrations.shepherd ask "How do I make only some fields optional?" --domain heldout"""
from __future__ import annotations

import argparse
import json
import re
import sys

import shepherd as sp

from defrost_ai.service import client


@sp.task
def record_memory_snapshot(repo: sp.GitRepo, topic: str, output_path: str, output_text: str) -> None:
    """Record a knowledge-memory snapshot (domain, source commits, counts, changed documents) as a reviewable file."""


@sp.task
def answer_from_memory(repo: sp.GitRepo, question: str, context: str, output_path: str = "answers/answer.md") -> None:
    """Answer `question` for the team using only `context`: sections retrieved from the company knowledge memory,
    each numbered [n] and cited as domain:path:Lstart-end, some followed by the code they name.

    Write a concise markdown answer to output_path. Cite every claim with the [n] of the section it comes from and
    end with a "Sources" list of the citations you used. If the context does not contain the answer, say so plainly
    and list what is missing; do not guess. Do not create or edit any other file."""


def _slug(text: str, n: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "answer"


def refresh(domain: str, workspace_dir: str = ".") -> dict:
    state = client.update(domain, wait=True)
    if state.get("state") != "done":
        return {"state": state.get("state"), "error": state.get("error")}
    m = state["manifest"]
    snapshot = {"domain": domain, "built_at": m["built_at"], "encoder": m["encoder"], "counts": m["counts"],
                "sources": m["sources"], "changed": m["changed"], "build_seconds": m["build_seconds"]}
    with sp.open(workspace_dir) as ws:
        ws.tasks.register(record_memory_snapshot)
        run = ws.run(record_memory_snapshot, repo=ws.git_repo(), topic=f"memory:{domain}",
                     output_path=f"memory/{domain}.snapshot.json", output_text=json.dumps(snapshot, indent=1) + "\n",
                     placement="advisory", runtime={"provider": "static"})
        changed = list(run.output().changeset().changed_paths)
    return {"run_ref": run.run_ref, "changed_paths": changed, "counts": m["counts"], "changed_docs": m["changed"]["n"],
            "next": [f"shepherd run select {run.run_ref}   # accept the new memory",
                     f"python -m defrost_ai.integrations.shepherd reject {run.run_ref} {domain}   # restore previous"]}


def reject(run_ref: str, domain: str, workspace_dir: str = ".") -> dict:
    import shutil
    import subprocess
    cmd = [shutil.which("shepherd") or "shepherd", "run", "discard", run_ref]
    r = subprocess.run(cmd, cwd=workspace_dir, capture_output=True, text=True)
    return {"discarded": r.returncode == 0, "shepherd": (r.stdout or r.stderr).strip()[-400:],
            "rollback": client.rollback(domain)}


def ask(question: str, domains=None, mode: str = "fast", k: int = 5, workspace_dir: str = ".") -> dict:
    res = client.search(question, domains, mode, k, context=True)
    out_path = f"answers/{_slug(question)}.md"
    with sp.open(workspace_dir) as ws:
        ws.tasks.register(answer_from_memory)
        run = ws.run(answer_from_memory, repo=ws.git_repo(), question=question, context=res["context"],
                     output_path=out_path, placement="jail", runtime={"provider": "claude"})
        output = run.output()
        changed = list(output.changeset().changed_paths)
        preview = output.read_text(out_path) if out_path in changed else None
    return {"run_ref": run.run_ref, "status": run.status, "mode_used": res["mode_used"],
            "cited": [f"{h['domain']}:{h['path']}:L{h['lines'][0]}-{h['lines'][1]}" for h in res["hits"]],
            "changed_paths": changed, "answer_preview": preview,
            "next": [f"shepherd run select {run.run_ref}", f"shepherd run discard {run.run_ref}"]}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m defrost_ai.integrations.shepherd")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refresh"); r.add_argument("domain")
    j = sub.add_parser("reject"); j.add_argument("run_ref"); j.add_argument("domain")
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("--domain", action="append")
    a.add_argument("--mode", default="fast"); a.add_argument("-k", type=int, default=5)
    args = ap.parse_args(argv)
    if args.cmd == "refresh":
        out = refresh(args.domain)
    elif args.cmd == "reject":
        out = reject(args.run_ref, args.domain)
    else:
        out = ask(args.question, args.domain, args.mode, args.k)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
