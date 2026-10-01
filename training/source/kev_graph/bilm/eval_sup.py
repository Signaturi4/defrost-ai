"""P2b S3: score Kev-Ret checkpoints on the frozen suites with the same protocol as the baselines.

    uv run --with ranx python -m kev_graph.bilm.eval_sup --adapters runs/kev_graph/sup-C/step-400,runs/kev_graph/sup-C/final \\
        --split dev

For each adapter: embed the held-out and client text KBs (sections + passages), then bench.py systems sec-dense,
sec-hybrid, pas-dense, pas-hybrid (+ BM25 v1 as the paired baseline). --split test is the one-time final scoring.
Writes runs/kev_graph/sup_eval/<split>/<name>__<suite>.json and prints one line per (adapter, suite)."""
import argparse
import json
from pathlib import Path

WS = {"heldout": "kev_graph/memory/workspaces/heldout.json", "client": "kev_graph/memory/workspaces/client.json"}
BENCH = {"heldout": "kev_graph/memory/evals/text_heldout_v1.jsonl", "client": "kev_graph/memory/evals/text_client_v1.jsonl"}
SYSTEMS = "sec-bm25,sec-dense,sec-hybrid,pas-dense,pas-hybrid"


def main(argv=None):
    from kev_graph.memory import bench, embed, layers
    from kev_graph.memory.embed import vectors_file
    from kev_graph.memory.workspace import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True, help="comma list of adapter dirs, or encoder names (bge-base, cgsa)")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--suites", default="heldout,client")
    args = ap.parse_args(argv)
    out = Path("runs/kev_graph/sup_eval") / args.split
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for a in args.adapters.split(","):
        enc = a if a in ("cgsa", "bge-small", "bge-base") else f"kevret:{a}"
        name = a.replace("/", "_") if enc.startswith("kevret:") else a
        for suite in args.suites.split(","):
            o = load(WS[suite])["out"]
            if not (o / vectors_file(enc)).exists():
                embed.main(["--workspace", WS[suite], "--encoder", enc, "--no_nodes"])
            if not (o / vectors_file(enc).replace("vectors", "layers")).exists():
                layers.main(["--workspace", WS[suite], "--encoder", enc])
            f = out / f"{name}__{suite}.json"
            bench.main(["--workspace", WS[suite], "--bench", BENCH[suite], "--split", args.split, "--encoder", enc,
                        "--systems", SYSTEMS, "--out", str(f)])
            t = json.loads(f.read_text())["table"]
            summary[(name, suite)] = {s: round(t[s]["ndcg@10"], 3) for s in t if "ndcg@10" in t[s]}
    print("\nnDCG@10 summary")
    for (n, s), v in summary.items():
        print(f"{n:50s} {s:8s} {v}")


if __name__ == "__main__":
    main()
