"""Clone the benchmark sources at the pinned commits and write a workspace.json for each benchmark.

    python scripts/fetch_benchmark_sources.py            # -> benchmarks/.sources/, benchmarks/*/workspace.json

The gold spans in benchmarks/*/questions.jsonl are (path, line range) pairs in these exact commits."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "benchmarks/.sources"
REPOS = {
    "jinja": ("https://github.com/pallets/jinja.git", "5ef70112a1ff19c05324ff889dd30405b1002044"),
    "werkzeug": ("https://github.com/pallets/werkzeug.git", "fa93b5cee027bc9baadba67f21360a8a39b2317b"),
    "marshmallow": ("https://github.com/marshmallow-code/marshmallow.git", "7f0792bd7a06f72e393ec866ac1e89e1502ccfe1"),
    "progit2": ("https://github.com/progit/progit2", "a013e3230a1207cfa5ae94d28ba7d2021063c337"),
    "Eloquent-JavaScript": ("https://github.com/marijnh/Eloquent-JavaScript", "4a58fbd76ef38763f2f3d850621c80295ccb0361"),
    "aosabook": ("https://github.com/aosabook/aosabook", "12308deff812eb3a4fa4e210ae5496e154961311"),
}
WORKSPACES = {
    "heldout": {"name": "heldout", "components": [
        {"name": "jinja", "path": "jinja", "role": "template engine"},
        {"name": "werkzeug", "path": "werkzeug", "role": "WSGI toolkit"},
        {"name": "marshmallow", "path": "marshmallow", "role": "serialization library"}]},
    "books": {"name": "books", "components": [
        {"name": "progit", "path": "progit2/book", "role": "Pro Git 2nd ed. (CC BY-NC-SA 3.0)", "text_only": True},
        {"name": "eloquentjs", "path": "Eloquent-JavaScript", "role": "Eloquent JavaScript (CC BY-NC)", "text_only": True,
         "exclude": ["README.md"]},
        {"name": "500lines", "path": "aosabook/src/500lines", "role": "500 Lines or Less (CC BY 3.0)", "text_only": True,
         "exclude": ["README.md", "LICENSE.md", "BUILD.md"]}]},
}


def main():
    SOURCES.mkdir(parents=True, exist_ok=True)
    for name, (url, commit) in REPOS.items():
        d = SOURCES / name
        if not d.exists():
            subprocess.run(["git", "clone", "--quiet", url, str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "--quiet", commit], check=True)
        print(f"{name} @ {commit[:10]}")
    for bench, ws in WORKSPACES.items():
        spec = {**ws, "out": f"~/.kev-memory/benchmark-{bench}",
                "components": [{**c, "path": str(SOURCES / c["path"])} for c in ws["components"]]}
        (ROOT / f"benchmarks/{bench}/workspace.json").write_text(json.dumps(spec, indent=1))
        print(f"-> benchmarks/{bench}/workspace.json")


if __name__ == "__main__":
    main()
