"""Assemble the private Kaggle dataset for the CGSA stage.

    uv run python -m kev_graph.bilm.kaggle.build_dataset [--slug <dataset name>] [--owner <kaggle user>]

Output: kev_graph/data/kaggle/<slug>/ (the dataset) and kev_graph/data/kaggle/<slug>.ipynb (the notebook).

Layout (what the notebook expects under /kaggle/input/<slug>/):

    corpus/corpus_cgsa.txt       272k sentences, the CGSA training set (techdoc_corpus.py)
    corpus/corpus_heldout.txt    held-out docs, used only by quick_check.py
    mntp-adapter/                final MNTP LoRA adapter (runs/kev_graph/techdoc-mntp, no checkpoints)
    code/experiments/run_cgsa.py KG-BiLM reference script (train_cgsa_ddp.py imports its dataclasses)
    code/train_cgsa_ddp.py, code/quick_check.py, code/cgsa_kaggle.json
    MANIFEST.json                sha256 of every file, checked by the notebook before training
    dataset-metadata.json        private, for `kaggle datasets create`

The notebook is generated for the same slug (make_notebook.write) next to the folder, not into it.

The corpus contains scrubbed local docs, so the dataset must stay private. raw/ is never copied.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from kev_graph.bilm.kaggle import make_notebook

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CORPUS = ROOT / "kev_graph/data/techdoc_mega/corpus"
MNTP = ROOT / "runs/kev_graph/techdoc-mntp"
REF = ROOT / "docs/jev_for_graph/research/kg-bilm-main/experiments/run_cgsa.py"
ADAPTER_FILES = [
    "adapter_config.json", "adapter_model.safetensors", "added_tokens.json", "merges.txt",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=make_notebook.DEFAULT_SLUG, help="Kaggle dataset name (lowercase, dashes)")
    ap.add_argument("--owner", default="YOUR_KAGGLE_USERNAME")
    args = ap.parse_args(argv)
    out = ROOT / "kev_graph/data/kaggle" / args.slug
    if out.exists():
        shutil.rmtree(out)

    copies = {f"corpus/{n}": CORPUS / n for n in ("corpus_cgsa.txt", "corpus_heldout.txt")}
    copies.update({f"mntp-adapter/{n}": MNTP / n for n in ADAPTER_FILES})
    copies["code/experiments/run_cgsa.py"] = REF
    copies.update({f"code/{n}": HERE / n for n in ("train_cgsa_ddp.py", "quick_check.py", "cgsa_kaggle.json")})
    for rel, src in copies.items():
        if not src.exists():
            raise FileNotFoundError(src)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / rel)

    manifest = {
        "files": {rel: {"sha256": sha256(out / rel), "bytes": (out / rel).stat().st_size} for rel in sorted(copies)},
        "mntp_source": str(MNTP.relative_to(ROOT)),
        "mntp_results": json.loads((MNTP / "all_results.json").read_text()),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": args.slug,
        "id": f"{args.owner}/{args.slug}",
        "isPrivate": True,
        "licenses": [{"name": "other"}],
    }, indent=2))
    notebook = make_notebook.write(args.slug, out.parent / f"{args.slug}.ipynb")  # uploaded separately
    total = sum(v["bytes"] for v in manifest["files"].values())
    print(f"wrote {out} ({len(copies)} files, {total / 1e6:.1f} MB) and {notebook}")


if __name__ == "__main__":
    main()
