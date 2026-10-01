"""Kaggle bundle for P2b (supervised retrieval stage): a private dataset + a notebook that runs train_sup.py on 2x T4.

    uv run python -m defrost_graph.bilm.kaggle.sup_bundle --owner <kaggle-user> [--arm A] [--steps 1000]
    kaggle datasets create -p defrost_graph/data/kaggle/kev-sup-v1 --dir-mode zip      # first time (version: datasets version)
    kaggle kernels push -p defrost_graph/data/kaggle/kernel-sup

Arms (plan S2): A = MNTP init, all data | B = MNTP+CGSA init, all data | C = MNTP init, public data only."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SLUG = "kev-sup-v1"
CODE = ["defrost_graph/__init__.py", "defrost_graph/bilm/__init__.py", "defrost_graph/bilm/bridge.py",
        "defrost_graph/bilm/kaggle/__init__.py", "defrost_graph/bilm/kaggle/train_sup.py"]
DATA = ["train.jsonl", "val.jsonl", "manifest.json"]
ADAPTERS = {"mntp-adapter": ROOT / "runs/defrost_graph/techdoc-mntp", "cgsa-adapter": ROOT / "runs/defrost_graph/techdoc-cgsa"}
ARMS = {"A": {"cgsa": False, "sources": ""}, "B": {"cgsa": True, "sources": ""},
        "C": {"cgsa": False, "sources": "msmarco,nq,hotpotqa,stackexchange,allnli,quora"}}

NOTEBOOK = '''import glob, json, os, subprocess, sys, time, zipfile
from pathlib import Path
ARM, STEPS, EXTRA = "{arm}", {steps}, "{extra}"

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

man = [Path(p) for p in glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)
       if (Path(p).parent / "code/defrost_graph/bilm/kaggle/train_sup.py").exists()]
assert man, "attach the private dataset {slug}"
D = man[0].parent
sh(f"{{sys.executable}} -m pip install -q 'transformers==5.17.0' 'peft==0.21.0' 'safetensors>=0.4.3'")
sh(f"{{sys.executable}} -m pip uninstall -y -q torchao")
sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")
OUT = Path(f"/kaggle/working/sup-{{ARM}}")
cgsa = f"--cgsa {{D}}/cgsa-adapter" if {cgsa} else ""
src = "--sources {sources}" if "{sources}" else ""
t0 = time.time()
sh(f"cd {{D}}/code && PYTHONUNBUFFERED=1 torchrun --nproc_per_node 2 defrost_graph/bilm/kaggle/train_sup.py "
   f"--data {{D}}/data --mntp {{D}}/mntp-adapter {{cgsa}} {{src}} --steps {{STEPS}} --out {{OUT}} {{EXTRA}} 2>&1 | tee {{OUT}}.log")
print(f"done in {{(time.time() - t0) / 60:.0f}} min")
with zipfile.ZipFile(f"/kaggle/working/sup-{{ARM}}.zip", "w") as z:
    for p in OUT.rglob("*"):
        if p.is_file() and "last" not in p.parts:
            z.write(p, f"sup-{{ARM}}/{{p.relative_to(OUT)}}")
    z.write(f"{{OUT}}.log", f"sup-{{ARM}}/train.log")
'''


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="<kaggle-user>")
    ap.add_argument("--arm", default="A", choices=sorted(ARMS))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--extra", default="", help="extra train_sup.py flags, e.g. '--max_doc 512'")
    ap.add_argument("--no_data", action="store_true", help="only rewrite the notebook")
    args = ap.parse_args(argv)
    base = ROOT / "defrost_graph/data/kaggle"
    out = base / SLUG
    if not args.no_data:
        if out.exists():
            shutil.rmtree(out)
        copies = {f"code/{c}": ROOT / c for c in CODE}
        copies.update({f"data/{n}": ROOT / "defrost_graph/data/sup_v1" / n for n in DATA})
        for name, d in ADAPTERS.items():
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                copies[f"{name}/{f}"] = d / f
        for rel, src in copies.items():
            if not src.exists():
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                (out / rel).write_text("") if rel.endswith("__init__.py") else (_ for _ in ()).throw(FileNotFoundError(src))
                continue
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / rel)
        man = {"files": {rel: {"sha256": sha256(out / rel), "bytes": (out / rel).stat().st_size} for rel in sorted(copies)}}
        (out / "MANIFEST.json").write_text(json.dumps(man, indent=2))
        (out / "dataset-metadata.json").write_text(json.dumps(
            {"title": SLUG, "id": f"{args.owner}/{SLUG}", "isPrivate": True, "licenses": [{"name": "other"}]}, indent=2))
        print(f"dataset -> {out} ({sum(v['bytes'] for v in man['files'].values()) / 1e6:.0f} MB)")
    arm = ARMS[args.arm]
    code = NOTEBOOK.format(arm=args.arm, steps=args.steps, extra=args.extra, slug=SLUG, cgsa=arm["cgsa"],
                           sources=arm["sources"])
    kdir = base / "kernel-sup"
    kdir.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                     "source": code.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (kdir / "defrost-sup.ipynb").write_text(json.dumps(nb, indent=1))
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.owner}/kev-sup-{args.arm.lower()}", "title": f"kev-sup-{args.arm.lower()}",
        "code_file": "defrost-sup.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "dataset_sources": [f"{args.owner}/{SLUG}"],
        "kernel_sources": [], "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}, indent=2))
    print(f"notebook -> {kdir} (arm {args.arm}, {args.steps} steps)")


if __name__ == "__main__":
    main()
