"""E1 (Defrost-Ret-B2) Kaggle bundle: dense-negative mining + a 3000-step warm-started run, chainable across sessions.

    uv run python -m defrost_graph.bilm.kaggle.e1_bundle            # writes dataset + notebook
    kaggle datasets create -p defrost_graph/data/kaggle/kev-e1-v1 --dir-mode zip
    kaggle kernels push -p defrost_graph/data/kaggle/kernel-e1      # session 1
    uv run python -m defrost_graph.bilm.kaggle.e1_bundle --chain    # session 2+: adds the previous output as an input

The notebook: (1) if a previous session's output is attached, copy its mined data and last/ checkpoint; else mine dense
negatives with Defrost-Ret-B on GPU 0; (2) torchrun train_sup.py on both GPUs, --init_adapter Defrost-Ret-B, resuming from
last/ when present, stopping cleanly at --time_budget 11.3 h."""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SLUG, KERNEL = "kev-e1-v1", "kev-sup-e1"
CODE = ["defrost_graph/__init__.py", "defrost_graph/bilm/__init__.py", "defrost_graph/bilm/bridge.py",
        "defrost_graph/bilm/kaggle/__init__.py", "defrost_graph/bilm/kaggle/train_sup.py",
        "defrost_graph/bilm/kaggle/eval_public_sup.py", "defrost_graph/bilm/kaggle/mine_hard.py"]

NOTEBOOK = r'''import glob, shutil, subprocess, sys, time
from pathlib import Path
STEPS, BUDGET = {steps}, 11.3
def sh(c):
    print("$", c, flush=True); subprocess.run(c, shell=True, check=True)
D = [Path(p).parent for p in glob.glob("/kaggle/input/**/MANIFEST_E1", recursive=True)][0]
sh(f"{{sys.executable}} -m pip install -q 'transformers==5.17.0' 'peft==0.21.0' 'safetensors>=0.4.3'")
sh(f"{{sys.executable}} -m pip uninstall -y -q torchao")
W = Path("/kaggle/working"); OUT = W / "sup-E1"; DATA = W / "data"; DATA.mkdir(exist_ok=True)
shutil.copy(D / "data/val_v2.jsonl", DATA / "val.jsonl")
prev = [Path(p).parent for p in glob.glob("/kaggle/input/**/sup-E1/last/state.pt", recursive=True)]
mined = [Path(p) for p in glob.glob("/kaggle/input/**/data/train.jsonl", recursive=True) if "kev-e1-v1" not in p]
if prev and mined:
    print("chained session: resuming from", prev[0], flush=True)
    shutil.copytree(prev[0], OUT / "last")
    shutil.copy(mined[0], DATA / "train.jsonl")
    for f in prev[0].parent.glob("step-*"):
        shutil.copytree(f, OUT / f.name, dirs_exist_ok=True)
else:
    t0 = time.time()
    sh(f"cd {{D}}/code && PYTHONUNBUFFERED=1 {{sys.executable}} defrost_graph/bilm/kaggle/mine_hard.py --data {{D}}/data "
       f"--adapter {{D}}/sup-B --mntp {{D}}/mntp-adapter --cgsa {{D}}/cgsa-adapter --out {{DATA}}/train.jsonl")
    print(f"mined in {{(time.time() - t0) / 60:.0f}} min", flush=True)
sh(f"cd {{D}}/code && PYTHONUNBUFFERED=1 torchrun --nproc_per_node 2 defrost_graph/bilm/kaggle/train_sup.py --data {{DATA}} "
   f"--mntp {{D}}/mntp-adapter --cgsa {{D}}/cgsa-adapter --init_adapter {{D}}/sup-B --steps {{STEPS}} --lr 1e-4 "
   f"--max_doc 512 --save_every 250 --eval_every 250 --time_budget {{BUDGET}} --out {{OUT}} 2>&1 | tee -a {{W}}/e1.log")
'''


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="<kaggle-user>")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--chain", action="store_true", help="attach the previous session's output (kernel_sources)")
    ap.add_argument("--no_data", action="store_true")
    args = ap.parse_args(argv)
    base = ROOT / "defrost_graph/data/kaggle"
    out = base / SLUG
    if not args.no_data and not args.chain:
        if out.exists():
            shutil.rmtree(out)
        for c in CODE:
            (out / "code" / c).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / c, out / "code" / c)
        (out / "data").mkdir(parents=True)
        for f in ("train_v2.jsonl", "val_v2.jsonl", "manifest_v2.json"):
            shutil.copy2(ROOT / "defrost_graph/data/sup_v1" / f, out / "data" / f)
        for name, src in {"mntp-adapter": "runs/defrost_graph/techdoc-mntp", "cgsa-adapter": "runs/defrost_graph/techdoc-cgsa",
                          "sup-B": "runs/defrost_graph/sup-B/final"}.items():
            (out / name).mkdir()
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                shutil.copy2(ROOT / src / f, out / name / f)
        (out / "MANIFEST_E1").write_text("")
        (out / "dataset-metadata.json").write_text(json.dumps(
            {"title": SLUG, "id": f"{args.owner}/{SLUG}", "isPrivate": True, "licenses": [{"name": "other"}]}))
        print(f"dataset -> {out}")
    k = base / "kernel-e1"
    k.mkdir(parents=True, exist_ok=True)
    code = NOTEBOOK.format(steps=args.steps)
    (k / "e1.ipynb").write_text(json.dumps({
        "cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                   "source": code.splitlines(True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}))
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.owner}/{KERNEL}", "title": KERNEL, "code_file": "e1.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{args.owner}/{SLUG}"],
        "kernel_sources": [f"{args.owner}/{KERNEL}"] if args.chain else [],
        "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}, indent=1))
    print(f"notebook -> {k} (chain={args.chain})")


if __name__ == "__main__":
    main()
