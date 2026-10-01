"""Write the Kaggle notebook (<slug>.ipynb) from the cell sources below.

    uv run python -m kev_graph.bilm.kaggle.make_notebook [--slug kev-techdoc-cgsa]

The slug is the Kaggle dataset name; build_dataset.py passes its own --slug so the two always agree.

The notebook is generated so its code stays reviewable as plain Python in git.
"""
import argparse
import json
from pathlib import Path

DEFAULT_SLUG = "kev-techdoc-cgsa"

CELLS = [
    ("markdown", """# Kev techdoc: KG-BiLM CGSA stage on 2 GPUs

This notebook trains the CGSA (SimCSE / InfoNCE contrastive) stage of KG-BiLM on top of the MNTP adapter,
using **distributed data parallel over both GPUs** (`torchrun`, one process per GPU, NCCL).

What makes it truly parallel, not just two processes:
* each GPU runs 64 sentences, and the embeddings are **all-gathered with gradient** before the loss, so every query is
  scored against all 128 in-batch negatives (identical to the single-GPU batch-128 recipe);
* the forward goes through the DDP wrapper, so gradients are all-reduced every step. The reference `run_cgsa.py`
  bypasses DDP and each GPU would silently train its own copy;
* the log proves both: the first step prints a `128 x 128` similarity matrix, and replica weight fingerprints are
  compared across GPUs after step 1 and every 100 steps (the run aborts if they differ).

**Settings (right-hand panel):** Accelerator **GPU T4 x2**, Internet **on**, Persistence *Files only* (optional).
**Input:** add the private dataset `kev-techdoc-cgsa`.
**Run:** *Save Version → Save & Run All (Commit)* so it runs headless (~1.5–2.5 h). Results land in the Output tab.

Recipe: Qwen2.5-0.5B, bidirectional, MNTP adapter merged, new LoRA r=16, lr 3e-5, global batch 128, max length 128,
SimCSE dropout 0.3, loss scale 20, mean pooling, WSD schedule 50 warmup / 750 stable / 200 decay = 1000 steps.
T4 has no bf16, so weights stay fp32 and compute runs in fp16 autocast with a grad scaler."""),
    ("code", """# Settings
import glob, json, os, subprocess, sys, time
from pathlib import Path

SLUG = "kev-techdoc-cgsa"      # the dataset slug
GLOBAL_BATCH = 128             # the recipe's batch; split evenly over GPUs
MAX_STEPS = 1000
OUT = Path("/kaggle/working/techdoc-cgsa")
LOG = Path("/kaggle/working/train.log")

def find_data():
    hits = sorted(glob.glob(f"/kaggle/input/**/{SLUG}*/MANIFEST.json", recursive=True)) or \\
           sorted(glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True))
    hits = [h for h in hits if Path(h).with_name("code").is_dir()]
    if not hits:
        raise FileNotFoundError("dataset not attached: Add Input -> your private dataset " + SLUG)
    return Path(hits[0]).parent

DATA = find_data()
print("data:", DATA)
print("python:", sys.version.split()[0])"""),
    ("code", """# Environment. llm2vec 0.2.3 needs transformers 4.43.1-4.44.2 (4.44 is the first with the WSD scheduler);
# peft 0.13.2 matches it. The preinstalled torch is kept. tokenizers 0.19.1 has no wheel for Python >= 3.13,
# so on such an image we build a 3.12 venv with uv instead.
PINS = ["transformers==4.44.2", "tokenizers==0.19.1", "peft==0.13.2", "accelerate==1.15.0",
        "huggingface_hub>=0.23.2,<1.0", "safetensors>=0.4.3"]

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

if sys.version_info < (3, 13):
    PY = sys.executable
    sh(f"{PY} -m pip install -q " + " ".join(f"'{p}'" for p in PINS))
    sh(f"{PY} -m pip install -q --no-deps llm2vec==0.2.3")
else:
    venv = Path("/kaggle/tmp/venv312")
    sh(f"{sys.executable} -m pip install -q uv")
    sh(f"uv venv -q --python 3.12 {venv}")
    PY = str(venv / "bin/python")
    sh(f"uv pip install -q --python {PY} torch numpy tqdm packaging " + " ".join(f"'{p}'" for p in PINS))
    sh(f"uv pip install -q --python {PY} --no-deps llm2vec==0.2.3")

sh(f"{PY} -c \\"import torch, transformers, peft, llm2vec, accelerate; "
   f"print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, "
   f"'| peft', peft.__version__, '| accelerate', accelerate.__version__)\\"")"""),
    ("code", """# Hardware + data integrity
sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv")
n_gpu = int(subprocess.check_output([PY, "-c", "import torch; print(torch.cuda.device_count())"]).decode().strip())
assert n_gpu >= 1, "no GPU: set Accelerator to GPU T4 x2"
if n_gpu < 2:
    print(f"WARNING: {n_gpu} GPU visible; this will run, but not in parallel. Choose 'GPU T4 x2'.")
assert GLOBAL_BATCH % n_gpu == 0
PER_DEVICE = GLOBAL_BATCH // n_gpu
print(f"{n_gpu} GPU(s) -> per-device batch {PER_DEVICE}, global batch {GLOBAL_BATCH}")

import hashlib
manifest = json.loads((DATA / "MANIFEST.json").read_text())
for rel, meta in manifest["files"].items():
    h = hashlib.sha256()
    with open(DATA / rel, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == meta["sha256"], f"checksum mismatch: {rel}"
print(f"{len(manifest['files'])} files verified; MNTP stage:", manifest["mntp_results"])"""),
    ("code", """# Download the base model once, before launch, so the two ranks do not race on the HF cache.
sh(f"{PY} -c \\"from huggingface_hub import snapshot_download; "
   f"print(snapshot_download('Qwen/Qwen2.5-0.5B', allow_patterns=['*.json', '*.safetensors', '*.txt']))\\"")"""),
    ("code", """# Resume: a checkpoint from this session, or from a previous run's output attached as an input
# (Add Input -> Notebook Output). Trainer restores weights, optimizer, scheduler, scaler and RNG state.
OUT.mkdir(parents=True, exist_ok=True)
if not glob.glob(str(OUT / "checkpoint-*")):
    prev = sorted(glob.glob("/kaggle/input/**/techdoc-cgsa/checkpoint-*", recursive=True),
                  key=lambda p: int(p.rsplit("-", 1)[1]))
    if prev:
        sh(f"cp -r '{prev[-1]}' '{OUT}/'")
print("checkpoints:", sorted(Path(p).name for p in glob.glob(str(OUT / "checkpoint-*"))) or "none (fresh start)")

cfg = json.loads((DATA / "code/cgsa_kaggle.json").read_text().replace("${DATA}", str(DATA)).replace("${OUT}", str(OUT)))
cfg.update(per_device_train_batch_size=PER_DEVICE, max_steps=MAX_STEPS, stop_after_n_steps=MAX_STEPS)
CFG = Path("/kaggle/working/cgsa_run.json")
CFG.write_text(json.dumps(cfg, indent=2))
print(json.dumps(cfg, indent=2))"""),
    ("code", """# Launch: one process per GPU. NCCL P2P is disabled because peer-to-peer between Kaggle's T4s has been
# known to hang; the traffic here is ~35 MB of LoRA gradients + a 128 x 896 gather per step, so the cost is small.
env = dict(os.environ,
           NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1", NCCL_ASYNC_ERROR_HANDLING="1",
           TORCH_NCCL_ASYNC_ERROR_HANDLING="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2",
           PYTHONUNBUFFERED="1", HF_HUB_DISABLE_PROGRESS_BARS="1", TRANSFORMERS_VERBOSITY="warning")
cmd = [PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpu}",
       str(DATA / "code/train_cgsa_ddp.py"), str(CFG)]
print(" ".join(cmd), flush=True)
t0 = time.time()
with open(LOG, "a") as log, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, bufsize=1) as proc:
    for line in proc.stdout:
        log.write(line)
        if "Warning" not in line and "warn(" not in line:
            print(line, end="", flush=True)
rc = proc.wait()
print(f"exit {rc} after {(time.time() - t0) / 60:.1f} min")
assert rc == 0, f"training failed; see {LOG}\""""),
    ("code", """# What happened: parallelism evidence + loss / lr curves
summary = json.loads((OUT / "train_summary.json").read_text())
print({k: v for k, v in summary.items() if k != "log_history"})
steps = [h for h in summary["log_history"] if "loss" in h]
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot([h["step"] for h in steps], [h["loss"] for h in steps]); ax[0].set_yscale("log"); ax[0].set_title("CGSA loss")
ax[1].plot([h["step"] for h in steps], [h["learning_rate"] for h in steps]); ax[1].set_title("lr (WSD)")
for a in ax: a.set_xlabel("step")
plt.tight_layout(); plt.show()"""),
    ("code", """# Smoke signal: anisotropy on held-out docs, MNTP-only vs MNTP+CGSA (lower = more isotropic).
# The earlier any-domain run went 0.53 -> 0.23. Full evaluation runs locally after download.
sh(f"{PY} {DATA / 'code/quick_check.py'} --data {DATA} --cgsa {OUT}")"""),
    ("code", """# Package the final adapter (top level of OUT; checkpoints stay in the output for resuming).
import shutil, zipfile
pkg = Path("/kaggle/working/techdoc-cgsa-adapter.zip")
with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            z.write(p, f"techdoc-cgsa/{p.name}")
    z.write(LOG, "techdoc-cgsa/train.log")
    z.write(CFG, "techdoc-cgsa/cgsa_run.json")
print(pkg, f"{pkg.stat().st_size / 1e6:.1f} MB")
print("Locally: unzip into runs/kev_graph/ ->  runs/kev_graph/techdoc-cgsa/")"""),
]


def write(slug, out):
    nb = {
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": src.replace(DEFAULT_SLUG, slug).splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if kind == "code" else {})}
            for kind, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(out).write_text(json.dumps(nb, indent=1))
    return Path(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    args = ap.parse_args(argv)
    print(f"wrote {write(args.slug, Path(__file__).with_name(f'{args.slug}.ipynb'))}")


if __name__ == "__main__":
    main()
