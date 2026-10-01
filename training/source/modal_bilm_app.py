"""Run the KG-BiLM / LLM2Vec pretraining stages on a Modal GPU.

    uv run modal run modal_bilm_app.py::probe          # verify the env in-container (cheap, no GPU)
    uv run modal run modal_bilm_app.py::bilm           # KMP (MNTP) then CGSA, on H100
    uv run modal run modal_bilm_app.py::run_kmp        # just the MNTP stage
    uv run modal run modal_bilm_app.py::bilm --kmp-config kmp_techdoc_modal.json --cgsa-config cgsa_techdoc_modal.json
                                                       # P2: tech-doc corpus (defrost_graph/bilm/techdoc_corpus.py), WSD,
                                                       # 30% masking, MNTP evaluated on held-out docs

Why this is a separate app from modal_graph_app.py. `llm2vec` pins transformers to 4.43.1-4.44.2; the main
project runs 5.17. The two cannot share an image, so the pretraining stages get their own, pinned to the exact
versions that were verified working locally (see PINS). The only interface back to the main project is
the LoRA adapter this writes into the kev-runs volume.

Deviations from the shipped `train_configs/*.json`, all deliberate:

  * `model_name_or_path` 7B -> Qwen2.5-0.5B. The reference targets Qwen2.5-7B-Instruct. A 7B backbone
    could not be compared against this project's 0.5B Stage 1 without the result being confounded by
    scale, which is the whole point of the comparison. Everything else about the recipe is unchanged.
  * `attn_implementation` flash_attention_2 -> sdpa. flash-attn needs a long source build in this image
    and SDPA is mathematically equivalent for these shapes. Not a modelling change.
  * data paths point at our own any-domain corpus rather than wikitext/Wiki1M (see
    defrost_graph/bilm/build_corpus.py for why).

Everything else -- the MNTP objective, mlm_probability 0.2, batch 32, LoRA r=16, lr, bf16, the CGSA
simcse_dropout 0.3 / loss_scale 20 / mean pooling -- is as shipped.

The P2 tech-doc configs (*_techdoc_modal.json) add two ModernBERT choices (Warner et al. 2024, section 2.2):
30% masking instead of 20%, and a warmup-stable-decay schedule (50 / 750 / 200 steps) instead of linear decay.
WSD is why the image moved from llm2vec 0.2.2 / transformers 4.40.2 to 0.2.3 / 4.44.2: 4.40 has no WSD scheduler.
The runs before that move (bilm-mntp, bilm-cgsa in the Re-DocRED comparison) used the old pins.
"""
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "defrost-bilm"
ROOT = Path(__file__).resolve().parent
REF = ROOT / "docs/jev_for_graph/research/kg-bilm-main"
GPU = os.environ.get("DEFROST_GPU", "H100")
RUNS_MOUNT, HF_MOUNT = "/runs", "/hf"

# Verified 2026-09-25 with a local smoke of run_kmp + run_cgsa (WSD, tech-doc corpus). llm2vec 0.2.3 requires
# transformers 4.43.1-4.44.2 (4.44 is the first with the warmup_stable_decay scheduler). peft 0.13.2 matches that
# transformers; newer peft (0.21) fails in LoRA layer init on torch.distributed.tensor.
PINS = [
    "torch",
    "transformers==4.44.2",
    "peft==0.13.2",
    "llm2vec==0.2.3",
    "accelerate==1.15.0",
    "datasets==5.0.1",
    "evaluate==0.4.6",
    "numpy==2.4.6",
    "scikit-learn==1.9.1",
    "tqdm==4.70.1",
    "tokenizers==0.19.1",
]

app = modal.App(APP_NAME)
image = (
    # Python 3.11, not 3.13: tokenizers 0.19.1 has no cp313 wheel, so uv/pip fall back to its sdist and
    # the build fails on a malformed pyproject.toml. 3.11 has wheels.
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*PINS)
    .env({"HF_HOME": HF_MOUNT, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1"})
    # experiments/ and train_configs/ only -- this deliberately excludes KG-BiLM.pdf (1.8MB of the 2.1MB).
    .add_local_dir(str(REF / "experiments"), "/kgbilm/experiments")
    .add_local_dir(str(REF / "train_configs"), "/kgbilm/train_configs")
    .add_local_dir(str(ROOT / "defrost_graph/bilm/configs"), "/configs")
    .add_local_dir(str(ROOT / "defrost_graph/data/bilm"), "/corpus")
    # corpus/ only: raw/ (unscrubbed local files) never leaves the machine
    .add_local_dir(str(ROOT / "defrost_graph/data/techdoc_mega/corpus"), "/corpus_techdoc")
)
hf_cache = modal.Volume.from_name("defrost-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("kev-runs", create_if_missing=True)


def _run(script, config, label):
    cmd = ["python", f"/kgbilm/experiments/{script}", f"/configs/{config}"]
    print(f"=== {label}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/kgbilm", text=True)
    runs_volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"{label} exited {proc.returncode}")
    print(f"=== {label} finished", flush=True)
    return {"stage": label, "returncode": proc.returncode}


@app.function(image=image, cpu=2, memory=8192, timeout=900)
def probe():
    """Verify the pinned env imports in-container. No GPU, so this is cheap and worth running first."""
    import transformers, torch, peft, llm2vec
    out = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "peft": peft.__version__,
        "cuda": torch.cuda.is_available(),
    }
    from llm2vec import LLM2Vec
    from llm2vec.models import Qwen2BiForMNTP
    out["imports"] = "LLM2Vec + Qwen2BiForMNTP OK"
    import glob
    out["corpus"] = {p: os.path.getsize(p) for p in glob.glob("/corpus*/*.txt")}
    out["configs"] = [os.path.basename(p) for p in glob.glob("/configs/*.json")]
    print(out, flush=True)
    return out


@app.function(image=image, gpu=GPU, cpu=4, memory=32768, timeout=21600,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache})
def run_kmp(config: str = "kmp_modal.json"):
    """Stage KMP: Masked Next Token Prediction -- turns the causal decoder into a bidirectional encoder."""
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    return _run("run_kmp.py", config, "kmp")


@app.function(image=image, gpu=GPU, cpu=4, memory=65536, timeout=21600,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache})
def run_cgsa(config: str = "cgsa_modal.json"):
    """Stage CGSA: InfoNCE/SimCSE contrastive, initialised from the KMP adapter."""
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    return _run("run_cgsa.py", config, "cgsa")


@app.local_entrypoint()
def bilm(kmp_config: str = "kmp_modal.json", cgsa_config: str = "cgsa_modal.json"):
    """Run the full recipe in order: KMP then CGSA. Results land in the kev-runs volume."""
    import json
    print("=== probe ===")
    print(json.dumps(probe.remote(), indent=2))
    print("\n=== KMP (MNTP) ===")
    print(json.dumps(run_kmp.remote(config=kmp_config), indent=2))
    print("\n=== CGSA (contrastive) ===")
    print(json.dumps(run_cgsa.remote(config=cgsa_config), indent=2))
    print("\nDone. Pull with:")
    for cfg in (kmp_config, cgsa_config):
        out = json.load(open(ROOT / "defrost_graph/bilm/configs" / cfg))["output_dir"].removeprefix("/runs")
        print(f"  uv run modal volume get kev-runs {out} runs{out}")
