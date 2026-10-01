"""Post-training check on Kaggle: anisotropy of MNTP-only vs MNTP+CGSA embeddings on held-out docs.

    python quick_check.py --data <dataset dir> --cgsa <output dir> [--n 500]

Same definition as kev_graph/bilm/bridge.py (mean pairwise cosine between different sentences'
mean-pooled bidirectional embeddings; lower = more isotropic) and the same line sampling, so the
number is comparable. This is a smoke signal that CGSA did something; the real evaluation runs
locally with bridge.py / compare.py after the adapter is pulled back.
"""
import argparse
import json
import os
import random

import torch
from peft import PeftModel

from llm2vec import LLM2Vec


def lines(path, n, seed=0, min_chars=40):
    with open(path) as fh:
        out = [l.strip() for l in fh if len(l.strip()) >= min_chars]
    random.Random(seed).shuffle(out)
    return out[:n]


@torch.no_grad()
def anisotropy(model, sentences):
    v = model.encode(sentences, batch_size=32, show_progress_bar=False, convert_to_tensor=True).float()
    v = torch.nn.functional.normalize(v, dim=-1)
    sim = v @ v.T
    n = sim.shape[0]
    return float((sim.sum() - sim.diagonal().sum()) / (n * (n - 1)))


def load(data, cgsa=None):
    model = LLM2Vec.from_pretrained(
        "Qwen/Qwen2.5-0.5B", peft_model_name_or_path=os.path.join(data, "mntp-adapter"), merge_peft=True,
        enable_bidirectional=True, pooling_mode="mean", max_length=128, torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    if cgsa:
        model.model = PeftModel.from_pretrained(model.model, cgsa).merge_and_unload()
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cgsa", required=True)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    sentences = lines(os.path.join(args.data, "corpus", "corpus_heldout.txt"), args.n)
    report = {"n_sentences": len(sentences)}
    for name, cgsa in (("mntp_only", None), ("mntp_cgsa", args.cgsa)):
        model = load(args.data, cgsa)
        report[f"anisotropy_{name}"] = round(anisotropy(model, sentences), 4)
        del model
        torch.cuda.empty_cache()
    print(json.dumps(report, indent=2))
    with open(os.path.join(args.cgsa, "quick_check.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
