"""Turn the KG-BiLM / LLM2Vec pretraining outputs into backbones kev_graph can train on -- and prove they arrived.

The two halves of this pipeline live in different environments on purpose:

  * pretraining  -- llm2vec pins transformers<4.47, so it runs in .venv-llm2vec / modal_bilm_app.py
  * extraction   -- kev_graph runs transformers 5.17 in the project venv

The only thing that crosses is a LoRA adapter. This script is that crossing point.

What it builds, following the reference recipe exactly:

  mntp   base + MNTP adapter, merged                   (what run_kmp.py produces)
  cgsa   base + MNTP merged + CGSA adapter, merged     (run_cgsa.py loads the MNTP adapter with merge_peft=True,
                                                        experiments/run_cgsa.py:350, then trains a new LoRA on top)

Each is saved as a full HF model directory with the stock Qwen tokenizer, so `kev_graph.train_stage1_cuda
--backbone <dir> --bidirectional` loads it and puts a fresh q/v LoRA plus the GPLinker heads on top -- the same
merge-then-new-LoRA pattern LLM2Vec uses for its own downstream fine-tuning.

Why the verification is not optional. The adapters were saved by peft 0.12 against llm2vec's Qwen2BiModel on
transformers 4.40, and are loaded here by peft 0.21 onto stock Qwen2Model on 5.17. A LoRA load that silently
matches only part of the keys does not raise; it trains to a normal-looking loss and a worse number. So each
backbone is checked against the objective its own stage optimised:

  mntp  masked-next-token accuracy (predict token i from position i-1, bidirectional). run_kmp.py measured 0.613
        on its own eval split. If the adapter really arrived, the bridge reproduces roughly that; stock Qwen does not.
  cgsa  anisotropy: mean pairwise cosine between DIFFERENT sentences' mean-pooled embeddings. Contrastive training
        exists to push this down. If it did not move relative to mntp, CGSA bought nothing.

    uv run python -m kev_graph.bilm.bridge --stage mntp
    uv run python -m kev_graph.bilm.bridge --stage cgsa
"""
import argparse
import json
import os
import random

import torch

BASE = "Qwen/Qwen2.5-0.5B"
MNTP_ADAPTER = "runs/kev_graph/bilm-mntp"
CGSA_ADAPTER = "runs/kev_graph/bilm-cgsa/checkpoint-1000"
OUT = {"mntp": "runs/kev_graph/kgbilm-backbone-mntp", "cgsa": "runs/kev_graph/kgbilm-backbone-cgsa"}
CORPUS_MNTP = "kev_graph/data/bilm/corpus_mntp.txt"
CORPUS_CGSA = "kev_graph/data/bilm/corpus_cgsa.txt"


def _merge(model, adapter_dir):
    """Apply a LoRA adapter and merge it, reporting how many LoRA modules actually attached."""
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    n_lora = sum(1 for n, _ in peft_model.named_modules() if n.endswith("lora_A"))
    with open(os.path.join(adapter_dir, "adapter_config.json")) as fh:
        cfg = json.load(fh)
    n_layers = model.config.num_hidden_layers
    expected = n_layers * len(cfg["target_modules"])
    print(f"  {adapter_dir}: {n_lora} LoRA modules attached (expected {n_layers} layers x "
          f"{len(cfg['target_modules'])} targets = {expected})")
    if n_lora != expected:
        raise SystemExit(f"partial adapter load: {n_lora} != {expected} -- the bridge did not work")
    return peft_model.merge_and_unload()


def build(stage):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(BASE, dtype=torch.float32)
    model = _merge(model, MNTP_ADAPTER)
    if stage == "cgsa":
        model = _merge(model, CGSA_ADAPTER)
    return model.eval()


def _lines(path, n, seed=0, min_chars=40):
    with open(path) as fh:
        lines = [l.strip() for l in fh if len(l.strip()) >= min_chars]
    random.Random(seed).shuffle(lines)
    return lines[:n]


@torch.no_grad()
def mntp_accuracy(model, tok, lines, max_len=256, mask_prob=0.2, seed=0, bidirectional=True):
    """run_kmp.py's own metric: mask tokens with the blank token '_', predict token i from hidden state i-1
    through the tied embedding matrix (Qwen2.5-0.5B ties lm_head to embed_tokens)."""
    from kev_graph.bilm.bka import bidirectional_mask
    mask_id = tok.convert_tokens_to_ids("_")
    emb = model.get_input_embeddings().weight
    g = torch.Generator().manual_seed(seed)
    correct = total = 0
    for line in lines:
        ids = tok(line, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :max_len]
        L = ids.shape[1]
        if L < 8:
            continue
        pick = torch.rand(L, generator=g) < mask_prob
        pick[0] = False                                   # position 0 has no i-1 to predict it from
        if not pick.any():
            continue
        masked = ids.clone()
        masked[0, pick] = mask_id
        attn = bidirectional_mask(L) if bidirectional else None
        h = model(masked, attention_mask=attn).last_hidden_state if attn is not None else model(masked).last_hidden_state
        logits = h[0, :-1] @ emb.T                         # position i-1 predicts token i
        pred = logits.argmax(-1)
        target = ids[0, 1:]
        sel = pick[1:]
        correct += int((pred[sel] == target[sel]).sum())
        total += int(sel.sum())
    return correct / max(total, 1), total


@torch.no_grad()
def anisotropy(model, tok, lines, max_len=128, bidirectional=True):
    """Mean pairwise cosine between different sentences' mean-pooled embeddings (lower = more isotropic)."""
    from kev_graph.bilm.bka import bidirectional_mask
    vecs = []
    for line in lines:
        ids = tok(line, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :max_len]
        L = ids.shape[1]
        attn = bidirectional_mask(L) if bidirectional else None
        h = model(ids, attention_mask=attn).last_hidden_state if attn is not None else model(ids).last_hidden_state
        vecs.append(h[0].mean(0))
    v = torch.nn.functional.normalize(torch.stack(vecs), dim=-1)
    sim = v @ v.T
    n = sim.shape[0]
    return float((sim.sum() - sim.diagonal().sum()) / (n * (n - 1)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["mntp", "cgsa"], required=True)
    ap.add_argument("--n_check", type=int, default=200, help="corpus lines used for the verification metric")
    ap.add_argument("--skip_save", action="store_true", help="verify only, do not write the merged backbone")
    args = ap.parse_args(argv)

    from transformers import AutoModel, AutoTokenizer
    import transformers
    import peft
    print(f"transformers {transformers.__version__}, peft {peft.__version__}")
    tok = AutoTokenizer.from_pretrained(BASE)

    print(f"\n=== building {args.stage} backbone ===")
    model = build(args.stage)

    report = {"stage": args.stage, "base": BASE}
    if args.stage == "mntp":
        lines = _lines(CORPUS_MNTP, args.n_check)
        stock = AutoModel.from_pretrained(BASE, dtype=torch.float32).eval()
        rows = {
            "stock_causal": mntp_accuracy(stock, tok, lines, bidirectional=False),
            "stock_bidirectional": mntp_accuracy(stock, tok, lines, bidirectional=True),
            "mntp_bidirectional": mntp_accuracy(model, tok, lines, bidirectional=True),
        }
        print("\nmasked-next-token accuracy (run_kmp.py measured 0.613 on its eval split):")
        for k, (acc, n) in rows.items():
            print(f"  {k:22s} {acc:.4f}  (n={n})")
        report["mntp_accuracy"] = {k: v[0] for k, v in rows.items()}
        ok = rows["mntp_bidirectional"][0] > rows["stock_bidirectional"][0] + 0.10
    else:
        lines = _lines(CORPUS_CGSA, args.n_check)
        mntp_only = build("mntp")
        stock = AutoModel.from_pretrained(BASE, dtype=torch.float32).eval()
        rows = {
            "stock_bidirectional": anisotropy(stock, tok, lines),
            "mntp_bidirectional": anisotropy(mntp_only, tok, lines),
            "cgsa_bidirectional": anisotropy(model, tok, lines),
        }
        print("\nmean pairwise cosine between different sentences (lower = more isotropic):")
        for k, v in rows.items():
            print(f"  {k:22s} {v:.4f}")
        report["anisotropy"] = rows
        ok = rows["cgsa_bidirectional"] < rows["mntp_bidirectional"] - 0.05

    report["verified"] = bool(ok)
    print(f"\nBRIDGE {'VERIFIED' if ok else 'FAILED -- the adapter did not carry its objective across'}")
    if not ok:
        raise SystemExit(1)

    if not args.skip_save:
        out = OUT[args.stage]
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
        json.dump(report, open(os.path.join(out, "bridge_report.json"), "w"), indent=1)
        print(f"saved merged backbone to {out}")
    return report


if __name__ == "__main__":
    main()
