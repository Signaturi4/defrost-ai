"""P3: train + evaluate the label-conditioned Stage 1 extractor on the P1 tech-doc data, one backbone arm per run.

    uv run python -m defrost_graph.techdoc.train_lc --arm bka --steps 40 --batch 2 --eval_limit 20 --out runs/defrost_graph/lc_smoke
    uv run python -m defrost_graph.techdoc.train_lc --arm mntp --mntp_adapter runs/defrost_graph/techdoc-mntp --out runs/defrost_graph/lc_mntp

Arms (tech_doc_graph_training_plan.md P3: "compare causal / BKA / domain-MNTP arms with paired bootstrap"):
  causal  stock Qwen2.5-0.5B, causal attention
  bka     stock weights, bidirectional 4D mask (defrost_graph/bilm/bka.py)
  mntp    base + tech-doc MNTP adapter merged, bidirectional
  cgsa    base + MNTP merged + CGSA adapter merged, bidirectional (bridge.py's recipe)
Everything else is identical across arms: data order, label sampling, LoRA, heads, steps, seed.

Data: P1 train passages (15 whole repos) + the train splits of the two public software-NER sets. Sources are mixed
by temperature (p ~ n^0.5, any_domain/sampler.py), negative labels are drawn only from the same label family (tech-doc
vs public), because the families annotate different things under the same names (Stack Overflow NER has `class`
but no `method`). Evaluation uses the full tech-doc label set on every passage of held-out repos (dev: marshmallow,
werkzeug; test: jinja, defrost) and writes per-document counts for defrost_graph/techdoc/compare_lc.py.

Metrics (exact word span + label; predictions overlapping an `ignore` region are dropped, as in training):
typed micro P/R/F1 (gate R1 >= 0.85), boundary-only F1, per-label and per-repo F1."""
import argparse
import contextlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from defrost_graph.any_domain.sampler import label_view, source_weights
from defrost_graph.techdoc.lc_model import Encoder, LabelConditionedExtractor

PUBLIC_FILES = ("public_stackoverflow_ner.jsonl", "public_ner_re_software_mentions.jsonl")


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def family(record):
    return "techdoc" if record["source"].startswith("techdoc") else "public"


def load(data_dir, public=True):
    data_dir = Path(data_dir)
    train = read_jsonl(data_dir / "train.jsonl")
    if public:
        for name in PUBLIC_FILES:
            train += [r for r in read_jsonl(data_dir / name) if r["split"] == "train"]
    splits = {s: read_jsonl(data_dir / f"{s}.jsonl") for s in ("dev", "test")}
    for recs in splits.values():                        # whole-repo split: no held-out repo may appear in train
        leaked = {r["repo"] for r in recs} & {r.get("repo") for r in train}
        if leaked:
            raise SystemExit(f"held-out repos found in train: {sorted(leaked)}")
    return train, splits


def batches(train, batch_size, n_batches, seed=0, alpha=0.5, **label_kw):
    """Temperature-sampled batches of (record, labels, entity targets). Label pools are per family within a batch;
    an example whose view has no labels (no entities, no negatives drawn) gets 1-3 labels from its family's set, so
    entity-free passages still teach "nothing here"."""
    by_source = defaultdict(list)
    for r in train:
        by_source[r["source"]].append(r)
    fam_labels = defaultdict(set)
    for r in train:
        fam_labels[family(r)].update(e[2] for e in r["entities"])
    fam_labels = {f: sorted(v) for f, v in fam_labels.items()}
    probs = source_weights({s: len(v) for s, v in by_source.items()}, alpha)
    sources, weights = zip(*sorted(probs.items()))
    rng = random.Random(seed)
    for _ in range(n_batches):
        batch = [rng.choice(by_source[rng.choices(sources, weights)[0]]) for _ in range(batch_size)]
        pools = defaultdict(set)
        for r in batch:
            pools[family(r)].update(e[2] for e in r["entities"])
        out = []
        for r in batch:
            labels, ents = label_view(r, sorted(pools[family(r)]), rng, **label_kw)
            if not labels:
                labels = rng.sample(fam_labels[family(r)], min(len(fam_labels[family(r)]), rng.randint(1, 3)))
            out.append((r, labels, ents))
        yield out


def build_backbone(arm, base, mntp_adapter, cgsa_adapter, attn):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(base, dtype=torch.float32, attn_implementation=attn)
    if arm in ("mntp", "cgsa"):
        from defrost_graph.bilm.bridge import _merge
        model = _merge(model, mntp_adapter)
        if arm == "cgsa":
            model = _merge(model, cgsa_adapter)
    return model


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def overlaps(s, e, regions):
    return any(s < b and e > a for a, b in regions)


@torch.no_grad()
def evaluate(model, encoder, records, labels, device, autocast, threshold=0.5):
    """-> (metrics, per_doc) where per_doc[doc] = [tp, fp, fn, btp, bfp, bfn] (typed, then boundary-only)."""
    model.eval()
    per_doc = defaultdict(lambda: [0] * 6)
    per_label = defaultdict(lambda: [0, 0, 0])
    per_repo = defaultdict(lambda: [0, 0, 0])
    for r in records:
        enc = encoder(r["words"], labels)
        with autocast:
            pred = model.predict(enc, labels, device, threshold)
        pred = {(s, e, lab) for s, e, lab, _ in pred if not overlaps(s, e, r["ignore"])}
        gold = {(s, e, lab) for s, e, lab, _ in r["entities"]}
        doc = f"{r.get('repo')}/{r.get('doc_path')}"
        c = per_doc[doc]
        for key, P, G in ((0, pred, gold), (3, {x[:2] for x in pred}, {x[:2] for x in gold})):
            c[key] += len(P & G); c[key + 1] += len(P - G); c[key + 2] += len(G - P)
        for x in pred | gold:
            k = per_label[x[2]]
            k[0 if x in pred and x in gold else 1 if x in pred else 2] += 1
        rp = per_repo[r.get("repo")]
        rp[0] += len(pred & gold); rp[1] += len(pred - gold); rp[2] += len(gold - pred)
    model.train()
    tot = [sum(c[i] for c in per_doc.values()) for i in range(6)]
    p, rc, f = prf(*tot[:3])
    metrics = {"typed_p": p, "typed_r": rc, "typed_f1": f, "boundary_f1": prf(*tot[3:])[2],
               "counts": dict(zip(["tp", "fp", "fn", "btp", "bfp", "bfn"], tot)), "n_records": len(records),
               "per_label_f1": {k: round(prf(*v)[2], 4) for k, v in sorted(per_label.items())},
               "per_repo_f1": {k: round(prf(*v)[2], 4) for k, v in sorted(per_repo.items())}}
    return metrics, dict(per_doc)


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--data", default="defrost_graph/data/techdoc_v1")
    a.add_argument("--arm", required=True, choices=["causal", "bka", "mntp", "cgsa"])
    a.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    a.add_argument("--mntp_adapter", default="runs/defrost_graph/techdoc-mntp")
    a.add_argument("--cgsa_adapter", default="runs/defrost_graph/techdoc-cgsa")
    a.add_argument("--no_public", action="store_true", help="train on tech-doc passages only")
    a.add_argument("--steps", type=int, default=2000)
    a.add_argument("--batch", type=int, default=8, help="examples per optimizer step (accumulated one at a time)")
    a.add_argument("--lr", type=float, default=2e-4, help="LoRA")
    a.add_argument("--head_lr", type=float, default=1e-3, help="heads start from scratch: 1e-4 needed >60 steps to fit 8 passages, 1e-3 fits them in 30")
    a.add_argument("--lora_r", type=int, default=16)
    a.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj")
    a.add_argument("--dim", type=int, default=256)
    a.add_argument("--dropout", type=float, default=0.4)
    a.add_argument("--focal_gamma", type=float, default=0.0)
    a.add_argument("--max_tokens", type=int, default=1024)
    a.add_argument("--max_types", type=int, default=25)
    a.add_argument("--threshold", type=float, default=0.5)
    a.add_argument("--eval_every", type=int, default=500)
    a.add_argument("--eval_limit", type=int, default=0, help="dev records scored during training (0 = all)")
    a.add_argument("--test", action="store_true", help="also score the locked test split at the end")
    a.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    a.add_argument("--attn", default="sdpa")
    a.add_argument("--checkpointing", action="store_true")
    a.add_argument("--device", default="")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--log_every", type=int, default=50)
    a.add_argument("--out", required=True)
    args = a.parse_args(argv)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    dtype = args.dtype
    if dtype == "auto":
        # native bf16 needs compute capability >= 8 (Ampere). is_bf16_supported() also says True on a T4 (7.5), where
        # bf16 is emulated and slow (seen on Kaggle, torch 2.10), so decide from the capability instead.
        dtype = ("bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16") if device.type == "cuda" else "fp32"
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype)
    autocast = torch.autocast(device.type, dtype=amp) if amp else contextlib.nullcontext()
    scaler = torch.amp.GradScaler("cuda") if dtype == "fp16" else None
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, splits = load(args.data, public=not args.no_public)
    eval_labels = sorted({e[2] for r in train if family(r) == "techdoc" for e in r["entities"]})
    print(f"arm {args.arm} | device {device} {dtype} | {len(train)} train records "
          f"({Counter(family(r) for r in train)}) | eval labels {eval_labels}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    encoder = Encoder(tok, args.max_tokens)
    base = build_backbone(args.arm, args.base, args.mntp_adapter, args.cgsa_adapter, args.attn)
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
                      target_modules=args.lora_targets.split(","), task_type="FEATURE_EXTRACTION")
    backbone = get_peft_model(base, lora)
    if args.checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.enable_input_require_grads()
    model = LabelConditionedExtractor(backbone, base.config.hidden_size, args.dim, dropout=args.dropout,
                                      bidirectional=args.arm != "causal").to(device)
    model.train()
    lora_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}],
                            weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * args.steps)), args.steps)

    dev = splits["dev"][:args.eval_limit] if args.eval_limit else splits["dev"]
    history, best = [], {"typed_f1": -1.0}
    t0, run_loss, run_pos = time.time(), 0.0, 0
    label_kw = {"max_types": args.max_types, "max_neg_ratio": 1, "drop_prob": 0.5, "key": "entities"}
    for step, batch in enumerate(batches(train, args.batch, args.steps, seed=args.seed, **label_kw), start=1):
        for r, labels, ents in batch:
            enc = encoder(r["words"], labels)
            if not enc["n_words"]:
                continue
            with autocast:
                loss, n_pos = model.loss(enc, labels, ents, r["ignore"], device, args.focal_gamma)
            loss = loss / args.batch
            (scaler.scale(loss) if scaler else loss).backward()
            run_loss += loss.item(); run_pos += n_pos
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
        (scaler.step(opt), scaler.update()) if scaler else opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % args.log_every == 0:
            el = time.time() - t0
            print(f"step {step}/{args.steps} loss {run_loss / args.log_every:.3f} pos/batch "
                  f"{run_pos / args.log_every:.1f} | {el / step:.2f} s/step, eta {(args.steps - step) * el / step / 60:.0f} min",
                  flush=True)
            history.append({"step": step, "loss": run_loss / args.log_every})
            run_loss, run_pos = 0.0, 0
        if step % args.eval_every == 0 or step == args.steps:
            m, _ = evaluate(model, encoder, dev, eval_labels, device, autocast, args.threshold)
            history.append({"step": step, "dev": {k: m[k] for k in ("typed_p", "typed_r", "typed_f1", "boundary_f1")}})
            print(f"  dev@{step}: typed P {m['typed_p']:.3f} R {m['typed_r']:.3f} F1 {m['typed_f1']:.3f} | "
                  f"boundary F1 {m['boundary_f1']:.3f}", flush=True)
            if m["typed_f1"] > best["typed_f1"]:
                best = {"step": step, **{k: m[k] for k in ("typed_p", "typed_r", "typed_f1", "boundary_f1")}}
                backbone.save_pretrained(out / "adapter")
                torch.save({n: p for n, p in model.state_dict().items() if not n.startswith("backbone.")},
                           out / "heads.pt")

    # final numbers come from the best dev checkpoint, on the full dev split (and test if asked)
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(backbone, load_file(out / "adapter/adapter_model.safetensors"))
    model.load_state_dict(torch.load(out / "heads.pt", map_location=device), strict=False)
    report = {"args": vars(args), "eval_labels": eval_labels, "best": best, "history": history,
              "train_minutes": round((time.time() - t0) / 60, 1)}
    for split in ["dev"] + (["test"] if args.test else []):
        m, per_doc = evaluate(model, encoder, splits[split], eval_labels, device, autocast, args.threshold)
        report[split] = m
        (out / f"per_doc_{split}.json").write_text(json.dumps(per_doc))
        print(f"{split} (best step {best['step']}): typed F1 {m['typed_f1']:.4f} (P {m['typed_p']:.3f} R "
              f"{m['typed_r']:.3f}) boundary F1 {m['boundary_f1']:.4f}\n  per label {m['per_label_f1']}\n"
              f"  per repo {m['per_repo_f1']}", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "config.json").write_text(json.dumps({"arm": args.arm, "base": args.base, "dim": args.dim,
                                                 "dropout": args.dropout, "labels": eval_labels,
                                                 "bidirectional": args.arm != "causal"}, indent=2))


if __name__ == "__main__":
    main()
