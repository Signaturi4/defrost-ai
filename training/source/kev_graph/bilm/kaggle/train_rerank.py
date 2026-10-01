"""Kev-Rerank: a cross-encoder on the tech-doc backbone (Qwen2.5-0.5B, bidirectional, MNTP merged) + LoRA r16 + a
linear score head on the mean-pooled pair representation.

  input   "<instruction>: <query>\\n\\n<passage>"  (query <= 64 tokens, passage <= 384), full bidirectional attention
          across query and passage: every passage token sees the question, which is what a bi-encoder cannot do
  loss    listwise softmax cross-entropy over each group [positive, 7 hard negatives] (label = the positive)
  batch   16 groups per step (8 per GPU on 2x T4) = 128 pairs; lr 1e-4 (LoRA) / 1e-3 (head), warmup 100, linear decay

Groups: kev_graph/data/sup_v1/rerank_{train,val}.jsonl (sup_data.py rerank): the leakage-gated bi-encoder rows, each
with its own hard negative + BM25-mined negatives from the same source. Distributed like train_sup.py: independent
per-rank losses, LoRA + head gradients all-reduced and averaged (= one-GPU training on the global batch).

    torchrun --nproc_per_node 2 train_rerank.py --data DIR --mntp DIR --out DIR"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from kev_graph.bilm.kaggle.train_sup import all_reduce_grads, fingerprint, log, rank, world  # noqa: E402

INSTRUCTION = "Given a developer question about a software project, retrieve the documentation passage that answers it"
INSTRUCTIONS = {"msmarco": "Given a web search query, retrieve relevant passages that answer the query",
                "nq": "Given a question, retrieve Wikipedia passages that answer the question",
                "hotpotqa": "Given a multi-hop question, retrieve documents that can help answer the question",
                "stackexchange": "Given a question title from a technical Q&A site, retrieve the body of that question",
                "allnli": "Given a premise, retrieve a hypothesis that is entailed by the premise",
                "quora": "Given a question, retrieve questions that are semantically equivalent to the given question",
                "techdoc": INSTRUCTION}


class PairEncoder:
    def __init__(self, tok, max_query=64, max_doc=384):
        self.tok, self.mq, self.md = tok, max_query, max_doc

    def __call__(self, pairs, instructions):
        seqs = []
        for (q, d), ins in zip(pairs, instructions):
            a = self.tok(f"{ins}: ", add_special_tokens=False)["input_ids"]
            b = self.tok(q, add_special_tokens=False)["input_ids"][:self.mq]
            c = self.tok("\n\n" + d, add_special_tokens=False)["input_ids"][:self.md]
            seqs.append(a + b + c)
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs])
        att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs])
        return ids, att


class Scorer(torch.nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, 1)

    def forward(self, ids, att, dtype):
        B, L = att.shape
        mask = torch.zeros(B, 1, L, L, device=ids.device, dtype=dtype)
        mask = mask.masked_fill(~att.bool()[:, None, None, :], torch.finfo(dtype).min)
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        m = att.to(h.dtype)[..., None]
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.norm(v.float())).squeeze(-1)


def build(base, mntp, grad_ckpt=True, cgsa=None, init_from=None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModel
    from kev_graph.bilm.bridge import _merge
    m = _merge(AutoModel.from_pretrained(base, dtype=torch.float32, attn_implementation="sdpa"), mntp)
    if cgsa:
        m = _merge(m, cgsa)
    if init_from:                          # warm start: continue the LoRA of a trained reranker (v2 from v1)
        m = PeftModel.from_pretrained(m, init_from, is_trainable=True)
    else:
        m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                         "down_proj"]))
    if grad_ckpt:
        m.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.base_model.model.enable_input_require_grads()
    sc = Scorer(m, m.config.hidden_size)
    if init_from:
        h = torch.load(Path(init_from) / "head.pt", map_location="cpu")
        sc.norm.load_state_dict(h["norm"]); sc.head.load_state_dict(h["head"])
    return sc


def group_batch(groups, enc, device):
    pairs, ins = [], []
    for g in groups:
        i = INSTRUCTIONS.get(g["source"], INSTRUCTION)
        for d in [g["positive"]] + g["negatives"]:
            pairs.append((g["query"], d)); ins.append(i)
    ids, att = enc(pairs, ins)
    return ids.to(device), att.to(device), len(groups), 1 + len(groups[0]["negatives"])


@torch.no_grad()
def validate(model, enc, val, device, dtype, bs=8):
    model.eval()
    top1, mrr, n = 0.0, 0.0, 0
    for b in range(0, len(val) - bs + 1, bs):
        ids, att, G, K = group_batch(val[b:b + bs], enc, device)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            s = model(ids, att, dtype).view(G, K)
        rank_of_pos = (s > s[:, :1]).sum(1).float()
        top1 += float((rank_of_pos == 0).float().sum()); mrr += float((1 / (rank_of_pos + 1)).sum()); n += G
    model.train()
    return {"top1": round(top1 / max(1, n), 4), "mrr": round(mrr / max(1, n), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mntp", required=True)
    ap.add_argument("--cgsa", default=None)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--groups", type=int, default=16, help="GLOBAL groups per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max_doc", type=int, default=384)
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init_from", default=None, help="adapter dir with head.pt of a trained reranker (warm start)")
    args = ap.parse_args(argv)
    if "RANK" in os.environ:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        device, dtype = torch.device("mps" if torch.backends.mps.is_available() else "cpu"), torch.float32
    torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train = [json.loads(l) for l in open(Path(args.data) / "rerank_train.jsonl")]
    val = [json.loads(l) for l in open(Path(args.data) / "rerank_val.jsonl")][:256]
    f_v1 = Path(args.data) / "rerank_val_v1.jsonl"          # v2: also track the v1 val set (no forgetting)
    val_v1 = [json.loads(l) for l in open(f_v1)][:256] if f_v1.exists() else None
    from transformers import AutoTokenizer
    enc = PairEncoder(AutoTokenizer.from_pretrained(args.base), max_doc=args.max_doc)
    model = build(args.base, args.mntp, cgsa=args.cgsa, init_from=args.init_from).to(device)
    model.train()
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n]
    head = list(model.norm.parameters()) + list(model.head.parameters())
    params = lora + head
    log(f"world {world()} | {len(train)} groups | {args.groups} groups/step | trainable {sum(p.numel() for p in params) / 1e6:.2f}M")
    opt = torch.optim.AdamW([{"params": lora, "lr": args.lr}, {"params": head, "lr": args.head_lr}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / args.warmup if s < args.warmup else
                                              max(0.0, (args.steps - s) / max(1, args.steps - args.warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    rng = random.Random(args.seed)
    order = list(range(len(train)))
    rng.shuffle(order)
    local = args.groups // world()
    t0, hist = time.time(), []
    if args.init_from:
        log(f"  val@0 (warm start): {validate(model, enc, val, device, dtype)} v1 "
            f"{validate(model, enc, val_v1, device, dtype) if val_v1 else None}")
    for step in range(args.steps):
        s0 = (step * args.groups) % (len(order) - args.groups)
        idx = order[s0 + rank() * local: s0 + (rank() + 1) * local]
        ids, att, G, K = group_batch([train[i] for i in idx], enc, device)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            s = model(ids, att, dtype).view(G, K)
        loss = F.cross_entropy(s.float(), torch.zeros(G, dtype=torch.long, device=device))
        scaler.scale(loss).backward()
        all_reduce_grads(params)
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        rec = {"step": step + 1, "loss": round(float(loss.detach()), 4),
               "top1": round(float((s.argmax(1) == 0).float().mean()), 3), "gnorm": round(float(gn), 3)}
        hist.append(rec)
        if (step + 1) % 10 == 0:
            log(f"step {step + 1}/{args.steps} loss {rec['loss']:.3f} top1 {rec['top1']:.2f} gnorm {rec['gnorm']:.2f} | "
                f"{(time.time() - t0) / (step + 1):.2f} s/step")
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            fp = fingerprint(params)
            rec["val"] = validate(model, enc, val, device, dtype)
            if val_v1:
                rec["val_v1"] = validate(model, enc, val_v1, device, dtype)
            log(f"  val@{step + 1}: {rec['val']} v1 {rec.get('val_v1')} | replicas {'in sync' if max(fp) - min(fp) < 1e-3 * (abs(fp[0]) + 1) else 'DIVERGED'}")
        if rank() == 0 and ((step + 1) % args.save_every == 0 or step + 1 == args.steps):
            d = out / ("final" if step + 1 == args.steps else f"step-{step + 1}")
            model.backbone.save_pretrained(d)
            torch.save({"norm": model.norm.state_dict(), "head": model.head.state_dict()}, d / "head.pt")
            (out / "train_log.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hist))
    if rank() == 0:
        (out / "config.json").write_text(json.dumps({**vars(args), "world": world(), "dtype": str(dtype)}, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
