"""P2b supervised retrieval stage (the KG-BiLM / llm2vec `lp` stage) for the tech-doc encoder.

Recipe (research/kg-bilm-main/train_configs/lp/Qwen2.5.json, adapted to 2x T4):
  init       Qwen2.5-0.5B, bidirectional attention (4D mask), MNTP adapter merged (arm B: + CGSA merged)
  adapter    new LoRA r16 / alpha 32 / dropout 0.05 on q,k,v,o,gate,up,down
  input      query = "<instruction>: <query>" with the instruction tokens excluded from mean pooling (llm2vec
             embed_mask); documents plain; query <= 64 tokens, document <= 384
  loss       InfoNCE, scale 20 (tau 0.05): each query vs every positive and hard negative in the GLOBAL batch
             (gathered across GPUs with gradient)
  batch      global 64 queries from ONE source per step (llm2vec E5 batching), split over the GPUs
  schedule   lr 2e-4, AdamW, warmup 100, linear decay, 1000 steps, fp16 autocast over fp32 weights (T4: no bf16)

Distributed: plain torch.distributed, no DDP wrapper. Every rank runs its own forwards, embeddings are all-gathered
with autograd (backward sums the remote gradients into their owners), then the LoRA gradients are all-reduced and
divided by the world size. That equals one-GPU training on the global batch with the mean loss, and it allows the
separate query / document forwards that DDP's once-per-backward hooks do not.

    torchrun --nproc_per_node 2 train_sup.py --data DIR --mntp DIR [--cgsa DIR] --out DIR
    python train_sup.py ... --steps 3 --batch 4 --max_doc 64        # local smoke (1 process, CPU/MPS)"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))        # repo root, or the bundle's code/


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def log(*a):
    if rank() == 0:
        print(*a, flush=True)


def gather(x):
    """all-gather rows with autograd; every rank must pass the same shape"""
    if world() == 1:
        return x
    from torch.distributed.nn.functional import all_gather
    return torch.cat(all_gather(x), 0)


class Batches:
    """Deterministic schedule shared by all ranks: each step picks one source (proportional to its size) and the
    next `global_batch` rows of it; rank r takes its slice."""
    def __init__(self, rows, global_batch, seed):
        self.by = defaultdict(list)
        for r in rows:
            self.by[r["source"]].append(r)
        self.rng = random.Random(seed)
        for v in self.by.values():
            self.rng.shuffle(v)
        self.pos = {k: 0 for k in self.by}
        self.gb = global_batch
        self.names = sorted(self.by)
        self.weights = [len(self.by[k]) for k in self.names]

    def next(self):
        src = self.rng.choices(self.names, self.weights)[0]
        v, p = self.by[src], self.pos[src]
        if p + self.gb > len(v):
            self.rng.shuffle(v)
            p = 0
        self.pos[src] = p + self.gb
        return src, v[p:p + self.gb]


def build_model(base, mntp, cgsa, attn, grad_ckpt=True, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel
    from kev_graph.bilm.bridge import _merge
    model = AutoModel.from_pretrained(base, dtype=torch.float32, attn_implementation=attn)
    model = _merge(model, mntp)
    if cgsa:
        model = _merge(model, cgsa)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=dropout, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    if grad_ckpt:
        model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.base_model.model.enable_input_require_grads()
    return model


class Tok:
    def __init__(self, base, max_query, max_doc):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(base)
        self.mq, self.md = max_query, max_doc

    def queries(self, pairs):
        """[(instruction, query)] -> ids, attention, pool mask (query tokens only)"""
        seqs, pool = [], []
        for ins, q in pairs:
            a = self.tok(f"{ins}: ", add_special_tokens=False)["input_ids"]
            b = self.tok(q, add_special_tokens=False)["input_ids"][:self.mq]
            seqs.append(a + b)
            pool.append([0] * len(a) + [1] * len(b))
        return self._pad(seqs, pool)

    def docs(self, texts):
        seqs = [self.tok(t, add_special_tokens=False)["input_ids"][:self.md] for t in texts]
        return self._pad(seqs, [[1] * len(s) for s in seqs])

    def _pad(self, seqs, pool):
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs])
        att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs])
        pm = torch.tensor([p + [0] * (L - len(p)) for p in pool])
        return ids, att, pm


def embed(model, ids, att, pm, device, dtype):
    ids, att, pm = ids.to(device), att.to(device), pm.to(device)
    B, L = att.shape
    # bidirectional: every token sees every real token; padded keys blocked (same mask as P2 / the memory encoder)
    mask = torch.zeros(B, 1, L, L, device=device, dtype=dtype)
    mask = mask.masked_fill(~att.bool()[:, None, None, :], torch.finfo(dtype).min)
    h = model(input_ids=ids, attention_mask=mask).last_hidden_state
    pm = pm.to(h.dtype)[..., None]
    v = (h * pm).sum(1) / pm.sum(1).clamp(min=1)
    return F.normalize(v.float(), dim=-1)


def step_loss(model, tok, rows, device, dtype, scale):
    q = embed(model, *tok.queries([(r["instruction"], r["query"]) for r in rows]), device, dtype)
    d = embed(model, *tok.docs([r["positive"] for r in rows] + [r["negative"] for r in rows]), device, dtype)
    n = len(rows)
    pos, neg = d[:n], d[n:]
    P, N = gather(pos), gather(neg)                      # [world*n, dim] each
    scores = q @ torch.cat([P, N], 0).T * scale          # local queries vs all global docs
    labels = torch.arange(n, device=device) + rank() * n
    loss = F.cross_entropy(scores, labels)
    with torch.no_grad():
        acc = (scores.argmax(1) == labels).float().mean()
        gap = ((q * pos).sum(1) - (q * neg).sum(1)).mean()
    return loss, acc, gap


def all_reduce_grads(params):
    if world() == 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad)
            p.grad /= world()


def fingerprint(params):
    s = torch.stack([p.detach().float().sum() for p in params]).sum()
    if world() > 1:
        t = [torch.zeros_like(s) for _ in range(world())]
        dist.all_gather(t, s)
        return [float(x) for x in t]
    return [float(s)]


@torch.no_grad()
def validate(model, tok, rows, device, dtype, scale, bs):
    """in-batch retrieval on the fixed val set (same on every rank, no gather): loss, top-1 acc, pos-neg gap"""
    model.eval()
    tot = defaultdict(float)
    n = 0
    for b in range(0, len(rows) - bs + 1, bs):
        chunk = rows[b:b + bs]
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            q = embed(model, *tok.queries([(r["instruction"], r["query"]) for r in chunk]), device, dtype)
            d = embed(model, *tok.docs([r["positive"] for r in chunk] + [r["negative"] for r in chunk]), device, dtype)
        s = q @ d.T * scale
        lab = torch.arange(len(chunk), device=device)
        tot["loss"] += float(F.cross_entropy(s, lab)); tot["acc"] += float((s.argmax(1) == lab).float().mean())
        tot["gap"] += float(((q * d[:len(chunk)]).sum(1) - (q * d[len(chunk):]).sum(1)).mean()); n += 1
    model.train()
    return {k: round(v / max(1, n), 4) for k, v in tot.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mntp", required=True)
    ap.add_argument("--cgsa", default=None)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="", help="comma list to keep (arm C: public only)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64, help="GLOBAL queries per step")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--max_query", type=int, default=64)
    ap.add_argument("--max_doc", type=int, default=384)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--init_adapter", default=None, help="warm start: load this supervised LoRA before training")
    ap.add_argument("--time_budget", type=float, default=0, help="hours; save last/ and stop before this (Kaggle 12 h)")
    args = ap.parse_args(argv)

    if "RANK" in os.environ:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float32
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(Path(args.data) / "train.jsonl")]
    val = [json.loads(l) for l in open(Path(args.data) / "val.jsonl")]
    if args.sources:
        keep = set(args.sources.split(","))
        rows = [r for r in rows if r["source"] in keep]
        val = [r for r in val if r["source"] in keep]
    assert args.batch % world() == 0
    local = args.batch // world()
    log(f"world {world()} | global batch {args.batch} ({local}/rank) | {len(rows)} rows | dtype {dtype} | "
        f"sources {dict((k, sum(r['source'] == k for r in rows)) for k in sorted({r['source'] for r in rows}))}")

    model = build_model(args.base, args.mntp, args.cgsa, args.attn, not args.no_grad_ckpt, args.dropout).to(device)
    model.train()
    if args.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(Path(args.init_adapter) / "adapter_model.safetensors")))
        log(f"warm start from {args.init_adapter}")
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable {sum(p.numel() for p in params) / 1e6:.2f}M params")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / args.warmup if s < args.warmup else
                                              max(0.0, (args.steps - s) / max(1, args.steps - args.warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    tok = Tok(args.base, args.max_query, args.max_doc)
    batches = Batches(rows, args.batch, args.seed)
    start = 0
    ck = out / "last"
    if (ck / "state.pt").exists():                      # resume
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(ck / "adapter_model.safetensors")))
        st = torch.load(ck / "state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start = st["step"]
        for _ in range(start):                          # replay the schedule so the data order is unchanged
            batches.next()
        log(f"resumed from step {start}")

    hist = []
    t0 = time.time()
    for step in range(start, args.steps):
        src, grows = batches.next()
        mine = grows[rank() * local:(rank() + 1) * local]
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            loss, acc, gap = step_loss(model, tok, mine, device, dtype, args.scale)
        scaler.scale(loss).backward()
        all_reduce_grads(params)
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        rec = {"step": step + 1, "source": src, "loss": round(float(loss.detach()), 4), "acc": round(float(acc), 3),
               "gap": round(float(gap), 4), "gnorm": round(float(gn), 3), "lr": sched.get_last_lr()[0]}
        hist.append(rec)
        if (step + 1) % 10 == 0 or step == start:
            el = time.time() - t0
            log(f"step {step + 1}/{args.steps} {src:13s} loss {rec['loss']:.3f} acc {rec['acc']:.2f} gap "
                f"{rec['gap']:+.3f} gnorm {rec['gnorm']:.2f} lr {rec['lr']:.2e} | {el / (step + 1 - start):.2f} s/step")
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            fp = fingerprint(params)
            v = validate(model, tok, val, device, dtype, args.scale, 32)
            rec["val"] = v
            log(f"  val@{step + 1}: {v} | replicas {'in sync' if max(fp) - min(fp) < 1e-3 * (abs(fp[0]) + 1) else 'DIVERGED ' + str(fp)}")
        if rank() == 0 and ((step + 1) % args.save_every == 0 or step + 1 == args.steps):
            for d in (out / f"step-{step + 1}", ck):
                model.save_pretrained(d)
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "step": step + 1}, ck / "state.pt")
            (out / "train_log.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hist))
        if world() > 1 and (step + 1) % args.save_every == 0:
            dist.barrier()
        if args.time_budget and (step + 1) % args.save_every == 0 and time.time() - t0 > args.time_budget * 3600:
            log(f"time budget reached at step {step + 1}: last/ saved, resume in the next session")
            if dist.is_initialized():
                dist.destroy_process_group()
            return
    if rank() == 0:
        model.save_pretrained(out / "final")
        (out / "train_log.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hist))
        (out / "config.json").write_text(json.dumps({**vars(args), "world": world(), "dtype": str(dtype),
                                                     "instruction_pooling": "excluded", "pooling": "mean"}, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
