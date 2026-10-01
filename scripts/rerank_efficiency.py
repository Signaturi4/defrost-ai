"""Kev-Rerank efficiency study: quantized / reduced-precision variants of Kev-Rerank against small open rerankers,
on the same candidate pools (BM25 top-20 + dense top-20) the evaluated system reranks.

Pools come from the research feature caches (kev repo: runs/kev_graph/router/<suite>__*.json), so every scorer sees
exactly the same 40 candidates per question. Reported per scorer: nDCG@10 of `rerank` and `fast`, top-1 agreement
with fp32 Kev-Rerank, and latency per query (40 candidates).

    .venv/bin/python scripts/rerank_efficiency.py [--only name,name] [--suites heldout,books,client]
-> results/private/rerank_efficiency/{scores__<name>.json, report.json}"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
KEV = ROOT.parent / "kev"
sys.path.insert(0, str(ROOT))
CACHE = KEV / "runs/kev_graph/router"
TAG = "kevret_runs_kev_graph_sup-B_final+cgsa"
KB = {"heldout": "~/heldout-memory", "client": "~/client-memory", "books": "~/books-memory"}
OUT = ROOT / "results/private/rerank_efficiency"
RRF_K = 60


# ---- data + metric -----------------------------------------------------------------------------------------------
def load(suites):
    items = []
    for s in suites:
        rows = json.load(open(CACHE / f"{s}__{TAG}.json"))["rows"]
        db = sqlite3.connect(Path(KB[s]).expanduser() / "text_kb.sqlite")
        meta = {i: (h, t) for i, h, t in db.execute("SELECT id, heading_path, text FROM sections")}
        for r in rows:
            pool = [x for x, _ in r["rerank"]]
            items.append({"suite": s, "id": r["id"], "q": r["question"], "pool": pool,
                          "heads": [meta[x][0] for x in pool], "texts": [meta[x][1] for x in pool],
                          "bm25": [x for x, _ in r["bm25"]], "dense": [x for x, _ in r["dense"]],
                          "gold": set(r["gold"]), "ref": [v for _, v in r["rerank"]]})
    return items


def rrf(lists):
    sc = {}
    for lst in lists:
        for i, x in enumerate(lst):
            sc[x] = sc.get(x, 0) + 1 / (RRF_K + i + 1)
    return sorted(sc, key=sc.get, reverse=True)


def ndcg(ranked, gold):
    for i, x in enumerate(ranked[:10]):
        if x in gold:
            return 1 / np.log2(i + 2)
    return 0.0


def metrics(it, scores):
    rr = [it["pool"][i] for i in np.argsort(-np.asarray(scores), kind="stable")]
    hy = rrf([it["bm25"], it["dense"]])
    rerank = rr + [x for x in hy if x not in set(rr)]
    agree = bool(it["bm25"]) and it["bm25"][0] == it["dense"][0]
    fast = hy if agree else rerank
    return ndcg(rerank, it["gold"]), ndcg(fast, it["gold"]), rr[0]


# ---- scorers -----------------------------------------------------------------------------------------------------
def mps():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Kev:
    """Kev-Rerank with a chosen dtype / device / doc length / batch size."""

    def __init__(self, dtype=torch.float32, device=None, max_doc=384, batch=8, int8=False):
        from kev_memory.models.reranker import KevReranker
        device = "cpu" if int8 else (device or mps())
        self.r = KevReranker(device="cpu", max_doc=max_doc)
        m = self.r.model
        if int8:        # dynamic int8 on every nn.Linear of the backbone (weights int8, activations quantized per batch)
            m.backbone = torch.ao.quantization.quantize_dynamic(m.backbone, {torch.nn.Linear}, dtype=torch.qint8)
        else:
            m.backbone.to(dtype)
        m.to(device)
        self.dtype, self.device, self.batch = (torch.float32 if int8 else dtype), device, batch

    @torch.no_grad()
    def __call__(self, q, heads, texts):
        docs = [f"{h}\n{t}" for h, t in zip(heads, texts)]
        order = sorted(range(len(docs)), key=lambda i: len(docs[i]))          # length-sorted: little padding
        out = np.zeros(len(docs), np.float32)
        m = self.r.model
        for b in range(0, len(docs), self.batch):
            idx = order[b:b + self.batch]
            ids, att = self.r._pairs(q, [docs[i] for i in idx])
            ids, att = ids.to(self.device), att.to(self.device)
            keep = att.bool()
            L = keep.shape[1]
            mask = torch.zeros(len(idx), 1, L, L, device=self.device, dtype=self.dtype)
            mask = mask.masked_fill(~keep[:, None, None, :], torch.finfo(self.dtype).min)
            h = m.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            w = att.to(h.dtype)[..., None]
            v = (h * w).sum(1) / w.sum(1).clamp(min=1)
            out[idx] = m.head(m.norm(v.float())).squeeze(-1).float().cpu().numpy()
        return out


class CrossEncoder:
    """Any HF sequence-classification reranker: score = logit (1 label) or logit[1] (2 labels)."""

    def __init__(self, name, dtype=torch.float32, max_len=512, batch=16, remote=False):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=remote)
        self.m = AutoModelForSequenceClassification.from_pretrained(name, dtype=dtype, trust_remote_code=remote)
        self.m.to(mps()).eval()
        self.max_len, self.batch = max_len, batch

    @torch.no_grad()
    def __call__(self, q, heads, texts):
        docs = [f"{h}\n{t}" for h, t in zip(heads, texts)]
        out = []
        for b in range(0, len(docs), self.batch):
            enc = self.tok([q] * len(docs[b:b + self.batch]), docs[b:b + self.batch], padding=True, truncation="only_second",
                           max_length=self.max_len, return_tensors="pt").to(self.m.device)
            lg = self.m(**enc).logits.float()
            out.append((lg[:, 0] if lg.shape[1] == 1 else lg[:, 1] - lg[:, 0]).cpu().numpy())
        return np.concatenate(out)


class Qwen3Reranker:
    """Qwen3-Reranker (causal LM): score = log p(yes) - log p(no) after its documented prompt."""

    def __init__(self, name="Qwen/Qwen3-Reranker-0.6B", dtype=torch.float16, max_len=512, batch=8):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(mps()).eval()
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        self.pre = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
                    "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
        self.suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.ins = "Given a developer question about a software project, retrieve the documentation passage that answers it"
        self.max_len, self.batch = max_len, batch

    @torch.no_grad()
    def __call__(self, q, heads, texts):
        docs = [f"{h}\n{t}" for h, t in zip(heads, texts)]
        pre, suf = self.tok(self.pre, add_special_tokens=False)["input_ids"], self.tok(self.suf, add_special_tokens=False)["input_ids"]
        out = []
        for b in range(0, len(docs), self.batch):
            body = [f"<Instruct>: {self.ins}\n<Query>: {q}\n<Document>: {d}" for d in docs[b:b + self.batch]]
            enc = self.tok(body, add_special_tokens=False, truncation=True, max_length=self.max_len - len(pre) - len(suf))
            seqs = [pre + x + suf for x in enc["input_ids"]]
            L = max(map(len, seqs))
            pad = self.tok.pad_token_id
            ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=self.m.device)
            att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=self.m.device)
            lg = self.m(input_ids=ids, attention_mask=att).logits[:, -1, :].float()
            out.append((torch.log_softmax(lg[:, [self.no, self.yes]], -1)[:, 1]).cpu().numpy())
        return np.concatenate(out)


SCORERS = {  # name: (factory, parameters (M), note)
    "kev-fp32": (lambda: Kev(torch.float32), 494, "shipped: fp32 on MPS, batch 8"),
    "kev-fp32-b40": (lambda: Kev(torch.float32, batch=40), 494, "fp32, all 40 pairs in one length-sorted batch"),
    "kev-fp16": (lambda: Kev(torch.float16, batch=40), 494, "fp16 weights + activations on MPS"),
    "kev-bf16": (lambda: Kev(torch.bfloat16, batch=40), 494, "bf16 on MPS"),
    "kev-fp16-doc256": (lambda: Kev(torch.float16, max_doc=256, batch=40), 494, "fp16, section truncated to 256 tokens"),
    "kev-int8-cpu": (lambda: Kev(int8=True, batch=40), 494, "dynamic int8 Linear (torch.ao) on CPU"),
    "kev-fp32-cpu": (lambda: Kev(torch.float32, device="cpu", batch=40), 494, "fp32 on CPU (reference for int8)"),
    "minilm-l6": (lambda: CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"), 23, "MS MARCO cross-encoder"),
    "mxbai-xsmall": (lambda: CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1"), 71, "DeBERTa-v3 xsmall"),
    "gte-modernbert": (lambda: CrossEncoder("Alibaba-NLP/gte-reranker-modernbert-base", max_len=1024), 149, "ModernBERT"),
    "granite-r2": (lambda: CrossEncoder("ibm-granite/granite-embedding-reranker-english-r2", max_len=1024), 149, "ModernBERT"),
    "bge-base": (lambda: CrossEncoder("BAAI/bge-reranker-base"), 278, "XLM-R base"),
    "bge-v2-m3-fp16": (lambda: CrossEncoder("BAAI/bge-reranker-v2-m3", dtype=torch.float16), 568, "XLM-R large, fp16"),
    "qwen3-rr-0.6b": (lambda: Qwen3Reranker(), 596, "Qwen3-Reranker, fp16"),
}


def run(name, items):
    f = OUT / f"scores__{name}.json"
    if f.exists():
        return json.load(open(f))
    scorer = SCORERS[name][0]()
    scorer(items[0]["q"], items[0]["heads"][:4], items[0]["texts"][:4])                  # warm-up
    res = []
    for n, it in enumerate(items, 1):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t0 = time.perf_counter()
        s = scorer(it["q"], it["heads"], it["texts"])
        dt = time.perf_counter() - t0
        res.append({"id": it["id"], "scores": [float(x) for x in s], "seconds": dt})
        print(f"\r{name} {n}/{len(items)} {dt:.2f}s", end="", flush=True)
    print()
    del scorer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    f.write_text(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--suites", default="heldout,books,client")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    suites = a.suites.split(",")
    items = load(suites)
    names = a.only.split(",") if a.only else list(SCORERS)
    report = {}
    ref_top = {it["id"]: it["pool"][int(np.argmax(it["ref"]))] for it in items}
    for name in names:
        try:
            res = {r["id"]: r for r in run(name, items)}
        except Exception as e:                                       # a model that fails to load is reported, not fatal
            print(f"{name}: FAILED {type(e).__name__}: {e}")
            report[name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        row = {"params_M": SCORERS[name][1], "note": SCORERS[name][2]}
        for s in suites + ["all"]:
            sub = [it for it in items if s == "all" or it["suite"] == s]
            m = [metrics(it, res[it["id"]]["scores"]) for it in sub]
            row[s] = {"n": len(sub), "rerank": float(np.mean([x[0] for x in m])), "fast": float(np.mean([x[1] for x in m])),
                      "top1_agree_kev": float(np.mean([x[2] == ref_top[it["id"]] for x, it in zip(m, sub)])),
                      "sec_per_query": float(np.median([res[it["id"]]["seconds"] for it in sub]))}
        report[name] = row
        a_ = row["all"]
        print(f"{name:16s} rerank {a_['rerank']:.3f} fast {a_['fast']:.3f} | agree {a_['top1_agree_kev']:.2f} | "
              f"{a_['sec_per_query']:.2f} s/q | " + " ".join(f"{s} {row[s]['rerank']:.3f}" for s in suites))
    prev = json.load(open(OUT / "report.json")) if (OUT / "report.json").exists() else {}
    (OUT / "report.json").write_text(json.dumps(prev | report, indent=1))


if __name__ == "__main__":
    main()
