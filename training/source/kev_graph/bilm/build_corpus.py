"""Flatten the any-domain training data into the plain-text corpus the LLM2Vec pretraining stages expect.

Why this file exists. The reference's `train_configs/kmp/Qwen2.5.json` pretrains on **wikitext** -- generic
English. That is fine for the paper, whose downstream task is link prediction over a knowledge graph. Our
downstream task is extraction from the same domains we train on, so the pretraining corpus should be our own
any-domain mixture (`kev_graph/data/any_domain/v1/train/`, built by `kev_graph.any_domain.build`). Pointing
MNTP and CGSA at that corpus is the adaptation that makes the recipe serve extraction rather than KG
embeddings.

Two outputs, because the two stages want different granularity:

  corpus_mntp.txt   one record per line  -- MNTP runs at max_seq_length 256-512, so whole documents are the
                                            right unit; the tokenizer truncates what does not fit
  corpus_cgsa.txt   one SENTENCE per line -- CGSA/SimCSE runs at max_seq_length 128 and contrasts a text
                                            against itself, so a whole document would be mostly padding

Sentence splitting reuses `kev_graph.etg.silver.sentence_ids`, which already handles both word conventions
in this repo (Re-DocRED tokenizes '.' separately, BioRED keeps it attached) and does not split on
abbreviations. Writing a second splitter here would be a second thing to keep in step.

    uv run python -m kev_graph.bilm.build_corpus --out kev_graph/data/bilm
    uv run python -m kev_graph.bilm.build_corpus --out /tmp/bilm_smoke --limit 200
"""
import argparse
import glob
import json
import os
from collections import Counter

from kev_graph.etg.silver import sentence_ids

DEFAULT_DATA = "kev_graph/data/any_domain/v1/train"
MIN_SENTENCE_WORDS = 4          # "Yes ." and similar are not useful contrastive examples
MIN_DOC_WORDS = 8


def iter_records(data_dir, limit=None):
    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"no *.jsonl under {data_dir} -- run `uv run python -m kev_graph.any_domain.build`")
    n = 0
    for path in files:
        source = os.path.basename(path)[:-6]
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield source, json.loads(line)
                n += 1
                if limit and n >= limit:
                    return


def sentences_of(words):
    """Split a record's words into sentences using the repo's existing convention."""
    ids = sentence_ids(words)
    out, cur, cur_id = [], [], ids[0] if ids else 0
    for w, sid in zip(words, ids):
        if sid != cur_id:
            out.append(cur)
            cur, cur_id = [], sid
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA, help="directory of any-domain *.jsonl training files")
    ap.add_argument("--out", default="kev_graph/data/bilm", help="output directory for the two corpora")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of records read (0 = all)")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    mntp_path = os.path.join(args.out, "corpus_mntp.txt")
    cgsa_path = os.path.join(args.out, "corpus_cgsa.txt")

    n_docs = n_sents = n_words = n_sent_words = 0
    per_source = Counter()

    with open(mntp_path, "w") as fm, open(cgsa_path, "w") as fc:
        for source, rec in iter_records(args.data, args.limit or None):
            words = rec.get("words") or []
            if len(words) < MIN_DOC_WORDS:
                continue
            fm.write(" ".join(words) + "\n")
            n_docs += 1
            n_words += len(words)
            per_source[source] += 1

            for sent in sentences_of(words):
                if len(sent) < MIN_SENTENCE_WORDS:
                    continue
                fc.write(" ".join(sent) + "\n")
                n_sents += 1
                n_sent_words += len(sent)

    print(f"records written : {n_docs:7d}  ({n_words:,} words)")
    print(f"sentences written: {n_sents:7d}  ({n_sent_words:,} words)")
    if n_docs:
        print(f"mean doc words  : {n_words / n_docs:.1f}")
    if n_sents:
        print(f"mean sent words : {n_sent_words / n_sents:.1f}")
    print("per source:")
    for src, c in per_source.most_common():
        print(f"   {src:20s} {c:7d}")
    print(f"\nwrote {mntp_path}")
    print(f"wrote {cgsa_path}")


if __name__ == "__main__":
    main()
