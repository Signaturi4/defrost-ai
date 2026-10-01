# Can defrost-ai serve an FAQ chatbot?

Short answer: yes, for the retrieval step, using `hybrid` mode and a small FAQ eval of your own before launch. It
does not generate answers. It finds the FAQ entry (or doc section) that answers the user's message and returns it,
with a confidence you can use to hand off to a human. Your chatbot's LLM writes the reply from that entry.

## Why it fits

| FAQ chatbot need | what defrost-ai does | evidence |
|---|---|---|
| Find the one entry that answers a paraphrased question | Kev-Ret-B embeds "heading + text", so an FAQ entry whose heading *is* the question matches the user's wording directly; BM25 catches product names, error codes and plan names | over 175 test questions, nDCG@10 is 0.696 for BM25, 0.776 for hybrid and 0.838 for `fast` ([RESULTS.md](RESULTS.md) §2) |
| Answer only from approved text | Returns the section verbatim with `path:Lstart-end`; the reply can quote or cite it | RAGAS groundedness 0.97 when the answer model sees only kev context ([E2E_GRAPHIFY.md](E2E_GRAPHIFY.md)) |
| Know when it does not know | Adaptive k uses a calibrated confidence (softmax over the dense scores); one confident entry gives k = 1, a spread gives up to 5 | calibration measured on held-out repos, not yet on FAQ traffic |
| Stay current when the FAQ changes | Incremental rebuilds (only edited entries are re-embedded, seconds); triggers on merge to main or every N hours | 4–17 s per refresh on a 560-section repo with a 6.8k-node code graph |
| Run privately | Everything is local: SQLite FTS5, a 0.5B model and 141 MB of adapters; no API calls | — |

## What to watch

- **Latency decides the mode.** Measured through the resident service on a 564-section project (Apple M-series, 10
  queries): `bm25`, `dense` and `hybrid` took a median of 44 ms (max 83 ms). `rerank` took 6.5 s. `fast`
  sent 9 of 10 of these queries to the reranker, so it took 6.8 s. For a chat UI use `hybrid`. Use `fast` or
  `rerank` only for offline jobs, for a CUDA server, or as a second pass when hybrid's confidence is low and a
  slower answer is acceptable.
- **Quality cost of skipping the reranker.** `hybrid` is about 0.06 nDCG below `fast` on the locked test (0.776 vs
  0.838 with Kev-Rerank v2).
  FAQ entries are short and distinct, which tends to narrow that gap, but measure it on your own data.
- **Abstention is not validated for FAQs.** The confidence is calibrated on technical docs. Before you use it for
  "hand off to a human", pick the threshold on 50–100 real tickets that include questions the FAQ does not answer.
- **Language.** The adapters were trained on English. The Qwen2.5 base is multilingual, but other languages are
  untested; check BM25 vs hybrid on your own non-English questions first.
- **Narrative or very long answers.** Long prose is the weakest domain. A larger public reranker scores 0.949 vs
  our 0.902 on books. Short FAQ entries are the opposite case and suit the model.
- **The service is local and has no auth.** `kev-memory serve` binds to 127.0.0.1. In production, call it from your
  chatbot backend on the same host, or put it behind your own API; never expose it directly.

## How to set it up

1. **One entry per section.** Export the FAQ to Markdown, one `##` heading per question:

   ```markdown
   ## How do I cancel my subscription?
   Also asked as: stop billing, end plan, unsubscribe.
   Open **Settings → Billing** and choose **Cancel plan**. Access continues until the end of the paid period.
   ```

   The "Also asked as" line gives BM25 and the embedder the words customers use. The doc rules in
   [WRITING_FOR_EXTRACTION.md](WRITING_FOR_EXTRACTION.md) apply as they are: self-contained sections, exact names,
   one topic per heading.
2. **Build a docs-only memory.** No code graph is needed:

   ```sh
   kev-memory setup ./faq --domain faq --build now --every-hours 6
   ```

3. **Query from the bot.**

   ```python
   from kev_memory.service import client
   res = client.search(user_message, ["faq"], "hybrid", "auto")
   top = res["hits"][0]          # section text, path:lines, heading; len(res["hits"]) == 1 means a confident match
   ```

   Send the returned entries to your answer model with the instruction "answer only from these entries; if they do
   not answer the question, say so and offer a human". With `k="auto"`, a single hit means high confidence. Five
   hits mean the retriever is unsure: prefer a clarifying question or a hand-off.
4. **Measure before launch.** Collect 50–100 real customer messages and label the FAQ entry that answers each one.
   Write one JSON line per message, with the path and lines as `memory_search` reports them, and compare recall@1
   for `bm25`, `hybrid` and `fast`:

   ```sh
   # faq_eval.jsonl: {"id": "t1", "question": "how do i stop being charged", "gold": [{"path": "faq/faq/billing.md", "lines": [12, 15]}]}
   kev-memory benchmark --suite faq_eval.jsonl --memory ~/.kev-memory/faq --split all
   ```

   Keep the messages the FAQ does not answer in a separate list. Run them through `client.search` and check how often
   `k="auto"` returns five hits: that is your hand-off signal.

   Ship `hybrid` if it is within a few points of `fast`. Otherwise run the reranker on CUDA, where it is much faster
   than on Apple MPS, or rerank only low-confidence queries.

## Verdict

| | |
|---|---|
| Retrieval for an FAQ bot | **Good fit**: short, distinct, question-shaped entries are the easy case for hybrid search |
| Real-time default | `hybrid` (~45 ms); not `fast`/`rerank` on Apple hardware (6–7 s) |
| Needs before production | an FAQ eval with "no answer" cases, an abstention threshold, a check in non-English languages |
| Not covered | answer generation, conversation state, multi-turn rewriting ("and for the yearly plan?" needs the previous turn folded into the query by your bot) |
