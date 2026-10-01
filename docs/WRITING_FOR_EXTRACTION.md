# Writing docs that people can read and machines can extract

Research summary (2026-10-01) on how to write documentation, by hand or with AI, so that:
- people still read it as normal Markdown;
- retrieval (BM25 + Defrost-Ret-B + Defrost-Rerank) finds the right section;
- the key facts convert to graph triples cheaply, without an LLM pass over every refresh.

No existing CLAUDE.md, AGENTS.md or skill in this workspace covers this. The closest local material is about
*extracting* triples (`kev/docs/jev_for_graph/new/part_optimization_of_extraction_proximity.md`), not about
*writing* for extraction. Public agent skills exist for parts of it: Diátaxis skills, an ASD-STE100 Claude Code
skill, and the developer-docs-framework skill (sources at the end).

The scores below are my judgment from the sources and from how our pipeline works. They are not measured. Section 5
proposes how to measure them on our own stack.

## 1. What the sources agree on

1. **Self-contained sections win.** Chunked text loses what "it", "this" and "the service" refer to (the
   "anaphoric reference problem"). Rewriting chunks so they name the entity improves retrieval and QA, most for
   small models and mean-pooled embedders ([arXiv 2507.07847](https://arxiv.org/pdf/2507.07847),
   [CLAP, arXiv 2508.06941](https://arxiv.org/pdf/2508.06941)). Defrost-Ret-B is a mean-pooled 0.5B embedder, so this
   applies to us directly.
2. **One meaning per term, one fact per sentence.** Simplified Technical English (ASD-STE100) uses one word per
   meaning, one instruction per sentence, at most 20–25 words per sentence, active voice, no dropped articles or
   verbs, and noun clusters of at most 3 words ([ASD](https://www.asd-europe.org/standards-specifications/simplified-technical-english/what-are-the-basics-of-simplified-technical-english/),
   [Wikipedia](https://en.wikipedia.org/wiki/Simplified_Technical_English)). The CNL literature names
   information extraction as a main reason for controlled language ([CNL survey, arXiv 1507.01701](https://arxiv.org/pdf/1507.01701),
   [FrameNet CNL, arXiv 1406.2538](https://arxiv.org/pdf/1406.2538)). No source measures STE's effect on
   extraction directly.
3. **Consistent entity names and a glossary.** LLM KG pipelines spend much of their effort on deduplicating and
   resolving entities ([NVIDIA](https://developer.nvidia.com/blog/insights-techniques-and-evaluation-for-llm-driven-knowledge-graphs/)).
   Docs that already use one canonical name per thing remove that step.
4. **One page type per page (Diátaxis).** The four types are tutorial, how-to, reference and explanation. Mixing
   them ("content drift") hurts both readers and RAG. A retriever can then load only the type a question needs
   ([Diátaxis + AI](https://pasqualepillitteri.it/en/news/5528/diataxis-framework-documentation-ai),
   [Canonical](https://ubuntu.com/blog/diataxis-a-new-foundation-for-canonical-documentation)).
5. **Examples beat descriptions** for LLM use of docs ([arXiv 2509.19931](https://arxiv.org/pdf/2509.19931)).
6. **Readable inline triples already exist.**
   - Dataview's `key:: value` and `[key:: value]` are the most widely used human-readable triple syntax in
     Markdown, and Logseq uses the same form ([Dataview docs](https://blacksmithgu.github.io/obsidian-dataview/annotation/add-metadata/)).
   - Formal options with RDF semantics: [MD-LD](https://github.com/davay42/mdld-parse) (`[Alice Smith] {ex:fullName}`),
     [Semantic Markdown](https://blog.sparna.fr/2020/02/20/semantic-markdown/), [LOKF](https://github.com/nicholsn/lokf).
7. **Machine-added context instead of rewriting.** Anthropic's Contextual Retrieval prepends a 50–100 token
   LLM-written context to each chunk before indexing. It cut top-20 retrieval failures by 35% (embeddings),
   49% (+ BM25) and 67% (+ rerank) on their benchmarks
   ([Anthropic](https://www.anthropic.com/engineering/contextual-retrieval)). It costs LLM tokens at index time,
   so it does not fit the "no token burn on refresh" goal. Well-written sections get most of that benefit for free.
8. **llms.txt** is an index of the important pages, each with a one-line description. It helps agents find pages;
   it does nothing for extraction inside a page ([llmtxt.info](https://llmtxt.info/best-practices/),
   [Mintlify](https://www.mintlify.com/blog/real-llms-txt-examples)).

## 2. How our own pipeline reads a doc (what the rules must serve)

| pipeline step | what it uses | writing rule that helps |
|---|---|---|
| section splitter (`defrost_ai/ingest/documents.py`) | Markdown / RST / AsciiDoc headings | one topic per heading; no giant sections; 60–400 words is the range the models were trained on |
| retriever + reranker | `heading path + text` | descriptive headings ("Retry policy for failed webhooks", not "Notes"); name the entity in the first sentence |
| doc→code linker (`ingest/links.py`) | backticked spans, file paths, camelCase / snake_case words resolved against graphify's AST | always backtick real identifiers exactly as in code (`charge_invoice()`, `src/billing/retry.py`) |
| BM25 | literal words | use the user's words and the canonical term together once ("refund (reversal)") |
| a future triple extractor | explicit subject, relation, object | the Facts block (section 4) |

## 3. Choice matrix

Scores run from 1 (poor) to 5 (best). **Read** = how natural it is for a human. **Extract** = how reliably a
regex or a small model turns it into facts or the right retrieval unit. **Cost** = extra author effort.
**Tokens** = overhead on the text itself.

| # | technique | Read | Extract | Cost | Tokens | tooling today | verdict |
|---|---|---|---|---|---|---|---|
| 1 | Self-contained sections: name the subject, no cross-section "it/this/above" | 5 | 4 | low | +0–5% | none needed | **adopt** |
| 2 | Descriptive heading per topic, one topic per section (60–400 words) | 5 | 4 | low | 0 | our splitter | **adopt** |
| 3 | Canonical names + glossary page; backtick every identifier | 5 | 4 | low | 0 | our linker | **adopt** |
| 4 | STE-lite sentences: active voice, ≤25 words, one fact or instruction each | 4 | 4 | medium | 0 / shorter | ASD-STE100 skill | **adopt (lite)** |
| 5 | Diátaxis: one page type per page | 5 | 3 | medium (restructure) | 0 | several skills | **adopt for new docs** |
| 6 | Tables / definition lists for reference data (options, defaults, limits) | 4 | 5 | low | ≈0 | Markdown | **adopt** |
| 7 | **Facts block of pseudo-triplets** at the end of a section (section 4) | 4 | 5 | low–medium | +5–15% | regex; Dataview-compatible | **adopt, 3–7 facts per section** |
| 8 | Dataview inline fields inside sentences `[owner:: billing-team]` | 3 | 5 | medium | +5–10% | Obsidian, Logseq | optional (if you use Obsidian) |
| 9 | YAML frontmatter (page-level: type, owner, status, version, updated) | 4 | 5 | low | small | everywhere | **adopt, page-level only** |
| 10 | Worked examples next to the rule | 5 | 3 | medium | +10–30% | none | adopt where non-obvious |
| 11 | llms.txt index of key pages | 5 | 2 | low | separate file | agents, crawlers | optional, public docs |
| 12 | MD-LD / Semantic Markdown RDF annotations | 2 | 5 | high | +15–30% | parsers exist | only if you need real RDF |
| 13 | Attempto Controlled English (formal CNL) | 2 | 5 | very high | 0 | ACE parser | no |
| 14 | JSON-LD / RDF sidecar files | 1 (not prose) | 5 | high | separate | full | no, generate it instead |
| 15 | Contextual Retrieval (LLM-written context per chunk at index time) | n/a (machine) | 4 | none for authors | +50–100 tokens/chunk; **LLM cost per refresh** | Anthropic cookbook | no (conflicts with zero-token refresh) |

**Ranking by combined value (Read + Extract, with cost as tie-break):**
1. Self-contained sections (1)
2. Descriptive headings, one topic each (2)
3. Canonical names, backticked identifiers (3)
4. Tables for reference data (6)
5. Facts block of pseudo-triplets (7)
6. YAML frontmatter (9)
7. STE-lite sentences (4)
8. Diátaxis page types (5)
9. Worked examples (10)
10. Dataview inline fields (8)
11. llms.txt (11)
12. MD-LD (12), then RDF sidecars (14), ACE (13) and Contextual Retrieval (15) last for this goal

```
 Extract ▲
       5 │  (12)MD-LD  (13)ACE        (8)Dataview   (6)Tables (7)Facts (9)YAML
       4 │                    (15)CtxRetrieval  (4)STE-lite    (1)(2)(3)
       3 │                                          (5)Diátaxis (10)Examples
       2 │                                                     (11)llms.txt
         └──────────────────────────────────────────────────────────────▶ Read
            1          2          3                4                5
```

The top-right corner holds the cheap, readable techniques (1, 2, 3, 6, 7, 9). Formal RDF syntaxes buy little extra
extraction for a large loss in readability.

## 4. The recommended pseudo-triplet format

Requirements: readable as a plain bullet list, parseable with one regex, the subject explicit, few extra tokens.

**Format: a `Facts:` list at the end of a section. Each line holds one fact, written as subject, relation, object.**

```markdown
## Retry policy for failed webhooks

`WebhookDispatcher` retries a failed delivery 5 times with exponential backoff (1 s, 2 s, 4 s, 8 s, 16 s).
After the fifth failure, `WebhookDispatcher` moves the event to the `webhook_dead_letter` table and sends an
alert to the on-call channel. Retries stop early when the endpoint returns 410 Gone.

Facts:
- `WebhookDispatcher` → retries → failed deliveries (max 5, exponential backoff)
- `WebhookDispatcher` → writes to → `webhook_dead_letter`
- `webhook_dead_letter` → stores → events that failed 5 deliveries
- HTTP 410 response → stops → webhook retries
```

Rules:
- The subject and object are canonical names. Code entities are backticked so they link to graphify code nodes.
- The relation is a short verb phrase from a small vocabulary, kept in the glossary: `calls`, `uses`, `reads`,
  `writes to`, `owns`, `depends on`, `configures`, `replaces`, `deprecated by`, `requires`, `stops`, `defaults to`.
- Use 3–7 facts per section, and only for the key facts. The prose stays the source of truth; the Facts block
  is a summary, not a second copy.
- Qualifiers go in parentheses after the object.

**Parsing:** use one regex, ``^- (.+?) → (.+?) → (.+?)(?: \((.*)\))?$``. The `→` arrows keep it unambiguous.
Authors can type `->`; the parser accepts both.

**Alternative for Obsidian users:** Dataview fields under the section, with the heading's entity as the subject.

```markdown
writes-to:: `webhook_dead_letter`
max-retries:: 5
```

These are queryable in Obsidian for free, but the subject is implicit (the page or heading), so cross-entity
facts need the arrow form.

**Page-level metadata goes in YAML frontmatter:**

```yaml
---
type: reference        # tutorial | how-to | reference | explanation (Diátaxis)
entity: WebhookDispatcher
owner: platform-team
status: current        # current | deprecated
updated: 2026-10-01
---
```

## 5. How to measure it on our stack (no Claude tokens needed)

1. Take 30 sections from the e2e repos and rewrite each in two styles: (a) the self-contained / STE-lite profile
   only, and (b) the same plus a Facts block. Do the rewriting by hand, or with an LLM only after confirmation.
2. Rebuild the memory for each style and rerun the existing e2e questions locally: section hit@5, nDCG@10, the
   rate of doc→code links resolved, and tokens per section.
3. Extraction: run the regex over the Facts blocks, then score precision and recall against a small hand-labelled
   gold set (about 100 facts).

Adopt a style only if it does not lower retrieval and it raises link or triple yield.

## 6. CLAUDE.md block for AI-written docs (paste-ready)

```markdown
## Writing docs (humans first, extractable by machines)
- One topic per section; a descriptive heading that names the topic; 60–400 words per section.
- Make each section self-contained: name the subject in the first sentence; no "it/this/the above" across sections.
- Use one canonical name per thing (see docs/GLOSSARY.md); backtick every code identifier and path exactly as in code.
- Sentences: active voice, at most 25 words, one fact or instruction each; numbered steps for procedures.
- Put options, defaults and limits in tables, not prose.
- End reference and explanation sections with "Facts:" and 3–7 lines of the form
  `- Subject → relation → Object (qualifier)`, using only relations from docs/GLOSSARY.md.
- One Diátaxis type per page (tutorial, how-to, reference or explanation); declare it in frontmatter `type:`.
- Never copy a fact into two places; link to the section that owns it.
```

## Sources

- Anthropic, [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval);
  [Claude cookbook guide](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)
- [From Ambiguity to Accuracy: coreference resolution in RAG (arXiv 2507.07847)](https://arxiv.org/pdf/2507.07847);
  [CLAP (arXiv 2508.06941)](https://arxiv.org/pdf/2508.06941)
- ASD-STE100: [ASD](https://www.asd-europe.org/standards-specifications/simplified-technical-english/what-are-the-basics-of-simplified-technical-english/),
  [Wikipedia](https://en.wikipedia.org/wiki/Simplified_Technical_English),
  [Claude Code skill (danyuchn)](https://github.com/danyuchn/asd-ste100-skill)
- Controlled natural language: [survey (arXiv 1507.01701)](https://arxiv.org/pdf/1507.01701),
  [FrameNet CNL (arXiv 1406.2538)](https://arxiv.org/pdf/1406.2538),
  [Wikipedia](https://en.wikipedia.org/wiki/Controlled_natural_language)
- Diátaxis: [with AI](https://pasqualepillitteri.it/en/news/5528/diataxis-framework-documentation-ai),
  [Canonical](https://ubuntu.com/blog/diataxis-a-new-foundation-for-canonical-documentation),
  [developer-docs-framework skill](https://github.com/anivar/developer-docs-framework),
  [Lespinasse skill](https://www.romainlespinasse.dev/posts/diataxis-documentation-skill/)
- Inline triples: [Dataview metadata](https://blacksmithgu.github.io/obsidian-dataview/annotation/add-metadata/),
  [Logseq inline properties](https://discuss.logseq.com/t/syntax-for-inline-properties/10031),
  [MD-LD](https://github.com/davay42/mdld-parse), [Semantic Markdown](https://blog.sparna.fr/2020/02/20/semantic-markdown/),
  [LOKF](https://github.com/nicholsn/lokf)
- KG from text: [NVIDIA](https://developer.nvidia.com/blog/insights-techniques-and-evaluation-for-llm-driven-knowledge-graphs/),
  [CocoIndex docs KG](https://cocoindex.io/blogs/knowledge-graph-for-docs/)
- Docs for LLMs: [arXiv 2509.19931](https://arxiv.org/pdf/2509.19931), [llms.txt best practices](https://llmtxt.info/best-practices/),
  [Mintlify llms.txt examples](https://www.mintlify.com/blog/real-llms-txt-examples)
