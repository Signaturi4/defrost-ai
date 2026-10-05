# Why the rules look like this

Docs are now read by two audiences: people, and tools that cut the docs into pieces before anyone reads them.
Search (BM25 keyword search, embedding search, rerankers) and RAG pipelines return **one section at a time**.
Knowledge-graph builders turn sentences into `subject → relation → object` triples. Each rule removes a common way
these tools fail, while keeping the Markdown normal to read.

| rule | what goes wrong without it |
|---|---|
| One topic per section, 60–400 words | Search returns a section, not a page. A section about three things matches none of them well; a 20-word section has too little text to match; a 1,500-word one dilutes its embedding. |
| Heading names the topic in searchable words | Most retrievers embed the heading with the text. "Notes" adds nothing; "Retry policy for failed webhooks" is half the match. |
| Self-contained, subject named first, no opening "It/This" | A section returned alone has no "above". A pronoun with no referent leaves the reader and the embedding not knowing what the section is about. |
| Exact backticked identifiers | Keyword search and code-linking tools match `charge_invoice()` exactly. A paraphrase ("the charge function") or a typo breaks the link between doc and code. |
| One canonical name per concept, glossary with aliases | Three names for one thing split the evidence three ways in search and create three nodes in a graph. Aliases in the glossary keep user wording findable. |
| Active voice, ≤ 25 words, one fact per sentence | Extractors read "`Scheduler` retries the job" as one triple. "The job is retried, unless it was cancelled, in which case…" yields no actor and tangled facts. |
| Options, defaults and limits in tables | Tables keep each value next to its name, so a lookup like "default of `MAX_RETRIES`" hits one row instead of a paragraph. |
| Facts block (`- Subject → relation → Object`) | It gives graph tools the key triples without an LLM pass over every page on every change. The prose stays the source of truth; the block is 3–7 lines of the most important facts. |
| Page types (tutorial, how-to, reference, explanation) | Mixing "how to do it" with "why it works" makes sections long and multi-topic. The type in the frontmatter also lets tools weight or filter pages. |
| Short always-loaded block, full rules read on demand | `CLAUDE.md` / `AGENTS.md` is loaded into every agent session. Six lines cost little; the full rules (~1k tokens) are read only when a doc is being written. |

The rules borrow from Diátaxis (page types), plain-language and controlled-English guides such as ASD-STE100
(short active sentences, one name per concept), and from how retrieval and triple-extraction pipelines split and
score text.
