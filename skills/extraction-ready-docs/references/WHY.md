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
| One home per file, folder map and indexes | An agent that finds two files on one topic retrieves two answers and has to guess. A map and one-line indexes let it go straight to the right folder. |
| What-who-when names, full `YYYY-MM-DD` dates | File names are the first thing search and agents see. `interview-maya-2026-10-12.md` is found by person, kind and date; `notes (2).md` by nothing. Partial dates (`2026-10`) sort and match ambiguously. |
| Codes for 100+ files (optional) | In large sets, a defined prefix (`P1_MR_INT_01-…`) groups and sorts files where folders alone get deep. The legend keeps codes searchable; the words after the code keep the name readable. |
| No `_v2` / `final` copies; one lifecycle per file | Three versions of one document are three conflicting "truths" for retrieval, however well each is written. A declared lifecycle says whether history lives in git, in `archive/`, or nowhere because the file never changes. |
| `source: true` files with a Sources section | Answers are only as trustworthy as their origin. Marking where facts first enter the project, and citing those files elsewhere, lets a reader or agent check any claim in one hop. |
| Text twin for binaries | Search and embedding tools skip PDFs, images and decks. A same-name `.md` makes their content findable. |
| Decision register | Decisions in chat are invisible to search and get argued again. One register row per decision, never deleted, is the history. |
| Folder splits at ~12 files or 3 shared prefixes | Crowded folders slow people and agents scanning by name; shared first words show the natural sub-folder. |
| Short always-loaded block, full rules read on demand | `CLAUDE.md` / `AGENTS.md` is loaded into every agent session. Eight short rules cost little; the full rules (~3k tokens) are read only when a doc is written or a file moved. |

The rules borrow from Diátaxis (page types), plain-language and controlled-English guides such as ASD-STE100
(short active sentences, one name per concept), and from how retrieval and triple-extraction pipelines split and
score text.
