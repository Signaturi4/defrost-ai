# defrost-ai: brand and positioning

This page covers the name, the promise, the voice and the visual rules. It is the reference for the README, the
release notes, the docs site, the social posts and any talk slides. Numbers quoted here come from
[RESULTS.md](RESULTS.md) and [E2E_GRAPHIFY.md](E2E_GRAPHIFY.md). Update both places together.

## 1. Core concept

**Your project's knowledge is frozen in docs nobody reads. defrost-ai thaws it into a memory your AI tools can
search.**

Docs, READMEs, ADRs and runbooks hold the "how" and "why" of a codebase. AI agents rarely reach them: grep finds
names, not explanations, and graph tools turn paragraphs into nodes that lose the conditions and exceptions.
defrost-ai indexes every doc section locally, links it to the code it names, and returns the paragraph that answers
the question, with `path:Lstart-end`.

The metaphor is **frozen → thawed**: knowledge that exists but cannot be reached becomes usable again. Use it
lightly. It names the product and the tagline; it is not a theme for every sentence.

## 2. Positioning

| | |
|---|---|
| **Category** | Local project memory for AI coding tools (retrieval over docs + linked code) |
| **For** | Developers and teams who use Claude Code, Cursor or other MCP clients on real codebases with real docs |
| **Who** | find that agents ignore their docs, cost too much per question, or answer "how/why" questions from code alone |
| **defrost-ai is** | an open-source library, CLI and MCP server that turns a repo's docs into a searchable, code-linked memory |
| **Unlike** | grep (finds names, not explanations), knowledge-graph tools (lose paragraph detail; paid LLM extraction), hosted RAG (sends code to a third party) |
| **It** | runs entirely on your machine, costs $0 to build, and returns the answering doc section 94% of the time on repos it never saw |

**Positioning statement (one line):** The local, open-source memory that lets AI agents read your docs the way
your senior engineers do.

## 3. Value proposition

### Headline

**Docs your AI actually reads.**

### Sub-headline

Local, code-aware memory for Claude Code and any MCP client. One install, one question, cited answers.

### Three pillars

| pillar | promise | proof |
|---|---|---|
| **Finds the answer, not the keyword** | Returns the doc paragraph that answers a how/why question, plus the code it names | 94% vs 6% "answering section in context" against stock graphify; RAGAS 0.945 vs 0.337 |
| **Free and private** | Builds and searches on your laptop; no API calls, no tokens, no data leaving the machine | $0 build vs $11.87 of Claude usage for the same 3 repos; 0.5B model, 141 MB of adapters |
| **Stays fresh by itself** | Refreshes on merge to main (or on a schedule) and only re-embeds what changed | 4–17 s incremental refresh on a 565-section repo |

### Benefits by audience

| audience | benefit |
|---|---|
| **Individual developers** | Claude answers "why does X work like this" from your docs with citations, 25% cheaper per question than stock graphify in our agent eval |
| **Teams** | New members and agents get the same answers the docs give; `.mcp.json` + slash commands are committed, so setup is shared |
| **Doc writers** | A rule kit for CLAUDE.md that makes docs easy for both people and retrieval to use, plus a linter |
| **Privacy-sensitive orgs** | Client code and internal docs never leave the machine; weights are open and checksummed |
| **RAG / chatbot builders** | A small, measured retriever + reranker with an HTTP API and Python library, usable for FAQ bots (`hybrid` ~45 ms) |

### What we do not claim

Honesty is part of the brand. Say these plainly wherever results appear:
- On small repos, a strong agent with grep is about as accurate and is the cheapest option.
- A reranked search takes about 1.7 s on an Apple M5 (MLX); BM25-only and hybrid searches take milliseconds.
- Multi-hop questions are unsolved, and a larger public reranker still wins on long narrative prose.

## 4. Name and naming system

| element | rule |
|---|---|
| **Product** | `defrost-ai`, always lowercase, hyphenated, in code and prose. At the start of a sentence, still `defrost-ai`. |
| **Short form** | "defrost" (CLI talk, MCP server name `defrost`). |
| **Package / CLI** | Package `defrost-ai` (import `defrost_ai`), CLI `defrost`. The old name `kev-memory` stays as a command alias, and `KEV_MEMORY_*` / `~/.kev-memory` are still read, until 1.3. |
| **Models** | **Defrost-Ret-B** (retriever) and **Defrost-Rerank v2** (reranker). Models are ingredients; the product is defrost-ai. Weight folders are `defrost-ret-b/` and `defrost-rerank/`; the v1.1.0 archive still uses the old folder names, which the loader accepts. |
| **Commands** | `/defrost-setup` for setup; `/memory-*` for daily use (search, update, domains). Verbs, not nouns. |
| **Versions** | Software and weights share SemVer (`v1.1.0` = Defrost-Rerank v2 weights). A weights change that moves metrics is at least a minor bump. |

Avoid: "AI brain", "second brain", "AGI", "revolutionary", ice puns in technical docs.

## 5. Voice and tone

Follow the README style already in use: simple technical English that addresses the reader directly and shows code
early.

| do | don't |
|---|---|
| State the developer's problem first, then the fix | Open with a slogan |
| Give a number with its condition and CI ("0.840 nDCG@10 on 106 held-out questions") | Write "blazing fast", "best-in-class", "10x" without a measurement |
| Name the limits in the same section as the wins | Hide caveats in a footnote |
| Use "you" and short sentences | Use passive voice and stacked adjectives |
| Credit what we build on (Qwen2.5, graphify, KG-BiLM) | Imply we built everything |

Example, good: *"Claude ignored our MCP tools until CLAUDE.md told it to use them. Setup now adds that rule for you."*
Example, bad: *"Unleash the power of AI-native knowledge with defrost!"*

## 6. Visual identity

| element | choice | why |
|---|---|---|
| **Logo mark** | A snowflake whose lower arms turn into a code bracket `{` `}` / a document line, drawn on a 24 px grid with 2 px strokes | frozen → usable; reads at favicon size |
| **Wordmark** | `defrost-ai` in Geist Mono (or JetBrains Mono), lowercase, medium weight | matches the CLI-first product; free fonts |
| **Primary color** | Thaw blue `#2F6FEB` | trust, ice; passes WCAG AA on white |
| **Accent** | Warm amber `#F5A524` | the "heat" that thaws; use for highlights and the "after" state only |
| **Neutrals** | `#0B0F14` (ink), `#5B6573` (muted), `#F4F6F8` (frost background) | terminal-friendly dark and light modes |
| **Charts** | Baseline in neutral gray, ours in thaw blue, direct labels, no legends, CIs as thin bars | same rules as the project's existing chart style |

Assets (in `docs/brand/`, rebuilt by `scripts/brand_assets.py` from `source-artwork.jpg`):

| file | use |
|---|---|
| `app-icon-1024.png` | square, unmasked, full-bleed: Icon Composer, app stores (the system applies the corner mask, per the HIG) |
| `logo-512.png`, `logo-256.png` | masked (continuous-corner squircle, transparent outside): README header, docs, slides |
| `favicon.ico` (16–64), `favicon-32.png` | browser tab icon for a docs site |
| `apple-touch-icon.png` | 180×180 unmasked, iOS home-screen bookmark |
| `social-preview.png` | 1280×640 GitHub social preview: icon, wordmark, tagline, the 94% vs 6% stat |

The icon carries its own depth and glow from the source artwork. For a native app, rebuild it in Icon Composer as
layers (background gradient, white snowflake, amber braces) and let the system add highlights and the dark/tinted
variants. Still to produce: a terminal GIF of `/defrost-setup` → first cited answer (≤ 20 s).

## 7. Messaging hierarchy (where each line goes)

| surface | message |
|---|---|
| GitHub description (≤ 120 chars) | Local, code-aware memory that lets Claude Code and MCP agents read your docs. Free, private, open source. |
| README first screen | Headline, one-line install, a real search output with citations, the 94% vs 6% table |
| Topics/tags | `mcp`, `claude-code`, `rag`, `retrieval`, `reranker`, `documentation`, `local-first`, `apple-silicon`, `knowledge-base` |
| Release notes | What changed, measured effect with CI, upgrade command, known limits |
| Social post | Problem (agents skip docs) → one number → GIF → repo link |
| Talk / blog | "Why a knowledge graph was the wrong place for answers" (DESIGN_NOTES) as the story; defrost-ai as the result |

## 8. Open-source brand practices

- **Trust signals on the first screen.** License (MIT code; Apache-2.0 base weights), CI badge, latest release,
  weights sha256, "runs offline".
- **Reproducible claims.** Every number in the README links to the script and suite that produced it; locked test
  splits are read once and labelled as such.
- **Contributor path.** `CONTRIBUTING.md` with: how to run the 10 unit tests, how to add a doc format, how to
  propose a model change (must report dev + locked-test deltas with CIs). Label "good first issue" items (doc
  formats, MCP client guides, Linux scheduler tests).
- **Governance.** `CODE_OF_CONDUCT.md` (Contributor Covenant), `SECURITY.md` (local service binds 127.0.0.1, no
  auth: report issues privately), a public roadmap issue.
- **Attribution.** Keep the "Built on" table: Qwen2.5-0.5B, graphify, KG-BiLM, the public training datasets.
- **Consistency.** One name everywhere (repo, package, CLI, MCP server) after the PyPI rename; redirects for the
  old names.

## 9. Launch plan

| phase | when | actions | success signal |
|---|---|---|---|
| **0. Foundation** | before announcing | logo + social preview, PyPI `defrost-ai`, CONTRIBUTING/SECURITY/CoC, README first screen per §7, sub-3 s search milestone | install-to-first-answer < 5 min on a clean Mac |
| **1. Soft launch** | week 1 | Show HN / r/LocalLLaMA / Claude Code community post with the 94% vs 6% result and the GIF; MCP server directories | 100 stars, 10 issues from real users |
| **2. Proof** | weeks 2–6 | 3 case studies (OSS repo, private monorepo, FAQ bot), a reproducible benchmark page, Cursor + Claude Desktop guides | 3 external repos using `/defrost-setup`; first outside PR |
| **3. Ecosystem** | months 2–3 | Hugging Face model cards for Defrost-Ret-B / Defrost-Rerank v2, graphify upstream PR for the memory integration, MLX/fast path release | weights downloads, upstream merge |

## 10. One-page summary

- **What:** open-source local memory for AI coding tools: doc sections + linked code, searchable via MCP, CLI, HTTP and Python.
- **Why:** agents answer from code and skip the docs; graph tools lose paragraph detail and cost tokens to build.
- **Proof:** 94% vs 6% answering-section hit rate, RAGAS 0.945 vs 0.337, $0 vs $11.87 build, 25% cheaper agent runs.
- **Promise:** Docs your AI actually reads.
- **Personality:** precise, honest about limits, developer-to-developer.
