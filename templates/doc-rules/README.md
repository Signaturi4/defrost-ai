# Doc rules kit

Writing rules, templates and two small tools that keep a project's documentation easy to read **and** easy for
retrieval and knowledge-graph tools to use (BM25, embeddings, rerankers, triple extraction). The kit works for any
technical or coding project. It needs Python 3 for the tools and nothing else.

Why these rules, with the research behind them: [../../docs/WRITING_FOR_EXTRACTION.md](../../docs/WRITING_FOR_EXTRACTION.md).

## Install

```sh
bash templates/doc-rules/install.sh /path/to/your/project
```

The installer:
- copies `DOC_RULES.md` → `docs/DOC_RULES.md` (the full rules; read only when writing docs);
- creates `docs/GLOSSARY.md` from the template (if it does not exist yet);
- copies `PAGE.template.md` → `docs/templates/` and the tools → `docs/tools/`;
- appends a 13-line, highlighted rule block to the **end** of `CLAUDE.md`.

Re-running the installer replaces the block instead of duplicating it. The block sits between
`<!-- defrost-ai:doc-rules:start/end -->` markers. To install by hand, paste `CLAUDE.snippet.md` at the end
of your `CLAUDE.md` and copy the files yourself.

## Why the CLAUDE.md block is short

CLAUDE.md is loaded into every session, so it holds only the six rules an agent must never skip, plus a pointer.
The full rules (`docs/DOC_RULES.md`, about 1k tokens) are read only when the agent is about to write or edit a doc.
Do not `@import` them: an import loads them into every session.

## Files

| file | goes to | purpose |
|---|---|---|
| `CLAUDE.snippet.md` | end of `CLAUDE.md` | the highlighted, always-loaded rule block |
| `DOC_RULES.md` | `docs/DOC_RULES.md` | the full rules: pages, sections, names, sentences, tables, Facts |
| `GLOSSARY.template.md` | `docs/GLOSSARY.md` | canonical names + aliases, and the allowed Facts relations |
| `PAGE.template.md` | `docs/templates/` | starting point for a new page (frontmatter, section, table, Facts) |
| `tools/doc_lint.py` | `docs/tools/` | checks frontmatter, section length, opening pronouns, long sentences, un-backticked identifiers, Facts syntax |
| `tools/extract_facts.py` | `docs/tools/` | turns Facts blocks into JSONL triples (with code identifiers flagged) |

## Use

```sh
python docs/tools/doc_lint.py docs/**/*.md README.md     # exit code 1 on errors (use it in CI or a pre-commit hook)
python docs/tools/extract_facts.py docs > facts.jsonl    # {"subject", "relation", "object", "qualifier", "path", "line", ...}
```

The format agents and people write, at the end of a section:

```markdown
Facts:
- `WebhookDispatcher` → retries → failed deliveries (max 5, exponential backoff)
- `WebhookDispatcher` → writes to → `webhook_dead_letter`
```

## With defrost-ai

The rules match what the memory reads:
- **Sections:** one topic per heading, 60–400 words.
- **Headings:** they are embedded together with the text.
- **Backticked identifiers:** they become doc→code links into graphify's AST graph.
- **Facts:** the triples from `extract_facts.py` can be loaded next to the graph.

No LLM call is needed at index time.
