---
name: extraction-ready-docs
description: Write and maintain technical documentation that people read easily AND that search, RAG and knowledge-graph tools extract well (one topic per section, exact backticked identifiers, Facts triples, linter). Use this skill whenever the user writes, edits, restructures or reviews docs, READMEs, ADRs, runbooks or `docs/**` pages in a code project, asks to make docs "AI-friendly", "RAG-ready", "searchable" or "good for agents", wants documentation rules or a docs style guide added to CLAUDE.md / AGENTS.md for future sessions, or wants to set up doc rules in a new project — even if they don't mention extraction or this skill by name. Works with or without defrost-ai.
---

# Extraction-ready docs

Docs today have two readers: people, and tools that split docs into sections before anyone reads them (BM25,
embedding search, rerankers, RAG, triple extractors). These rules keep Markdown normal to read while making each
section findable on its own and each key fact extractable without an LLM pass. The reasoning behind every rule is
in `references/WHY.md`; read it when you need to explain or adapt a rule.

The skill has two jobs. Pick the one the user asked for, or both.

## Job 1: set the rules up in a project (so every future session follows them)

Run the installer from this skill's folder against the project root:

```sh
python <skill-dir>/scripts/install.py <project>              # CLAUDE.md
python <skill-dir>/scripts/install.py <project> --agents-md  # also AGENTS.md, for non-Claude agents
python <skill-dir>/scripts/install.py <project> --remove     # take the block out again
```

It copies the full rules to `docs/DOC_RULES.md`, the page and glossary templates to `docs/templates/`, and the
linter and Facts extractor to `docs/tools/`. Then it puts a six-rule block at the **end** of `CLAUDE.md`. The block
is short on purpose: `CLAUDE.md` loads into every session, so it holds only the rules an agent must never skip and
a pointer to the full rules, which are read only when a doc is written. Don't `@import` `DOC_RULES.md` from
`CLAUDE.md` — that would load it into every session.

Re-running replaces the block instead of duplicating it. If the project uses defrost-ai, `defrost setup` writes the
same block between the same markers, so the two never conflict. Tell the user to commit `CLAUDE.md` and `docs/` so
teammates and future sessions get the rules.

Use Python 3 (`python` on Windows, `python3` on macOS/Linux if `python` is missing). Nothing else is needed.

## Job 2: write or edit docs by the rules

Read `references/DOC_RULES.md` (about 1k tokens) before writing — if the project already has `docs/DOC_RULES.md`,
that copy wins, since the team may have adapted it. The core of it:

1. **One topic per section**, heading named the way a reader would search ("Retry policy for failed webhooks", not
   "Notes"), 60–400 words.
2. **Self-contained sections:** name the subject in the first sentence; never open with "It", "This", "These",
   "Here" or "As mentioned above". Search returns sections alone.
3. **Exact names:** backtick every identifier and path exactly as in the code; one canonical name per concept,
   aliases in `docs/GLOSSARY.md`.
4. **Plain sentences:** active voice with a named actor, ≤ 25 words, one fact or one instruction each; options,
   defaults, limits and error codes go in tables.
5. **Facts block** at the end of each reference and explanation section, 3–7 lines:

   ```markdown
   Facts:
   - `WebhookDispatcher` → retries → failed deliveries (max 5, exponential backoff)
   - `MAX_RETRIES` → defaults to → 5
   ```

   Relations come from the glossary's relation table; add a row there before inventing one.
6. **Frontmatter** with `type` (`tutorial` | `how-to` | `reference` | `explanation`), `entity`, `status`, `updated`;
   one page type per page.

For a new page, start from `assets/PAGE.template.md` (or `docs/templates/PAGE.template.md` once installed).
Create `docs/GLOSSARY.md` from the glossary template only when adding the first real term, so no placeholder page
gets indexed.

Before you finish, lint what you changed and fix every ERROR. WARNs are judgment calls: a 45-word section that
holds a table is fine; a 600-word section about two topics should be split.

```sh
python docs/tools/doc_lint.py <changed files or dirs>       # installed copy
python <skill-dir>/scripts/doc_lint.py <files or dirs>       # without installing
```

To get the Facts as graph triples (JSONL with `subject`, `relation`, `object`, `qualifier`, `path`, `line`):

```sh
python docs/tools/extract_facts.py docs > facts.jsonl
```

## Editing existing docs

Don't rewrite a whole doc set unprompted. When editing a page, bring the sections you touch up to the rules and
leave the rest. When the user asks for a full conversion, go page by page: split multi-topic sections, rename vague
headings, add backticks and Facts, then lint. Keep the meaning identical — the rules change form, not content — and
when a fact looks wrong or out of date against the code, flag it to the user instead of silently "fixing" it.

A README that is the project's landing page may skip the frontmatter; say so rather than forcing one in.
