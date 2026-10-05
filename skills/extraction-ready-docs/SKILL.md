---
name: extraction-ready-docs
description: A complete, easy-to-maintain documentation layer for any repository — how pages are written (one topic per section, exact backticked identifiers, Facts triples) AND how the docs are organised (one home per file, what-who-when file names with full dates, no _v2/final copies, per-file lifecycle living/versioned/immutable, a source flag on files where facts first enter, decision register, folder splits, safe moves). Use this skill whenever the user writes, edits, reorganises, renames, versions or reviews docs, READMEs, ADRs, research notes, interview transcripts or `docs/**` files; asks where a document should go or what to call it; mentions duplicate or outdated copies, "final_v3" chaos, messy docs folders, or docs that are hard to search; wants docs to be "AI-friendly", "RAG-ready" or "look once, find the answer"; or wants documentation rules in CLAUDE.md / AGENTS.md for future sessions, even unnamed. Works in any repository; an optional defrost mode adds linters for defrost-ai projects.
---

# Extraction-ready docs

Docs have two readers: people, and tools that split them into pieces before anyone reads them (keyword and
embedding search, RAG, triple extractors). Two layers make docs work for both:

- **Pages** (`references/DOC_RULES.md`): how a page is written — sections, sentences, names, Facts, frontmatter.
- **Knowledge** (`references/KNOWLEDGE_RULES.md`): where files live, what they are called, how many copies exist,
  how they change over time, and where their facts came from.

A well-written page still fails if three versions of it exist. The goal is **look once, find the answer**. The
reasoning behind every rule is in `references/WHY.md`; read it when you need to explain or adapt a rule.

The skill has two jobs. Pick the one the user asked for, or both.

## Job 1: set the rules up in a project

Run the installer from this skill's folder against the project root. It needs only Python 3 (`python` on
Windows, `python3` on macOS and Linux).

```sh
python <skill-dir>/scripts/install.py <project>              # set up (any repository)
python <skill-dir>/scripts/install.py <project> --defrost    # defrost-ai projects only
python <skill-dir>/scripts/install.py <project> --remove     # undo the rule block
```

Optional on set-up: `--lifecycle living` or `--lifecycle versioned` (see below).

**Set-up** fits any repository. It copies both rule files to `docs/`, the templates to `docs/templates/`, creates
the `docs/README.md` map and `docs/decision-register.md` when they are missing (existing files are never
overwritten), and puts an eight-rule block at the end of `CLAUDE.md` — and of `AGENTS.md` when the project has one,
for non-Claude agents. Fill in the placeholders in the map and the register for the project. The block is short on
purpose:
`CLAUDE.md` loads into every session, so it holds only what an agent must never skip, plus pointers to the full
rules, which are read only when a doc is written or a file is moved. Don't `@import` the rule files from
`CLAUDE.md`; that loads them into every session.

**Lifecycle mode** says how files change; ask the user if they have a preference, otherwise keep the default:

| mode | meaning |
|---|---|
| `per-document` (default) | you choose `living`, `versioned` or `immutable` for each page and write it into its frontmatter |
| `living` | edit in place; git keeps history; no copies |
| `versioned` | superseded copies go to `archive/`; the current file keeps the clean name |

**Defrost mode** (`--defrost`) is only for projects indexed by defrost-ai. It also copies the linters and the
Facts extractor to `docs/tools/`, adds a ninth rule ("lint before you finish"), and writes the block between
defrost-ai's markers so `defrost setup` and this installer update the same block. Don't use it elsewhere.

Re-running replaces the block instead of duplicating it. If the project's `CLAUDE.md` already holds hand-written
doc rules, merge them into the block's wording rather than leaving two rule sets.

**Then bring the existing docs in line.** The installer ends by scanning every `.md` file in the repository
(skipping `CLAUDE.md`, `AGENTS.md`, `CHANGELOG.md`, vendored folders and templates). Get the full list with
`python <skill-dir>/scripts/audit.py <project>`; it splits the work in two:

- **fix** items change only the inside of a file — add frontmatter (choose `type`, `entity`, `lifecycle` and
  `source` per page), rename vague headings, split or merge sections, shorten long sentences, backtick
  identifiers, add Facts blocks, add the file to its folder index and the docs map, fix broken references. Apply
  these right away, keeping the meaning identical. Pages marked `lifecycle: immutable` (raw sources) get frontmatter
  and, when `source: true`, an appended `## Sources` section — nothing else; their body is evidence and stays
  untouched, even if it breaks the writing rules. A landing `README.md` keeps its own shape.
- **ASK** items change paths other people rely on — renames, moves into `docs/`, merges of duplicate copies,
  folder splits. List them as one proposal (old path → new path, and why) and wait for a yes; then follow
  KNOWLEDGE_RULES §8. Decisions found in old notes go to the decision register.

In a large repository, work folder by folder and say how far you got. Rerun the audit at the end: no ERROR should
remain in the files you fixed. Tell the user to commit `CLAUDE.md` and `docs/` so teammates and future sessions get
the rules.

## Job 2: write, organise and maintain docs by the rules

Read `references/DOC_RULES.md` before writing a page and `references/KNOWLEDGE_RULES.md` before creating, naming,
versioning, moving or splitting files. If the project has its own copies in `docs/`, those win: the team may have
adapted them. Find the lifecycle mode in the `CLAUDE.md` block; without a block, use per-document.

**Writing a page** (DOC_RULES): one topic per section, heading named the way a reader searches, 60–400 words;
self-contained sections that name their subject first; identifiers backticked exactly as in code; active
sentences of at most 25 words; options and limits in tables; a 3–7 line Facts block
(`- Subject → relation → Object (qualifier)`) at the end of reference and explanation sections; frontmatter with
`type`, `entity`, `status`, `updated`, plus `lifecycle` and `source` from the knowledge rules. Start new pages
from `assets/PAGE.template.md`.

**Before creating a file** (KNOWLEDGE_RULES):
1. Look in `docs/README.md` and the folder's `README.md`. If a file on the topic exists, extend it.
2. Name it what-who-when: lowercase words and hyphens, a full `YYYY-MM-DD` date at the end only for one-time
   things, no `_v2` / `final` / `copy`. In large doc sets the team may use codes defined in `docs/CODES.md`
   (`P1_MR_INT_01-maya-2026-10-12.md`).
3. Set `lifecycle:` (per-document mode) and `source: true` if facts first enter the project in this file — then
   give it a `## Sources` section saying where each fact came from (person and date, link and date read, dataset,
   experiment).
4. Add one line for it to the folder `README.md`. A binary (PDF, image, deck, spreadsheet) also gets a same-name
   `.md` text twin.

**Changing a file:** follow its lifecycle. Living: edit in place, update `updated:`. Versioned: archive a
superseded copy first (`assets/ARCHIVE_BANNER.md`). Immutable: don't edit; write a correction note. Never create
a second current file for the same entity.

**Claims:** cite the source file (and line when it helps) for every claim about users, numbers or quotes. Never
invent a source; with none, say "no source yet".

**Decisions** the user makes go in the decision register (`assets/DECISION_REGISTER.template.md`). Record them;
don't decide for the team. When two docs disagree, show both and ask.

**Moves, renames and folder splits** change other people's paths, so propose them and wait for a yes. Then copy,
verify, delete the original, update every reference and folder index (KNOWLEDGE_RULES §7–8).

## Checking the result

In defrost mode, run both linters and fix every ERROR; WARNs are judgment calls (a 45-word section holding a
table is fine; a 600-word section about two topics should be split):

```sh
python docs/tools/doc_lint.py <changed files or dirs>   # page rules
python docs/tools/repo_lint.py                          # names, copies, lifecycle, sources, indexes, twins, refs
```

In general mode the linters are optional; run them from this skill's `scripts/` folder when a doc set is large or
the user asks for a check. Otherwise review against the rules by reading. `scripts/extract_facts.py docs` turns
Facts blocks into JSONL triples for a knowledge graph.

## Editing existing doc sets

Don't reorganise a whole doc set unprompted. When editing a page, bring the parts you touch up to the rules and
leave the rest. For a full clean-up, first list the proposed renames, moves, merges of duplicate copies and
lifecycle choices, get a yes, then work folder by folder. Keep the meaning identical — the rules change form, not
content — and flag facts that look wrong against the code or their sources instead of silently "fixing" them.

A landing `README.md` may skip the frontmatter; give it a "Start here" table (rules, docs map, decision register,
the most-used pages) instead.
