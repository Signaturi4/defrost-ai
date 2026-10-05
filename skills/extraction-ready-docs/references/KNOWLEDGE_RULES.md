# Knowledge rules

`DOC_RULES.md` says how a page is written. These rules say where pages live, what they are called, how many
copies exist, how they change, and how their facts stay traceable. Goal: **look once, find the answer** —
anyone, or any agent, finds an answer in under a minute and can check where it came from.

## 1. One home per file

- **Map:** `docs/README.md` lists every top-level folder and what it owns, one row each.
- **Folder index:** each folder with more than 3 files has a `README.md` that lists its files, one line each.
  When you add a file, add its line in the same change.
- **Check before you create.** Search for an existing file on the topic and extend it. A second file on the same
  topic splits the answer in two.
- **One owner per fact.** Write a fact once and link to it. Example: the launch date lives in the roadmap; the
  pitch deck links to the roadmap instead of repeating the date.
- **No empty folders.** Create a folder when its first real file exists, not as a placeholder.

## 2. File names: what, who, when

Read the name out loud: it says **what** the file is, **who or what** it is about, and **when**, if it happened
once.

| ✅ Do | ❌ Don't |
|---|---|
| `interview-maya-2026-10-12.md` | `Interview Maya FINAL (2).md` |
| `survey-freelancers-2026-10-01.csv` | `survey results 10-2026.csv` |
| `competitors-time-tracking-apps.md` | `competitors_v3_FINAL.xlsx` |

- Lowercase words joined by hyphens. No spaces, `_`, `&`, brackets or version words (`v2`, `final`, `new`, `copy`).
- **Dates are always full: `YYYY-MM-DD`**, at the end of the name, and only for one-time things (an interview, a
  meeting, a survey). Living documents get no date.
- Confidential files end with `-confidential` (after the date): `pitch-deck-acme-2026-03-01-confidential.pdf`.
- Standard files keep their usual names (`README.md`, `CLAUDE.md`, `CHANGELOG.md`); code follows its language.

### Coded names for large doc sets (optional, from about 100 files)

Past about 100 files, a short code prefix sorts and groups files better than folders alone. Use it only when the
team agrees, and define every code in `docs/CODES.md` before using it:

| code | meaning |
|---|---|
| `P1` | Phase 1 |
| `MR` | Marketing research |
| `INT` | Interview |

The name is the codes joined by `_`, a number, then the usual what-who-when words:
`P1_MR_INT_01-maya-2026-10-12.md` (Phase 1, marketing research, interview 1, with Maya). Pad numbers to two
digits so files sort in order. Keep the words after the code: a bare `P1_MR_INT_01.md` tells a search tool nothing.

## 3. Lifecycle: how a file changes

Each project picks one mode, written into the `CLAUDE.md` block. The default is **per-document**.

| mode | what happens | pick it when |
|---|---|---|
| `per-document` (default) | the agent decides per page with the table below and writes `lifecycle:` into the frontmatter | most projects |
| `living` | every page is edited in place; history lives in git | a git team that wants the least to search |
| `versioned` | old versions are kept in `archive/`, out of the search path | audits, signed material, or no git |

**Per-document choice:**

| page kind | `lifecycle:` | why |
|---|---|---|
| reference, how-to, README, glossary, insights, competitor list, roadmap | `living` | readers need the current state only |
| signed or sent material: contract, scope of work, sent pitch deck, approved spec, release notes | `versioned` | someone will ask "what did we agree on that date?" |
| raw evidence: transcripts, recordings, data exports, screenshots used as evidence | `immutable` | evidence must not change |
| decision register | `living`, rows never deleted | the register is its own history |

**Living page:** edit it in place, update `updated:`, and write *why* in the commit message. Never create a copy
with `_v2`, `-final`, `.bak` or a date suffix. Old versions come from `git log -p <file>`.

**Versioned page:** the current file keeps the clean name.
1. Before a material change, copy it to `archive/<same path>/<name>--superseded-YYYY-MM-DD.md`.
2. In the copy, set `status: superseded` and `superseded_by: <path>`, and put the banner from
   `ARCHIVE_BANNER.md` at the top.
3. Edit the current file and add a line to its "Change log" section: date, what changed, why, who.

Search tools and agents skip `archive/`.

**Immutable page:** never edit its body after saving, even to meet the writing rules. You may add frontmatter and
an appended `## Sources` section. Corrections go in a separate note that links to it.

**Replaced page:** set `status: deprecated` and `replaced_by: <path>` for one review cycle, then delete it.

**Two current pages about one entity is always an error,** whatever the mode.

## 4. Source files and citations

A **source file** is the first file where a fact enters the project: an interview transcript, survey data, an
experiment log, a research write-up, a data export. Mark it in the frontmatter with `source: true`.

- A source file ends with a `## Sources` section that says where its facts came from, one line each:
  - a person: name, role and date ("Maya, freelance designer, interviewed 2026-10-12");
  - the web: link and the date it was read;
  - data: dataset, query or tool, and the date it was pulled;
  - an experiment: its ID and date.
- Every other file **cites the source file** for each claim about users, numbers or quotes, with the line when
  it helps: "Freelancers send invoices weekly (`interview-maya-2026-10-12.md`, L45)."
- Never invent a quote, number or reference. With no source, say "no source yet".
- Raw sources are usually both `source: true` and `lifecycle: immutable`.

## 5. Text twin for every binary

Search tools cannot read a PDF, image, deck or spreadsheet. Every binary used as knowledge gets a `.md` with the
same base name: `pitch-deck-acme-2026-03-01.pdf` → `pitch-deck-acme-2026-03-01.md`. The twin holds slide text as
sections, tables as Markdown tables, and a screenshot transcribed into what it shows.

## 6. Decisions go in the register

A decision that lives only in chat gets argued again. Keep one register, `docs/decision-register.md` (template:
`DECISION_REGISTER.template.md`), one row per decision: ID, decision, status, why, trigger to reopen, owner,
review date. Statuses: `proposed`, `decided`, `open`, `killed`. Open questions live only here. Rows are never
deleted; a stopped decision becomes `killed` with its reason. Agents record decisions the user makes; they do not
decide for the team.

## 7. Folder splits

Split a folder when it holds more than 12 files (not counting `README.md`), or when 3 or more files share a first
word (`interview-*`, `keyword-*`).

1. Group the files by the shared word or topic. Name sub-folders with plain nouns: `interviews/`, `keywords/`,
   never `misc/`, `other/` or `new/`.
2. A sub-folder needs at least 3 files, unless it holds one file and its source or text twin.
3. Stay within 3 levels below `docs/`; deeper means the top split is wrong.
4. Agents propose the split and wait for a yes.
5. Move the files (section 8), add the sub-folder to the parent `README.md`, and give it its own `README.md` when
   it holds more than 3 files.

A sub-folder left with 1 file for a month merges back into its parent.

## 8. Moving and renaming

1. Copy the file to its new path and name (or use `git mv`).
2. Check the copy is identical, byte for byte.
3. Delete the original.
4. Update every reference to the old path, and the folder `README.md` lines.
5. Record moves of many files in a "Where files went" table (old path → new path) in the page that explains why.

Agents never move or rename files unasked; they list the proposed moves instead.

## 9. What to commit

- Commit what readers and agents use. Git-ignore working files that help only one person (presenter sources,
  scratch notes, exports you can regenerate).
- Files with `-confidential` in the name never go to external services, public repos or published material.

## 10. The landing README

The repository `README.md` starts with a "Start here" table: the rules, the docs map, the decision register, and
the one or two pages most people need. A landing README may skip the frontmatter.

Facts:
- `docs/README.md` → lists → every top-level docs folder
- File names → use → lowercase words, hyphens and full `YYYY-MM-DD` dates
- `lifecycle:` → defaults to → per-document choice by the agent
- `source: true` → requires → a `## Sources` section
- `archive/` → holds → superseded copies, out of the search path
