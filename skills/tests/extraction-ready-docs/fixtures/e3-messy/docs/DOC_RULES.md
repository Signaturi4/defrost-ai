# Documentation rules

These rules keep docs easy for people to read and easy for retrieval and knowledge-graph tools (BM25, embeddings,
rerankers, triple extractors) to use, without extra LLM passes. They apply to every technical doc in this repo.
Why each rule exists: `references/WHY.md` in the `extraction-ready-docs` skill.

## 1. Pages

- **One page type per page.** Declare it in the frontmatter: `tutorial` (learn by doing), `how-to` (do one task),
  `reference` (facts to look up), `explanation` (why and how it works). Do not mix types on one page.
- **Frontmatter** on every page:

  ```yaml
  ---
  type: reference          # tutorial | how-to | reference | explanation
  entity: PaymentService   # the main thing this page is about (canonical name)
  owner: payments-team
  status: current          # current | deprecated | superseded | draft
  updated: 2026-10-01
  lifecycle: living        # living | versioned | immutable — how the file changes (KNOWLEDGE_RULES.md §3)
  source: false            # true = facts first enter the project here; needs a Sources section (§4)
  ---
  ```

- **One owner per fact.** Write a fact in one place and link to it from everywhere else. Copies drift apart.
- **Where files live, names, versions and sources** are in `KNOWLEDGE_RULES.md`.

## 2. Sections

- **One topic per section.** The heading names the topic in words a reader would search for:
  "Retry policy for failed webhooks", not "Notes" or "Details".
- **Length:** 60–400 words. Split longer sections. Merge shorter ones into their parent unless they hold a table.
- **Self-contained:** a section must make sense when read alone (search tools return single sections).
  - Name the subject in the first sentence.
  - Do not start a section with "It", "This", "These", "They", "Here" or "As mentioned above".
  - Repeat the name instead of a pronoun when the referent is more than one sentence back.

## 3. Names

- **Backtick every code identifier and path exactly as written in code:** `charge_invoice()`, `PaymentService`,
  `src/billing/retry.py`, `MAX_RETRIES`, `--dry-run`. Tools link these to the code graph; misspelled or
  un-backticked names are lost.
- **One canonical name per concept.** List it in `docs/GLOSSARY.md` with its aliases (create the file from
  `docs/templates/GLOSSARY.template.md` when you add the first term). Use the canonical name in
  headings and Facts, and mention a common alias once if users search for it: "refund (also called reversal)".
- **No undefined abbreviations.** Spell an abbreviation out on first use in each page.

## 4. Sentences

- Use active voice and name the actor: "`Scheduler` retries the job", not "the job is retried".
- At most 25 words per sentence, and one fact or one instruction per sentence.
- Keep all parts of the sentence: no dropped articles or verbs ("Set timeout" → "Set the timeout").
- Noun stacks of at most 3 words ("payment retry queue" is fine; "payment retry queue worker config file" is not).
- Procedures are numbered steps that start with a verb, one action per step.
- Prefer one concrete example to a paragraph of description.

## 5. Tables

Put options, defaults, limits, environment variables, error codes and API fields in tables:

| option | default | meaning |
|---|---|---|
| `MAX_RETRIES` | `5` | retries before an event moves to the dead-letter table |

## 6. Facts block

End every `reference` and `explanation` section with a Facts list. Optional in how-tos and tutorials.

```markdown
Facts:
- `WebhookDispatcher` → retries → failed deliveries (max 5, exponential backoff)
- `WebhookDispatcher` → writes to → `webhook_dead_letter`
- HTTP 410 response → stops → webhook retries
```

- 3–7 lines per section, key facts only. The prose stays the source of truth.
- Form: `- Subject → relation → Object (optional qualifier)`; `->` is accepted for `→`.
- Subjects and objects use canonical names; code entities are backticked.
- Relations come from the list in `docs/GLOSSARY.md`. Add a new relation there before using it.

## 7. Before you finish

Run `python docs/tools/doc_lint.py <files>`. It checks frontmatter, section length, opening pronouns, sentence
length and Facts syntax. Fix every error; warnings are judgment calls.
