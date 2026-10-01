---
description: Document a code change (update stale doc sections, add docs for new code) following the doc rules
allowed-tools: Read, Grep, Glob, Edit, Write, Bash(defrost docs-plan:*), Bash(defrost docs-resolve:*), Bash(git show:*), Bash(git diff:*), Bash(git log:*), Bash(python docs/tools/doc_lint.py:*), mcp__defrost__memory_docs_plan, mcp__defrost__memory_search, mcp__defrost__memory_handoff
---
Bring the documentation in line with a code change. Target: $ARGUMENTS (a commit sha; empty = the uncommitted
changes; `--staged` = what is about to be committed).

1. **Plan.** Run `defrost docs-plan` with `--commit <sha>`, `--staged` or nothing, matching the target. It lists:
   - the doc sections linked to the changed code, grouped by priority;
   - the changed files that no section documents yet;
   - the docs already edited in this change.
   Then read the diff with `git show <sha>` or `git diff`.
2. **Check the code before the docs.** For each listed section, read the section and the changed code. The code is
   the truth.
   - Update the section only where it now says something false or incomplete: a default, a flag, a step, a
     behaviour, a file name.
   - Leave sections the change does not affect untouched. Do not reword them for style.
3. **Document new code** from the "no doc section yet" list, but only if a user, operator or another developer must
   know about it: a command, endpoint, env var, config key, migration, deploy step or public function. Add one
   section in the page that already covers that area (`memory_search` finds it). Create a new page only when no
   page fits.
4. **Follow `docs/DOC_RULES.md`** (short version in CLAUDE.md):
   - one topic per section, under a heading that names it, 60–400 words;
   - self-contained: name the subject in the first sentence, no "it/this/above";
   - backtick every identifier and path exactly as in code;
   - active voice, sentences of 25 words or fewer;
   - options, defaults, limits and env vars go in tables;
   - end reference and explanation sections with 3–7 `- Subject → relation → Object` Facts lines;
   - set `updated:` in the page frontmatter to today.
5. **Mark contradictions.** If a doc contradicts the code and you cannot tell which one is intended, do not guess.
   Add a line `> Open question (YYYY-MM-DD): doc says X, code does Y (`path:line`).` and list it in your reply.
6. **Check.** Run `python docs/tools/doc_lint.py <every doc you changed>` and fix every ERROR.
7. **Finish.**
   - Do not commit unless the user asked you to.
   - For a commit target, run `defrost docs-resolve <sha>`.
   - Call `memory_handoff` with goal "document <target>", the sections changed and added as state, and any open
     questions as next steps.
   - Reply with a short list: updated `path:Lx-y` (one line each: what changed), added sections, and open questions.
