<!-- defrost-ai:doc-rules:start -->
## ⚠️ IMPORTANT: documentation rules (read before writing or editing any doc)

**Before you create or edit any documentation file (`*.md`, `*.rst`, `docs/**`, READMEs, ADRs), read
`docs/DOC_RULES.md` and follow it.** Short version:

1. **One topic per section**, with a heading that names it; 60–400 words per section.
2. **Self-contained:** name the subject in the first sentence; no "it / this / see above" across sections.
3. **Exact names:** backtick every identifier and path exactly as in code; one name per concept (`docs/GLOSSARY.md`).
4. **Plain sentences:** active voice, ≤ 25 words, one fact or step each; options, defaults and limits go in tables.
5. **Facts:** end reference and explanation sections with 3–7 lines: `- Subject → relation → Object (qualifier)`.
6. **Check:** run `python docs/tools/doc_lint.py <changed files>` and fix every error before finishing.
<!-- defrost-ai:doc-rules:end -->
