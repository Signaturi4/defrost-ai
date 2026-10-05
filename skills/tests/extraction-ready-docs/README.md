# Tests for the extraction-ready-docs skill

The skill package (`skills/extraction-ready-docs/`) ships without tests. This folder holds the two test layers
used to keep its behaviour consistent between versions.

| layer | what it checks | how to run |
|---|---|---|
| `unit/` | the linters (`repo_lint.py`, `doc_lint.py`), the post-install audit (`audit.py`) and the installer, on throwaway fixture trees | `python -m unittest discover -s skills/tests/extraction-ready-docs/unit` |
| `evals/` + `fixtures/` + `grade.py` | what an agent **does** with the skill: four realistic tasks, each run with and without the skill | see below |

## Behaviour evals

Each task in `evals/evals.json` names a fixture repository. A run copies the fixture, gives an agent the prompt,
and saves the repository plus `final_message.md` and `questions.md`. The agent writes questions to `questions.md`
instead of asking, so "asks before moving files" is testable.

1. Lay out runs as `<iteration>/eval-<id>-<name>/{with_skill,without_skill}/run-<k>/outputs/`.
2. Run every task at least twice with the skill: two runs that score differently show inconsistent behaviour.
3. Grade with `python grade.py <iteration>`. Every assertion is checked by code, so the same output always gets
   the same grade; differences between runs come from the agent, not the grader.
4. Aggregate and review with the skill-creator tools (`scripts.aggregate_benchmark`,
   `eval-viewer/generate_review.py`).

| task | the skill must make the agent… |
|---|---|
| `setup-living-mode` | install in general mode with lifecycle `living`, keep the existing `CLAUDE.md`, add no defrost tooling |
| `add-interview-source-file` | create one `interview-maya-2026-10-12.md` with `source: true`, `lifecycle: immutable`, a Sources section, an index line, and no invented quotes |
| `messy-folder-propose-first` | move nothing before a yes, propose rule-following names with full dates, merge the duplicate copies, add a text twin and a decision register |
| `install-and-update-existing-docs` | install, then fix every existing `.md` in place without losing a fact (frontmatter, backticks, lint clean), list the docs in the map, record the old meeting decision in the register, and only propose the rename and move of `notes/Meeting Notes.md` |

Results (2026-10-05), 2 runs per task with the skill, 1 without:

| iteration | tasks | with the skill | without |
|---|---|---|---|
| 1 | 1–3 | 100% (identical scores across runs) | 49% |
| 2 | 1–3 (simplified installer) | 100% (identical) | 49% |
| 3 | 4 (post-install scan) | 100% (identical) | 22% |
