---
type: how-to
entity: agent QA benchmark
owner: maintainers
status: current
updated: 2026-10-05
lifecycle: living
---

# Agent QA benchmark

The agent QA benchmark asks Claude Code the same project questions in several **arms** and scores the answers
with a blind judge. An arm is a git ref of the project plus what the agent gets: its `CLAUDE.md`, the defrost
memory (MCP server, prompt hook or both), skills, and optionally a dev build of defrost. It measures what users
feel: correct answers, flagged conflicts, hallucinations, turns, latency and cost.

Facts:
- `aqa.py` → runs → Claude Code headless per arm, question and repeat
- The judge → is → a different model from the answering model, blind to the arm
- Ground truth → comes from → code and git, never from the docs

## Run it on a project

1. Write `questions.json` next to a config, outside the arms' trees (see `questions.example.json`). Take each
   ground truth from code or git, with `file:line` evidence. Use the kinds `stale-trap`, `conflict-trap`,
   `navigation`, `decision` and `fact`.
2. Copy `config.example.json` and set `repo`, the refs of the arms and `strip` (paths that reveal the answers).
3. Run the steps below. Only `run` and `judge` spend money, and only with `--yes`.

```sh
python3 bench/agent_qa/aqa.py CONFIG setup            # worktrees under /tmp/aqa/<letter>
python3 bench/agent_qa/aqa.py CONFIG memory           # one defrost memory per defrost arm, no triggers
python3 bench/agent_qa/aqa.py CONFIG hooks            # what the prompt hook injects per question (free)
python3 bench/agent_qa/aqa.py CONFIG run --yes --repeats 1 --only Q01   # pilot: the real cost per run
python3 bench/agent_qa/aqa.py CONFIG run --yes
python3 bench/agent_qa/aqa.py CONFIG judge --yes
python3 bench/agent_qa/aqa.py CONFIG sample           # hand-score 10 answers, then: agree
python3 bench/agent_qa/aqa.py CONFIG report [--exclude Q07]
python3 bench/agent_qa/aqa.py CONFIG teardown
```

Facts:
- `aqa.py` → spends money → only in `run` and `judge` with `--yes`
- `questions.json` → lives → outside the arms' trees

## Isolation rules

The arms must differ only in what the config says.

| Rule | Why |
|---|---|
| The question alone is the prompt; the instruction goes in `--append-system-prompt` | The prompt hook searches the prompt text. A shared preamble steers every search to the same pages |
| No `git` tool | `git show other-branch:file` lets an arm read another arm's docs |
| `--strict-mcp-config` with only the arm's servers | User-level MCP servers would answer from another memory |
| `disableAllHooks` in arms without the hook | The project's own defrost hook would inject memory |
| `--disable-slash-commands` in arms without skills | Denying the `Skill` tool alone does not stop skills from loading |
| `DEFROST_DOMAIN` pins the MCP server and the hook | Search would otherwise merge every registered memory |
| `memory` installs no git hooks | Worktrees share the repository's hooks; a normal setup would overwrite the real project's refresh hook |
| Neutral worktree paths, answer key stripped, fresh session per run | The path or a leftover file would reveal the arm or the answer |

Facts:
- `DEFROST_DOMAIN` → pins → the MCP server and the prompt hook to one memory
- `--append-system-prompt` → carries → the instruction, the same in every arm

## Research loop

Improvements are searched with a loop that combines Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch) (one metric, a fixed budget per experiment, keep or
`git reset`, a results log, simplicity) and NVIDIA's [SoL-Pi](https://nvlabs.github.io/SoL-Pi/) (map-reduce
analysis of trajectories, a frozen held-out set, two gates).

1. **Rollouts:** run the arms and keep every trajectory, plus the hook output (`hooks`).
2. **Map:** analyse each failed answer alone and give it one root cause: not documented, not retrieved, ranked too
   low, retrieved but not used, wrong ground truth, judge too strict, or model reasoning.
3. **Reduce:** group the root causes into hypotheses, one lever each (docs, retrieval, hook text, instruction).
4. **Experiment:** change one lever in its own git worktree or branch. Measure it with the cheapest isolated test:
   - retrieval: `defrost benchmark` gold-span recall, free;
   - hook: `aqa.py hooks`, free;
   - agent: `run --only <failing ids> --repeats 2`, cents.
5. **Gate:** keep the change only if its metric improves and correctness does not drop; otherwise `git reset`. A
   small gain that adds complexity is not kept; removing code at equal results is.
6. **Log** every experiment, kept or not, in `results.tsv`: commit, metric, gate, status, description.
7. **Held-out:** run the frozen final configuration end to end on questions the loop never saw, ideally from
   another project. Held-out results never feed back into the loop.

Facts:
- The research loop → combines → autoresearch (keep or reset) and SoL-Pi (map-reduce, held-out gate)
- A change → is kept → only if its metric improves and correctness does not drop

## Results

Measured results live with each study; see [`docs/AGENT_QA.md`](../../docs/AGENT_QA.md).
