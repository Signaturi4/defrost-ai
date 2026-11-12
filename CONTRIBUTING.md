# Contributing to defrost-ai

Thanks for helping. Bug reports, doc fixes and small focused pull requests are all welcome.

## Report a bug or ask for a feature

Open an [issue](https://github.com/Signaturi4/defrost-ai/issues) with:
- what you ran (the command or MCP call) and what you expected;
- what happened instead (the error or a short excerpt of the output);
- `defrost --version` and your OS. Windows is not supported yet.

## Set up a development checkout

```sh
git clone https://github.com/Signaturi4/defrost-ai.git
cd defrost-ai
uv venv --python 3.12 && uv pip install -e ".[code,dev]"   # or: python -m pip install -e ".[code,dev]"
defrost download-weights                                    # only needed for search and build, not for the unit tests
```

## Run the checks

The same checks CI runs (no model weights, no GPU, no torch needed):

```sh
python -m pytest -q tests
python -m defrost_ai.cli --help > /dev/null
bash -n install.sh
```

Tests that need torch or the weights are skipped when they are missing.

## Pull requests

- One topic per pull request; keep it small. Branch from `main`.
- Add or update a test for a behaviour change, and a line under `## Unreleased` in `CHANGELOG.md`.
- Update the doc section that describes what you changed (`defrost docs` lists the sections a change touches).
- Retrieval or model changes: include the benchmark numbers before and after (see `docs/EVALUATION.md`); the
  quality gate is in `docs/OPTIMIZATION_PLAN.md`.
- Releases follow `docs/RELEASING.md`; maintainers cut them.

By contributing you agree that your contribution is licensed under the MIT licence of this project.
