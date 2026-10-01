# Releasing

## Versions

- **One version for the code:** `pyproject.toml`, `defrost_ai.__version__` and the `DEFROST_VERSION` default in
  `install.sh` always match. `scripts/release_check.py` fails if they don't.
- **Weights keep the version of the release that first shipped them.** A code-only release keeps pointing at the
  older weights release (`WEIGHTS_VERSION` may lag the code version, never lead it). A release that ships new
  weights sets `WEIGHTS_VERSION` to its own version, attaches the archive, and pins its sha256 in `WEIGHTS_SHA256`.
- **`install.sh` installs a release tag by default.** `DEFROST_REF=main` installs the moving branch,
  `DEFROST_REPO=<path>` a local checkout.

## Checklist

1. **Data provenance, for any release that publishes new weights:**
   - Every training source is on the explicit allowlist. Never publish weights trained on a source that is not on
     it: no client, employer or personal documents, and no folders collected by walking a home directory.
   - Write `training/provenance/<version>.json` with every source (name, URL or path, license, commit) and
     `"all_sources_allowlisted": true`, and check it: `python scripts/release_check.py --new-weights`.
   - The leakage gate runs against every training stage's corpus (MNTP, CGSA, supervised), not only the
     supervised data. Document the result in `docs/EVALUATION.md`.
2. **Quality:** the quality gate in `docs/OPTIMIZATION_PLAN.md` §5 passes against the previous release.
3. **Versions and changelog:** bump the version in `pyproject.toml`, `defrost_ai/__init__.py` and `install.sh`. Add a
   `## <version>` section to `CHANGELOG.md`.
4. **Local check:** `python scripts/release_check.py` (versions, changelog, tests).
5. **Tag and release:** push the commit and the tag `v<version>`. For new weights, attach
   `defrost-ai-weights-v<version>.tar.gz` to that release.
6. **Online check:** `python scripts/release_check.py --network` (the weights archive and tag exist on GitHub).
7. **Smoke test:** in a clean environment, run
   `UV_TOOL_DIR=$(mktemp -d) UV_TOOL_BIN_DIR=$(mktemp -d) sh install.sh`, then `defrost --version`.
