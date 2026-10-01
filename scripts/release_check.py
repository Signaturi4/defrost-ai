"""Pre-release checks: run before tagging a release. No network unless --network.

    python scripts/release_check.py [--network] [--new-weights] [--skip-tests]

Checks
1. One version: pyproject `version` == defrost_ai.__version__ == install.sh default (DEFROST_VERSION).
2. Weights version rule: WEIGHTS_VERSION <= package version. Weights keep the version of the release that first
   shipped them (a code-only release keeps pointing at the older weights release); a release that ships new weights
   sets WEIGHTS_VERSION to its own version and attaches the archive whose sha256 is WEIGHTS_SHA256.
3. CHANGELOG.md has a "## <version>" section.
4. Tests pass (pytest, unless --skip-tests).
5. --new-weights: the weights being published need a provenance manifest (training/provenance/<version>.json) that
   lists every training source and states that all of them are on the allowlist (docs/RELEASING.md).
6. --network: the weights archive URL answers (HTTP HEAD) and the release tag exists on GitHub.
Exit code 1 when any check fails."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ver(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", s)[:3])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", action="store_true")
    ap.add_argument("--new-weights", action="store_true", help="this release publishes new weights")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    fails, ok = [], []

    def check(cond: bool, msg: str):
        (ok if cond else fails).append(msg)

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    from defrost_ai import __version__
    from defrost_ai.models import weights
    m = re.search(r'DEFROST_VERSION:-([0-9][0-9.]*)', (ROOT / "install.sh").read_text())
    install = m.group(1) if m else None
    check(pyproject == __version__ == install,
          f"one version: pyproject {pyproject}, defrost_ai.__version__ {__version__}, install.sh {install}")
    check(_ver(weights.WEIGHTS_VERSION) <= _ver(pyproject),
          f"weights v{weights.WEIGHTS_VERSION} <= package v{pyproject}")
    check(f"v{weights.WEIGHTS_VERSION}/" in weights.WEIGHTS_URL and len(weights.WEIGHTS_SHA256) == 64,
          "WEIGHTS_URL points at the WEIGHTS_VERSION release and WEIGHTS_SHA256 is a sha256")
    changelog = ROOT / "CHANGELOG.md"
    check(changelog.exists() and re.search(rf"^## v?{re.escape(pyproject)}\b", changelog.read_text(), re.M),
          f"CHANGELOG.md has a '## {pyproject}' section")

    if a.new_weights:
        prov = ROOT / "training/provenance" / f"{pyproject}.json"
        if not prov.exists():
            check(False, f"provenance manifest {prov.relative_to(ROOT)} exists (required to publish new weights)")
        else:
            p = json.loads(prov.read_text())
            check(bool(p.get("sources")) and p.get("all_sources_allowlisted") is True,
                  "provenance: every training source listed and allowlisted")
        check(weights.WEIGHTS_VERSION == pyproject, f"new weights: WEIGHTS_VERSION == {pyproject}")

    if a.network:
        import urllib.request
        try:
            req = urllib.request.Request(weights.WEIGHTS_URL, method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as r:
                check(r.status == 200, f"weights archive reachable: {weights.WEIGHTS_URL}")
        except Exception as e:                                     # noqa: BLE001
            check(False, f"weights archive reachable: {weights.WEIGHTS_URL} ({e})")
        tag = subprocess.run(["git", "ls-remote", "--exit-code", "--tags",
                              "https://github.com/Signaturi4/defrost-ai", f"refs/tags/v{weights.WEIGHTS_VERSION}"],
                             capture_output=True)
        check(tag.returncode == 0, f"tag v{weights.WEIGHTS_VERSION} exists on GitHub")

    if not a.skip_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        check(r.returncode == 0, f"tests pass ({last})")

    for msg in ok:
        print(f"  ok    {msg}")
    for msg in fails:
        print(f"  FAIL  {msg}")
    print("RELEASE CHECK", "OK" if not fails else "FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
