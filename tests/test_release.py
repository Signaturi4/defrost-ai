"""Release hygiene, no network: one version everywhere, weights version rule, CLI --version without ML imports."""
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ver(s):
    return tuple(int(x) for x in re.findall(r"\d+", s)[:3])


def test_one_version_in_pyproject_package_and_installer():
    from defrost_ai import __version__
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    install = re.search(r"DEFROST_VERSION:-([0-9][0-9.]*)", (ROOT / "install.sh").read_text()).group(1)
    assert pyproject == __version__ == install


def test_weights_version_never_leads_the_code_version():
    from defrost_ai import __version__
    from defrost_ai.models import weights
    assert _ver(weights.WEIGHTS_VERSION) <= _ver(__version__)
    assert f"v{weights.WEIGHTS_VERSION}/" in weights.WEIGHTS_URL


def test_cli_version_flag_needs_no_ml_stack():
    code = ("import sys, builtins\n"
            "real = builtins.__import__\n"
            "def guard(name, *a, **k):\n"
            "    if name.split('.')[0] in ('torch', 'transformers', 'mlx'):\n"
            "        raise ImportError('heavy import: ' + name)\n"
            "    return real(name, *a, **k)\n"
            "builtins.__import__ = guard\n"
            "from defrost_ai.cli import main\n"
            "main(['--version'])\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("defrost "), r.stderr


def test_install_script_parses():
    assert subprocess.run(["sh", "-n", str(ROOT / "install.sh")]).returncode == 0
