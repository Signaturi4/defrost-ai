"""Tests for install.py: one command for any repository; --defrost and --remove are the only variants."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[2] / "extraction-ready-docs" / "scripts"
INSTALL = SCRIPTS / "install.py"


def run(root, *args):
    r = subprocess.run([sys.executable, str(INSTALL), str(root), *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "my project"          # a space in the path
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_creates_rules_map_and_register(self):
        run(self.root)
        for rel in ("docs/DOC_RULES.md", "docs/KNOWLEDGE_RULES.md", "docs/README.md", "docs/decision-register.md",
                    "docs/templates/PAGE.template.md", "CLAUDE.md"):
            self.assertTrue((self.root / rel).exists(), rel)
        self.assertFalse((self.root / "docs/tools").exists())   # linters are defrost-only
        self.assertFalse((self.root / "AGENTS.md").exists())    # never created
        cm = (self.root / "CLAUDE.md").read_text()
        self.assertIn("<!-- extraction-ready-docs:start -->", cm)
        self.assertIn("Lifecycle mode: per-document", cm)
        self.assertNotIn("doc_lint.py", cm)

    def test_existing_files_are_kept_and_claude_md_preserved(self):
        (self.root / "docs").mkdir()
        (self.root / "docs/decision-register.md").write_text("# ours\n")
        (self.root / "docs/README.md").write_text("# our map\n")
        (self.root / "CLAUDE.md").write_text("# Project\n\nUse pnpm.\n")
        run(self.root)
        self.assertEqual("# ours\n", (self.root / "docs/decision-register.md").read_text())
        self.assertEqual("# our map\n", (self.root / "docs/README.md").read_text())
        self.assertTrue((self.root / "CLAUDE.md").read_text().startswith("# Project\n\nUse pnpm.\n\n"))

    def test_agents_md_updated_only_when_present(self):
        (self.root / "AGENTS.md").write_text("# Agents\n")
        run(self.root)
        self.assertIn("extraction-ready-docs:start", (self.root / "AGENTS.md").read_text())

    def test_rerun_replaces_block_and_lifecycle_option(self):
        run(self.root)
        run(self.root, "--lifecycle", "living")
        cm = (self.root / "CLAUDE.md").read_text()
        self.assertEqual(1, cm.count(":start -->"))
        self.assertIn("Lifecycle mode: living", cm)

    def test_defrost_mode_adds_linters_rule_and_markers(self):
        run(self.root)
        run(self.root, "--defrost")
        cm = (self.root / "CLAUDE.md").read_text()
        self.assertEqual(1, cm.count(":start -->"))               # general block replaced, not duplicated
        self.assertIn("<!-- defrost-ai:doc-rules:start -->", cm)
        self.assertIn("doc_lint.py", cm)
        for tool in ("doc_lint.py", "repo_lint.py", "extract_facts.py"):
            self.assertTrue((self.root / "docs/tools" / tool).exists(), tool)

    def test_remove_restores_claude_md(self):
        (self.root / "CLAUDE.md").write_text("# Project\n\nUse pnpm.\n")
        run(self.root)
        run(self.root, "--remove")
        self.assertEqual("# Project\n\nUse pnpm.\n", (self.root / "CLAUDE.md").read_text())

    def test_fresh_install_lints_clean(self):
        run(self.root, "--defrost")
        r = subprocess.run([sys.executable, str(SCRIPTS / "repo_lint.py"), str(self.root)], capture_output=True,
                           text=True)
        self.assertEqual(0, r.returncode, r.stdout)
        self.assertIn("0 error(s), 0 warning(s)", r.stdout)

    def test_removed_flags_are_rejected(self):
        for flag in ("--scaffold", "--agents-md"):
            r = subprocess.run([sys.executable, str(INSTALL), str(self.root), flag], capture_output=True, text=True)
            self.assertNotEqual(0, r.returncode, flag)


if __name__ == "__main__":
    unittest.main()
