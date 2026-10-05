"""Tests for audit.py: every Markdown file is scanned; in-file fixes and path changes land in separate buckets."""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "extraction-ready-docs" / "scripts"))
import audit  # noqa: E402


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rel, text):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def test_finds_markdown_everywhere_and_splits_the_work(self):
        self.write("README.md", "# Project\n\nLanding page.\n")                      # landing: no frontmatter needed
        self.write("CLAUDE.md", "# rules\n")                                          # agent config: skipped
        self.write("notes/Meeting Notes.md", "# Meeting\n\nWe met.\n")                # outside docs + bad name
        self.write("docs/README.md", "---\ntype: reference\nentity: map\nstatus: current\nupdated: 2026-10-05\n"
                                     "lifecycle: living\n---\n# docs\n\n`setup.md`\n")
        self.write("docs/setup.md", "# Setup\n\nRun it.\n")                           # no frontmatter
        self.write("node_modules/pkg/README.md", "# vendored\n")                      # skipped
        mode, items = audit.audit(self.root)
        self.assertEqual("per-document", mode)
        self.assertNotIn("README.md", items)
        self.assertNotIn("CLAUDE.md", items)
        self.assertFalse(any(k.startswith("node_modules") for k in items))
        meeting = items["notes/Meeting Notes.md"]
        self.assertTrue(any(x.startswith("rename") for x in meeting["needs_yes"]))
        self.assertTrue(any(x.startswith("outside-docs") for x in meeting["needs_yes"]))
        setup = items["docs/setup.md"]
        self.assertTrue(any("frontmatter" in x for x in setup["fix_in_place"]))
        self.assertFalse(setup["needs_yes"])

    def test_duplicate_copies_need_a_yes(self):
        self.write("docs/README.md", "---\ntype: reference\nentity: map\nstatus: current\nupdated: 2026-10-05\n---\n"
                                     "# docs\n\n`competitors_v2.md` `competitors_v3_FINAL.md`\n")
        self.write("docs/competitors_v2.md", "# Competitors\n\nToggl.\n")
        self.write("docs/competitors_v3_FINAL.md", "# Competitors\n\nToggl, Harvest.\n")
        _, items = audit.audit(self.root)
        asks = [x for v in items.values() for x in v["needs_yes"]]
        self.assertTrue(any("duplicate-copies" in x for x in asks), asks)

    def test_clean_install_has_nothing_to_do(self):
        import subprocess
        subprocess.run([sys.executable, str(Path(audit.__file__).parent / "install.py"), str(self.root)],
                       check=True, capture_output=True)
        _, items = audit.audit(self.root)
        self.assertEqual({}, {k: v for k, v in items.items() if any(x.startswith("ERROR") for x in v["fix_in_place"])
                              or v["needs_yes"]})


if __name__ == "__main__":
    unittest.main()
