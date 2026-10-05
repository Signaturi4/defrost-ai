"""Tests for repo_lint.py and the page-level checks added to doc_lint.py (stdlib unittest).

    python -m unittest discover -s scripts/tests -v       # from the skill folder
"""
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'extraction-ready-docs' / 'scripts'))
import doc_lint  # noqa: E402
import repo_lint  # noqa: E402


def page(entity, body="", **fm):
    fields = {"type": "reference", "entity": entity, "status": "current", "updated": "2026-10-05", **fm}
    head = "\n".join(f"{k}: {v}" for k, v in fields.items())
    return f"---\n{head}\n---\n# {entity}\n\n{body}\n"


class Tree:
    """A throwaway repo: Tree({"docs/a.md": "...", ...}).root"""

    def __init__(self, files):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for rel, text in files.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(text, bytes):
                p.write_bytes(text)
            else:
                p.write_text(text, encoding="utf-8")

    def lint(self, **kw):
        return repo_lint.lint_repo(self.root, **kw)

    def close(self):
        self.tmp.cleanup()


def codes(issues):
    return sorted({i.code for i in issues})


class RepoLintTests(unittest.TestCase):
    def tearDown(self):
        for t in getattr(self, "_trees", []):
            t.close()

    def tree(self, files):
        t = Tree(files)
        self._trees = getattr(self, "_trees", []) + [t]
        return t

    def test_clean_tree_has_no_issues(self):
        t = self.tree({
            "docs/README.md": page("docs index", "| `research/` | interviews |\n\n`competitors.md`, `research/README.md`"),
            "docs/competitors.md": page("competitors", lifecycle="living"),
            "docs/research/README.md": page("research", "`interview-maya-2026-10-12.md`"),
            "docs/research/interview-maya-2026-10-12.md": page("interview with maya", lifecycle="immutable",
                                                               source="true")
            + "\n## Sources\n\n- Interview with Maya, recorded 2026-10-12\n",
        })
        self.assertEqual([], [i for i in t.lint(mode="per-document")], [str(i) for i in t.lint(mode="per-document")])

    def test_version_and_bad_names(self):
        t = self.tree({
            "docs/README.md": page("idx", "`competitors_v2.md` `Notes Monday.md` `plan-final.md` `old.md.bak`"),
            "docs/competitors_v2.md": page("a"),
            "docs/Notes Monday.md": page("b"),
            "docs/plan-final.md": page("c"),
            "docs/old.md.bak": "x",
            "docs/tools/doc_lint.py": "print()",          # code keeps its own conventions
        })
        bad = {i.path.name for i in t.lint(mode="living") if i.code == "name"}
        self.assertEqual({"competitors_v2.md", "Notes Monday.md", "plan-final.md", "old.md.bak"}, bad)

    def test_dates_in_names_use_the_full_format(self):
        t = self.tree({"docs/README.md": page("idx", "`survey-freelancers-2026-10.md` `interview-maya-2026-10-12.md` "
                                              "`notes-12-10-2026.md`"),
                       "docs/survey-freelancers-2026-10.md": page("s"), "docs/interview-maya-2026-10-12.md": page("i"),
                       "docs/notes-12-10-2026.md": page("n")})
        bad = sorted(i.path.name for i in t.lint(mode="living") if i.code == "name")
        self.assertEqual(["notes-12-10-2026.md", "survey-freelancers-2026-10.md"], bad)

    def test_coded_names_need_the_legend(self):
        files = {"docs/README.md": page("idx", "`P1_MR_INT_01-maya-2026-10-12.md` `P1_XX_INT_02-leo-2026-10-02.md` "
                                        "`P1_MR_INT_03.md`"),
                 "docs/P1_MR_INT_01-maya-2026-10-12.md": page("a"),
                 "docs/P1_XX_INT_02-leo-2026-10-02.md": page("b"),
                 "docs/P1_MR_INT_03.md": page("c")}
        no_legend = {i.path.name for i in self.tree(files).lint(mode="living") if i.code == "name"}
        self.assertEqual(3, len(no_legend))                       # without CODES.md, codes are just bad names
        files["docs/CODES.md"] = "| code | meaning |\n|---|---|\n| P1 | Phase 1 |\n| MR | Marketing research |\n" \
                                 "| INT | Interview |\n"
        bad = sorted(i.path.name for i in self.tree(files).lint(mode="living") if i.code == "name")
        self.assertEqual(["P1_MR_INT_03.md", "P1_XX_INT_02-leo-2026-10-02.md"], bad)

    def test_version_copies_without_frontmatter_are_duplicates(self):
        t = self.tree({"docs/README.md": page("idx", "`competitors_v2.md` `competitors_v3_FINAL.md` `plan.md`"),
                       "docs/competitors_v2.md": "# Competitors\n\nToggl.\n",
                       "docs/competitors_v3_FINAL.md": "# Competitors\n\nToggl, Harvest.\n",
                       "docs/plan.md": "# Plan\n"})
        dup = [i for i in t.lint(mode="living") if i.code == "duplicate-copies"]
        self.assertTrue(dup and dup[0].level == "ERROR", [str(i) for i in dup])
        self.assertIn("competitors", dup[0].msg)

    def test_two_current_files_for_one_entity_is_an_error(self):
        t = self.tree({
            "docs/README.md": page("idx", "`a.md` `b.md`"),
            "docs/a.md": page("Competitors"),
            "docs/b.md": page("competitors"),
        })
        err = [i for i in t.lint(mode="living") if i.code == "duplicate-entity"]
        self.assertTrue(err and err[0].level == "ERROR")

    def test_superseded_copy_in_archive_is_fine_outside_is_error(self):
        t = self.tree({
            "docs/README.md": page("idx", "`sow-client.md`"),
            "docs/sow-client.md": page("sow", lifecycle="versioned"),
            "archive/docs/sow-client--superseded-2026-10-01.md": page("sow", status="superseded",
                                                                     superseded_by="docs/sow-client.md"),
            "docs/sow-client--superseded-2026-09-01.md": page("sow", status="superseded",
                                                             superseded_by="docs/sow-client.md"),
        })
        issues = t.lint(mode="versioned")
        out = [i for i in issues if i.code == "superseded-outside-archive"]
        self.assertEqual(1, len(out))
        self.assertEqual("ERROR", out[0].level)
        self.assertNotIn("duplicate-entity", codes(issues))      # superseded copies are not "current"

    def test_identical_content(self):
        t = self.tree({"docs/README.md": page("idx", "`x.csv` `y.csv`"), "docs/x.csv": "a,b\n1,2\n",
                       "docs/y.csv": "a,b\n1,2\n"})
        self.assertIn("duplicate-content", codes(t.lint(mode="living")))

    def test_crowded_folder_and_shared_prefix(self):
        files = {"docs/README.md": page("idx", " ".join(f"`note-{i:02d}.md`" for i in range(13)))}
        files.update({f"docs/note-{i:02d}.md": page(f"n{i}") for i in range(13)})
        self.assertIn("crowded", codes(self.tree(files).lint(mode="living")))
        files = {"docs/README.md": page("idx", "`interview-a.md` `interview-b.md` `interview-c.md` `plan.md`"),
                 "docs/interview-a.md": page("a"), "docs/interview-b.md": page("b"),
                 "docs/interview-c.md": page("c"), "docs/plan.md": page("p")}
        self.assertIn("shared-prefix", codes(self.tree(files).lint(mode="living")))
        files.pop("docs/plan.md")                               # the group is the whole folder: already split
        files["docs/README.md"] = page("idx", "`interview-a.md` `interview-b.md` `interview-c.md`")
        self.assertNotIn("shared-prefix", codes(self.tree(files).lint(mode="living")))

    def test_file_missing_from_folder_index(self):
        t = self.tree({"docs/README.md": page("idx", "`a.md`"), "docs/a.md": page("a"), "docs/b.md": page("b")})
        missing = [i.path.name for i in t.lint(mode="living") if i.code == "not-indexed"]
        self.assertEqual(["b.md"], missing)

    def test_binary_needs_text_twin(self):
        t = self.tree({"docs/README.md": page("idx", "`deck.pdf` `deck.md` `shot.png`"),
                       "docs/deck.pdf": b"%PDF", "docs/deck.md": page("deck"), "docs/shot.png": b"\x89PNG"})
        self.assertEqual(["shot.png"], [i.path.name for i in t.lint(mode="living") if i.code == "no-text-twin"])

    def test_broken_backticked_path(self):
        t = self.tree({"docs/README.md": page("idx", "`a.md` see `docs/missing.md` and `docs/new.md` (planned)"
                                              " and `docs/interview-<name>.md`"),
                       "docs/a.md": page("a")})
        broken = [i for i in t.lint(mode="living") if i.code == "broken-ref"]
        self.assertEqual(1, len(broken))
        self.assertIn("docs/missing.md", broken[0].msg)

    def test_per_document_mode_requires_lifecycle(self):
        t = self.tree({"docs/README.md": page("idx", "`a.md`"), "docs/a.md": page("a")})
        self.assertIn("lifecycle-missing", codes(t.lint(mode="per-document")))
        self.assertNotIn("lifecycle-missing", codes(t.lint(mode="living")))

    def test_mode_read_from_claude_md_block(self):
        t = self.tree({"CLAUDE.md": "<!-- defrost-ai:doc-rules:start -->\nLifecycle mode: **versioned**\n"
                                    "<!-- defrost-ai:doc-rules:end -->\n",
                       "docs/README.md": page("idx")})
        self.assertEqual("versioned", repo_lint.read_mode(t.root))
        self.assertEqual("per-document", repo_lint.read_mode(self.tree({}).root))

    def test_immutable_file_edited_after_first_commit(self):
        t = self.tree({"docs/README.md": page("idx", "`t.md`"),
                       "docs/t.md": page("t", lifecycle="immutable")})
        git = ["git", "-C", str(t.root), "-c", "user.email=t@t", "-c", "user.name=t"]
        subprocess.run(git[:3] + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "-A"], check=True)
        subprocess.run(git + ["commit", "-qm", "one"], check=True)
        self.assertNotIn("immutable-edited", codes(t.lint(mode="per-document")))
        (t.root / "docs/t.md").write_text(page("t", "changed", lifecycle="immutable"), encoding="utf-8")
        subprocess.run(git + ["commit", "-qam", "two"], check=True)
        self.assertIn("immutable-edited", codes(t.lint(mode="per-document")))


class DocLintPageTests(unittest.TestCase):
    def lint_text(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.md"
            p.write_text(text, encoding="utf-8")
            return doc_lint.lint(p)

    def test_source_page_needs_sources_section(self):
        body = textwrap.dedent("""\
            ## Findings

            Maya invoices weekly (Maya, L12).
            """)
        errs = [m for lv, _, m in self.lint_text(page("maya", body, source="true")) if lv == "ERROR"]
        self.assertTrue(any("Sources" in m for m in errs), errs)
        ok = page("maya", body + "\n## Sources\n\n- Interview with Maya, recorded 2026-10-12\n", source="true")
        self.assertFalse([m for lv, _, m in self.lint_text(ok) if lv == "ERROR"])

    def test_invalid_lifecycle_value(self):
        errs = [m for lv, _, m in self.lint_text(page("a", lifecycle="forever")) if lv == "ERROR"]
        self.assertTrue(any("lifecycle" in m for m in errs), errs)


if __name__ == "__main__":
    unittest.main()
