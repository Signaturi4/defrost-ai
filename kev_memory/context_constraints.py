"""Validation rules for a context repository (frontmatter, indexes, size and depth limits).

Ported from Letta Code (Apache-2.0, https://github.com/letta-ai/letta-code, commit 3687ea51):
  src/memory-frontmatter.ts       validateMemoryFileFrontmatter  -> validate_frontmatter
  src/memory-constraints.ts       validateMemoryTreeConstraints  -> validate_tree
                                  parseMemoryConstraintsConfig   -> parse_config
                                  DEFAULT_MEMORY_CONSTRAINTS_CONFIG
  src/agent/memory-constraints.ts the self-contained validator run by the pre-commit hook -> hook_main
Changes: Python; one layout (Letta's "memfs-v2" root-marker layout) instead of three; allowed frontmatter keys are
`name`, `description` and the protected `read_only` (Letta v2 allows only name/description, its legacy layout
allows read_only); config edits are approved with DEFROST_CONTEXT_CONFIG_UPDATE=1 instead of
LETTA_MEMORY_CONSTRAINTS_UPDATE. See NOTICE.

This module must stay self-contained (standard library only, no kev_memory imports): `context_repo` embeds its
source text into the repository's pre-commit hook, so the hook runs without the package being importable."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

CONFIG_PATH = ".memfs.config.json"                     # same file name as Letta, so Letta memory repos validate too
CONFIG_UPDATE_ENV = "DEFROST_CONTEXT_CONFIG_UPDATE"
DEFAULT_CONFIG = {"version": 1, "maxDepth": 2, "maxFileCharacters": 20_000, "maxCoreMemoryCharacters": 65_536}
ALLOWED_KEYS = ("name", "description", "read_only")


# ---- frontmatter -------------------------------------------------------------------------------------------------
def frontmatter_value(text: str | None, key: str) -> str:
    """Scalar value of `key:` in the frontmatter of `text` ("" when absent). No YAML interpretation, like Letta."""
    lines = (text or "").split("\n")
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        return ""
    end = lines.index("---", 1)
    return "\n".join(line[len(key) + 1:].strip() for line in lines[1:end] if line.startswith(f"{key}:"))


def validate_frontmatter(path: str, content: str, previous: str | None) -> list[str]:
    """Errors for one Markdown file. `previous` is the accepted (HEAD) content, never content from the change."""
    lines = content.split("\n")
    if path.split("/")[-1] == "MEMORY.md":
        return [f"{path}: MEMORY.md must not have frontmatter"] if lines[0] == "---" else []
    if lines[0] != "---":
        return [f"{path}: missing frontmatter (must start with ---)"]
    if "---" not in lines[1:]:
        return [f"{path}: frontmatter opened but never closed (missing closing ---)"]
    if previous and frontmatter_value(previous, "read_only") == "true":
        return [f"{path}: file is read_only and cannot be modified"]
    errors, seen = [], set()
    for line in lines[1:lines.index("---", 1)]:
        if not line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key not in ALLOWED_KEYS:
            errors.append(f"{path}: unknown frontmatter key '{key}' (allowed: {' '.join(ALLOWED_KEYS)})")
            continue
        if key in seen:
            errors.append(f"{path}: duplicate frontmatter key '{key}'")
        seen.add(key)
        if key == "read_only":
            if not previous:
                errors.append(f"{path}: 'read_only' is a protected field and cannot be set by the agent")
            elif value != frontmatter_value(previous, key):
                errors.append(f"{path}: 'read_only' is a protected field and cannot be changed by the agent")
        elif not value or value in ('""', "''"):
            errors.append(f"{path}: '{key}' must not be empty")
    for key in ("name", "description"):
        if key not in seen:
            errors.append(f"{path}: missing required field '{key}'")
    if frontmatter_value(previous, "read_only") and not frontmatter_value(content, "read_only"):
        errors.append(f"{path}: 'read_only' is a protected field and cannot be removed by the agent")
    return errors


# ---- config ------------------------------------------------------------------------------------------------------
def parse_config(text: str) -> dict:
    """Parse the tracked limits file without filling defaults; raises ValueError listing every problem."""
    try:
        config = json.loads(text)
    except ValueError as e:
        raise ValueError(f"{CONFIG_PATH}: invalid JSON ({e})") from None
    if not isinstance(config, dict):
        raise ValueError(f"{CONFIG_PATH}: expected a JSON object")
    errors = [f"{CONFIG_PATH}: unknown field '{k}'" for k in config
              if k not in ("version", "maxDepth", "maxFileCharacters", "maxCoreMemoryCharacters", "fileCharacterLimits")]
    if config.get("version") != 1:
        errors.append(f"{CONFIG_PATH}: version must be 1")
    if "maxDepth" in config and not (isinstance(config["maxDepth"], int) and config["maxDepth"] >= 0):
        errors.append(f"{CONFIG_PATH}: maxDepth must be a non-negative integer")
    for key in ("maxFileCharacters", "maxCoreMemoryCharacters"):
        if key in config and not (isinstance(config[key], int) and not isinstance(config[key], bool) and config[key] > 0):
            errors.append(f"{CONFIG_PATH}: {key} must be a positive integer")
    overrides = config.get("fileCharacterLimits", [])
    if not isinstance(overrides, list):
        errors.append(f"{CONFIG_PATH}: fileCharacterLimits must be an array")
        overrides = []
    for i, o in enumerate(overrides):
        label = f"{CONFIG_PATH}: fileCharacterLimits[{i}]"
        if not isinstance(o, dict):
            errors.append(f"{label} must be an object")
            continue
        errors += [f"{label}: unknown field '{k}'" for k in o if k not in ("pattern", "maxCharacters")]
        p = o.get("pattern")
        if not isinstance(p, str) or not p or p.startswith("/") or "\\" in p or ".." in p.split("/"):
            errors.append(f"{label}: pattern must be a non-empty repo-relative glob using '/'")
        elif any("**" in seg and seg != "**" for seg in p.split("/")):
            errors.append(f"{label}: '**' must be a complete path segment")
        m = o.get("maxCharacters")
        if m is not None and not (isinstance(m, int) and m > 0):
            errors.append(f"{label}: maxCharacters must be a positive integer or null")
    if errors:
        raise ValueError("\n".join(errors))
    return config


def glob_regex(pattern: str) -> re.Pattern:
    """Letta's glob: `*` within a segment, `**/` any number of directories, `**` anything, `?` one character."""
    out, i = "^", 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if pattern[i + 1:i + 2] == "*":
                if pattern[i + 2:i + 3] == "/":
                    out += "(?:.*/)?"; i += 2
                else:
                    out += ".*"; i += 1
            else:
                out += "[^/]*"
        elif c == "?":
            out += "[^/]"
        else:
            out += re.escape(c)
        i += 1
    return re.compile(out + "$")


# ---- tree --------------------------------------------------------------------------------------------------------
def validate_tree(files: list[tuple[str, str]], read, config: dict | None = None) -> list[str]:
    """files: [(repo-relative path, git mode)]; read(path) -> text. Root MEMORY.md map, one MEMORY.md index per
    folder, depth and size limits, and the budget for root ("core") files."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    md = [(p, m) for p, m in files if p.endswith(".md")]
    paths = {p for p, _ in md}
    errors = [] if "MEMORY.md" in paths else ["MEMORY.md: root memory index is required"]
    for p, _ in md:
        current = ""
        for d in p.split("/")[:-1]:
            current = f"{current}/{d}" if current else d
            if f"{current}/MEMORY.md" not in paths:
                errors.append(f"{p}: missing required index {current}/MEMORY.md")
                break
    overrides = [(o["pattern"], o["maxCharacters"], glob_regex(o["pattern"]))
                 for o in config.get("fileCharacterLimits", [])]
    core = 0
    for p, mode in md:
        if not mode.startswith("100"):
            errors.append(f"{p}: memory Markdown must be a regular file")
            continue
        depth = p.count("/")
        if config.get("maxDepth") is not None and depth > config["maxDepth"]:
            errors.append(f"{p}: depth {depth} exceeds maxDepth {config['maxDepth']}")
        hit = next((o for o in overrides if o[2].match(p)), None)
        limit, source = (hit[1], f"glob '{hit[0]}'") if hit else (config.get("maxFileCharacters"), "maxFileCharacters")
        n = len(read(p))
        if "/" not in p:
            core += n
        if limit is not None and n > limit:
            errors.append(f"{p}: {n} characters exceeds {limit} from {source}")
    if config.get("maxCoreMemoryCharacters") is not None and core > config["maxCoreMemoryCharacters"]:
        errors.append(f"core memory: {core} characters exceeds {config['maxCoreMemoryCharacters']} "
                      "from maxCoreMemoryCharacters")
    return errors


# ---- working tree check (kev-memory context check) and the pre-commit hook ---------------------------------------
def _git(cwd, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def check_worktree(root: str) -> list[str]:
    """Validate the files on disk (tracked + untracked, not ignored), against HEAD for read_only protection."""
    listed = _git(root, "ls-files", "-co", "--exclude-standard", "-z").stdout.split("\0")
    files = [(p, "100644") for p in listed if p and os.path.isfile(os.path.join(root, p))]

    def read(p):
        with open(os.path.join(root, p), encoding="utf-8", errors="replace") as f:
            return f.read()
    errors = []
    config = None
    if os.path.isfile(os.path.join(root, CONFIG_PATH)):
        try:
            config = parse_config(read(CONFIG_PATH))
        except ValueError as e:
            return str(e).split("\n")
    for p, _ in files:
        if p.endswith(".md"):
            head = _git(root, "show", f"HEAD:{p}")
            errors += validate_frontmatter(p, read(p), head.stdout if head.returncode == 0 else None)
    return errors + validate_tree(files, read, config)


def hook_main(argv: list[str] | None = None) -> int:
    """pre-commit: validate the staged tree; a non-zero exit blocks the commit (staged changes stay staged)."""
    errors = []
    if _git(".", "diff", "--cached", "--quiet", "--", CONFIG_PATH).returncode != 0 \
            and os.environ.get(CONFIG_UPDATE_ENV) != "1":
        errors.append(f"{CONFIG_PATH} is protected; changing it needs the user's approval "
                      f"({CONFIG_UPDATE_ENV}=1)")
    staged = []
    for entry in _git(".", "ls-files", "--stage", "-z").stdout.split("\0"):
        if entry:
            meta, path = entry.split("\t", 1)
            staged.append((path, meta.split(" ", 1)[0]))

    def read(p):
        return _git(".", "show", f":{p}").stdout

    def head(p):
        r = _git(".", "show", f"HEAD:{p}")
        return r.stdout if r.returncode == 0 else None
    for line in _git(".", "diff", "--cached", "--name-status", "--no-renames").stdout.splitlines():
        status, _, p = line.partition("\t")
        if status == "D" and frontmatter_value(head(p), "read_only") == "true":
            errors.append(f"{p}: file is read_only and cannot be deleted")
    config = None
    if any(p == CONFIG_PATH for p, _ in staged):
        try:
            config = parse_config(read(CONFIG_PATH))
        except ValueError as e:
            errors += str(e).split("\n")
    for p, _ in staged:
        if p.endswith(".md"):
            errors += validate_frontmatter(p, read(p), head(p))
    errors += validate_tree(staged, read, config)
    if errors:
        print("Context repository validation blocked this commit. No files were committed; staged changes are "
              "still present.\nFix these problems:", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        print("Split files above their limit or move detail behind a folder MEMORY.md index. Do not raise the limits "
              f"in {CONFIG_PATH} unless the user approves it.", file=sys.stderr)
        return 1
    return 0
