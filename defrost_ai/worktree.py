"""Background jobs in git worktrees: a job edits files on its own branch, never in the user's checkout, and its
commits come back by a fast-forward merge only when that is clean. Otherwise the branch is kept for review.

Ported from Letta Code (Apache-2.0, https://github.com/letta-ai/letta-code, commit 3687ea51):
  src/agent/memory-worktree.ts   createReflectionMemoryWorktree, integrateMemoryWorkerWorktree -> create, integrate
  src/agent/memory-operation.ts  withMemoryOperation (one writer per checkout)                   -> operation_lock
  src/utils/worktree-lock.ts     the lock lives in the git admin dir, reaped when its process is dead
Changes: Python; the lock is fcntl.flock (released by the OS when the holder dies) instead of a pid file; merges are
fast-forward-only by default and a non-fast-forward or conflicting merge never runs: the branch is kept and
reported, with no LLM conflict repair (Letta has memory-conflict-repair.ts); an "explicit" mode keeps every branch
for review. Commits are authored "defrost-ai". See NOTICE."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

AUTHOR = {"GIT_AUTHOR_NAME": "defrost-ai", "GIT_AUTHOR_EMAIL": "defrost-ai@localhost",
          "GIT_COMMITTER_NAME": "defrost-ai", "GIT_COMMITTER_EMAIL": "defrost-ai@localhost"}
PREFIX = "defrost/"


class GitError(RuntimeError):
    pass


def git(cwd, *args, check: bool = True, env: dict | None = None) -> str:
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=str(cwd), capture_output=True, text=True,
                       env={**os.environ, **AUTHOR, **(env or {})})
    if check and r.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {cwd}: {(r.stderr or r.stdout).strip()}")
    return r.stdout


def common_dir(repo) -> Path:
    d = Path(git(repo, "rev-parse", "--git-common-dir").strip())
    return d if d.is_absolute() else (Path(repo) / d).resolve()


@contextlib.contextmanager
def operation_lock(repo, timeout: float = 120.0):
    """One harness writer per checkout (memory-operation.ts). Blocks up to `timeout` seconds."""
    f = open(common_dir(repo) / "defrost-operation.lock", "a+")
    deadline = time.time() + timeout
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() > deadline:
                f.close()
                raise TimeoutError(f"another defrost-ai job holds {repo}")
            time.sleep(0.1)
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


@dataclass
class Job:
    repo: Path
    dir: Path
    branch: str
    base: str


def worktrees_home() -> Path:
    return Path(os.environ.get("DEFROST_HOME", "~/.defrost-ai")).expanduser() / "worktrees"


def create(repo, label: str, branch: str | None = None) -> Job:
    """New worktree outside the repository on a fresh branch from HEAD (createReflectionMemoryWorktree)."""
    repo = Path(repo).resolve()
    base = git(repo, "rev-parse", "--verify", "HEAD").strip()
    jid = f"{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    branch = branch or f"{PREFIX}{label}/{jid}"
    d = worktrees_home() / f"{repo.name}-{hashlib.sha1(str(repo).encode()).hexdigest()[:8]}" \
        / f"{label.replace('/', '-')}-{jid}"
    d.parent.mkdir(parents=True, exist_ok=True)
    if git(repo, "branch", "--list", branch).strip():
        raise GitError(f"branch {branch} already exists (a previous job's result is waiting for review)")
    git(repo, "worktree", "add", "-q", "-b", branch, str(d), base)
    return Job(repo, d, branch, base)


def _cleanup(job: Job, delete_branch: bool) -> None:
    if job.dir.exists():
        git(job.repo, "worktree", "remove", "--force", str(job.dir), check=False)
        shutil.rmtree(job.dir, ignore_errors=True)
    git(job.repo, "worktree", "prune", check=False)
    if delete_branch:
        git(job.repo, "branch", "-D", job.branch, check=False)


def commit_all(job: Job, message: str) -> bool:
    """Commit everything the job changed in its worktree; False when there was nothing to commit."""
    if not git(job.dir, "status", "--porcelain").strip():
        return False
    git(job.dir, "add", "-A")
    git(job.dir, "commit", "-q", "-m", message)
    return True


def integrate(job: Job, merge: bool = True) -> dict:
    """Bring the job's commits back (integrateMemoryWorkerWorktree). merge=False ("explicit" mode) keeps the branch.
    Only a fast-forward is applied: if the checkout moved on, or its uncommitted files overlap, the branch is kept."""
    n = int(git(job.repo, "rev-list", "--count", f"{job.base}..{job.branch}").strip() or 0)
    if n == 0:
        _cleanup(job, delete_branch=True)
        return {"status": "no_changes"}
    _cleanup(job, delete_branch=False)
    if not merge:
        return {"status": "kept", "branch": job.branch, "commits": n}
    return merge_branch(job.repo, job.branch) | {"commits": n}


def merge_branch(repo, branch: str, ff_only: bool = True) -> dict:
    """Merge a kept job branch into the current branch of `repo`. A failed merge is aborted and the branch kept."""
    if ff_only:
        r = subprocess.run(["git", "merge", "--ff-only", "-q", branch], cwd=str(repo), capture_output=True, text=True,
                           env={**os.environ, **AUTHOR})
    else:
        r = subprocess.run(["git", "-c", "commit.gpgsign=false", "merge", "--no-edit", "-q", "-m",
                            f"merge(defrost): {branch}", branch], cwd=str(repo), capture_output=True, text=True,
                           env={**os.environ, **AUTHOR})
    if r.returncode != 0:
        git(repo, "merge", "--abort", check=False)
        return {"status": "merge_conflict", "branch": branch, "error": (r.stderr or r.stdout).strip()[:400]}
    git(repo, "branch", "-d", branch, check=False)
    return {"status": "merged", "head": git(repo, "rev-parse", "--short", "HEAD").strip()}


def run(repo, label: str, fn, message, merge: bool = True, branch: str | None = None) -> dict:
    """create -> fn(worktree_dir) edits files -> commit -> integrate, under the checkout's operation lock.
    `message` may be a callable, evaluated after fn (so it can report what fn did).
    An exception in fn discards the worktree and the branch (a cancelled worker in Letta's terms)."""
    with operation_lock(repo):
        job = create(repo, label, branch)
        try:
            fn(job.dir)
            commit_all(job, message() if callable(message) else message)
        except Exception as e:                                        # noqa: BLE001
            _cleanup(job, delete_branch=True)
            return {"status": "failed", "error": str(e)}
        return integrate(job, merge=merge)


def branches(repo, prefix: str = PREFIX) -> list[dict]:
    """Job branches waiting for review: name, commits ahead of HEAD, last subject, date."""
    out = []
    fmt = "%(refname:short)\t%(objectname:short)\t%(committerdate:short)\t%(subject)"
    for line in git(repo, "for-each-ref", f"--format={fmt}", f"refs/heads/{prefix}").splitlines():
        name, sha, date, subject = (line.split("\t") + ["", "", ""])[:4]
        ahead = int(git(repo, "rev-list", "--count", f"HEAD..{name}").strip() or 0)
        out.append({"branch": name, "sha": sha, "date": date, "subject": subject, "ahead": ahead})
    return out
