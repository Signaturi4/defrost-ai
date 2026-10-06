"""Personal settings: one small, commented TOML file instead of environment variables.

    ~/.defrost-ai/config.toml          (DEFROST_HOME moves it)

    defrost config                      show every setting, its value and where the value comes from
    defrost config search.mode fast     change one
    defrost config search.mode --reset  back to the default

Precedence: environment variable (if set) > config.toml > default. The environment variables still work, so CI and
scripts can override a setting for one run."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Setting:
    key: str
    default: object
    choices: tuple = ()
    env: str = ""
    help: str = ""


SETTINGS = [
    Setting("search.mode", "accurate", ("accurate", "fast"), "DEFROST_SEARCH_MODE",
            "accurate: best results, reranks when the two retrievers disagree (~1-2 s on Apple Silicon). "
            "fast: no reranker (~0.1 s), a little less accurate."),
    Setting("search.k", "auto", (), "DEFROST_SEARCH_K",
            'sections per answer: a number, or "auto" (1-5, fewer when the top hit is clearly right)'),
    Setting("project.doc_trust", "low", ("low", "high"), "DEFROST_DOC_TRUST",
            "default for new projects. low: docs are hints, Claude checks the code. high: docs are reliable."),
    Setting("models.backend", "auto", ("auto", "mlx", "torch"), "DEFROST_BACKEND",
            "auto: MLX on Apple Silicon, PyTorch elsewhere. torch on a Mac is a debug/reference path (slower)."),
    Setting("models.rerank_dtype", "auto", ("auto", "fp16", "bf16", "fp32"), "DEFROST_RERANK_DTYPE",
            "auto: fp16 on MLX (fp32 retry on overflow), bf16 on CUDA, fp32 on CPU and on PyTorch-MPS"),
    Setting("models.rerank_cache", 20000, (), "DEFROST_RERANK_CACHE",
            "reranker scores kept in memory for repeated questions (0 = off)"),
    Setting("models.allow_older_weights", False, (), "DEFROST_ALLOW_OLDER_WEIGHTS",
            "run older cached weights when the pinned version cannot be downloaded (every result then says so). "
            "Off: fail with a clear error instead."),
    Setting("prompt_context.min_cosine", 0.34, (), "DEFROST_PROMPT_MIN_COSINE",
            "the prompt hook adds memory sections to a question only when the best one is at least this similar "
            "(0-1; higher = fewer, surer injections)"),
    Setting("prompt_context.mode", "accurate", ("accurate", "fast"), "DEFROST_PROMPT_MODE",
            "search mode of the prompt hook. accurate reranks (~1-2 s; falls back to fast when the reranker is cold)"),
    Setting("prompt_context.k", 5, (), "DEFROST_PROMPT_K",
            "sections the prompt hook adds (both sides of a doc conflict need room)"),
    Setting("retrieval.encoder", "defrost-ret-b", ("defrost-ret-b", "qwen3-emb-0.6b"), "DEFROST_ENCODER",
            "dense retriever for new builds. A memory remembers its encoder; changing it needs a full rebuild."),
    Setting("service.port", 8765, (), "",
            "local port of the background search service"),
]
BY_KEY = {s.key: s for s in SETTINGS}


def path() -> Path:
    return Path(os.environ.get("DEFROST_HOME", "~/.defrost-ai")).expanduser() / "config.toml"


def _file() -> dict:
    try:
        raw = tomllib.loads(path().read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    flat = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat[f"{section}.{k}"] = v
    return flat


def _coerce(s: Setting, value):
    if isinstance(s.default, bool):
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(s.default, float) and not isinstance(value, (int, float)):
        value = float(value)
    if isinstance(s.default, int) and not isinstance(value, int):
        value = int(value)
    if s.choices and value not in s.choices:
        raise ValueError(f"{s.key} must be one of {', '.join(s.choices)} (got {value!r})")
    return value


def get(key: str):
    s = BY_KEY[key]
    if s.env and os.environ.get(s.env):
        return _coerce(s, os.environ[s.env])
    v = _file().get(key)
    return s.default if v is None else _coerce(s, v)


def source(key: str) -> str:
    s = BY_KEY[key]
    if s.env and os.environ.get(s.env):
        return f"env {s.env}"
    return "config.toml" if key in _file() else "default"


def set_(key: str, value) -> None:
    if key not in BY_KEY:
        raise KeyError(f"unknown setting {key!r}; see `defrost config`")
    s = BY_KEY[key]
    values = _file()
    if value is None:
        values.pop(key, None)
    else:
        values[key] = _coerce(s, int(value) if isinstance(s.default, int) and not isinstance(s.default, bool)
                              and str(value).isdigit() else value)
    _write(values)


def _write(values: dict) -> None:
    """Write every known setting, commented, so the file explains itself; unset ones stay commented out."""
    lines = ["# defrost-ai settings. Edit here or with `defrost config KEY VALUE`; delete a line for the default.", ""]
    section = None
    for s in SETTINGS:
        sec, name = s.key.split(".", 1)
        if sec != section:
            lines += ([""] if section else []) + [f"[{sec}]"]
            section = sec
        lines.append(f"# {s.help}" + (f" Choices: {', '.join(s.choices)}." if s.choices else ""))
        v = values.get(s.key)
        shown = s.default if v is None else v
        text = f'"{shown}"' if isinstance(shown, str) else str(shown).lower() if isinstance(shown, bool) else str(shown)
        lines.append(f"{'' if v is not None else '# '}{name} = {text}")
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")


def show() -> str:
    rows = [f"settings file: {path()}", ""]
    for s in SETTINGS:
        rows.append(f"  {s.key:22s} {str(get(s.key)):10s} ({source(s.key)})")
        rows.append(f"  {'':22s} {s.help}")
    return "\n".join(rows)


def export_env() -> None:
    """Make config.toml values visible to code that reads the environment (model backend, dtype, cache)."""
    for s in SETTINGS:
        if s.env and not os.environ.get(s.env):
            v = _file().get(s.key)
            if v is not None and v != "auto":
                os.environ[s.env] = str(v).lower() if isinstance(v, bool) else str(v)
