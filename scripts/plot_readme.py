"""README charts, light and dark: RAGAS + doc hit vs stock graphify (e2e) and locked-test nDCG@10 vs BM25.

    uv run --no-project --with matplotlib --with numpy python scripts/plot_readme.py

Style (docs/BRAND.md §6): baseline gray, ours thaw blue, direct labels, no legends, 95% CIs as thin bars, Geist."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/img"
for f in (ROOT / "docs/brand/fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Geist", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "svg.fonttype": "path"})
BLUE, GRAY = "#2F6FEB", "#9AA4B2"
THEMES = {"light": {"bg": "#FFFFFF", "ink": "#0B0F14", "muted": "#5B6573", "base": "#B8C0CC"},
          "dark": {"bg": "#0D1117", "ink": "#F4F6F8", "muted": "#9AA4B2", "base": "#4A5361"}}


def bars(ax, t, rows, base_label, ours_label, xmax=1.0, fmt="{:.2f}"):
    """rows: [(label, base (mean, lo, hi) | None, ours (mean, lo, hi))] -> paired horizontal bars, labels on the bars."""
    h = 0.36
    for i, (label, base, ours) in enumerate(rows):
        y = len(rows) - 1 - i
        for dy, val, color, who in ((h / 2 + 0.02, base, t["base"], base_label), (-h / 2 - 0.02, ours, BLUE, ours_label)):
            m, lo, hi = val
            ax.barh(y + dy, m, height=h, color=color)
            if lo is not None:
                ax.plot([lo, hi], [y + dy] * 2, color=t["ink"], lw=1, alpha=0.55)
            name = f"  {who}" if i == 0 else ""                  # series named once, on the first row
            ax.text(max(m, hi or m) + 0.012 * xmax, y + dy, fmt.format(m) + name, va="center", fontsize=9.5,
                    color=t["ink"] if color == BLUE else t["muted"],
                    fontweight="semibold" if color == BLUE else "regular")
        ax.text(-0.012 * xmax, y, label, ha="right", va="center", fontsize=10.5, color=t["ink"])
    ax.set_xlim(0, xmax * 1.32); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([]); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)


def e2e(t):
    r = json.loads((ROOT / "benchmarks/e2e/results/ragas_report.json").read_text())["table"]
    h = json.loads((ROOT / "benchmarks/e2e/results/hits.json").read_text())["all"]
    g, d = r["graphify-full"], r["defrost"]
    rows = [("Answering doc section\nin context", (h["graphify-full"]["doc_hit"], None, None), (h["defrost"]["doc_hit"], None, None)),
            ("RAGAS answer accuracy", tuple(g["nv_accuracy"]), tuple(d["nv_accuracy"])),
            ("RAGAS context relevance", tuple(g["nv_context_relevance"]), tuple(d["nv_context_relevance"])),
            ("RAGAS groundedness", tuple(g["nv_response_groundedness"]), tuple(d["nv_response_groundedness"]))]
    fig, ax = plt.subplots(figsize=(8.6, 3.9), facecolor=t["bg"]); ax.set_facecolor(t["bg"])
    bars(ax, t, rows, "stock graphify", "graphify + defrost-ai")
    fig.text(0.02, 0.95, "End to end on 3 repos never used in training (100 questions)", fontsize=12.5,
             fontweight="semibold", color=t["ink"])
    fig.text(0.02, 0.885, "RAGAS NVIDIA metrics, 95% bootstrap CIs. Stock graphify uses its paid semantic tier.",
             fontsize=9.5, color=t["muted"])
    fig.subplots_adjust(left=0.27, right=0.99, top=0.83, bottom=0.03)
    return fig


def locked(t):
    # docs/EVALUATION.md "Locked test results (scored once)", weights v1.1.0
    rows = [("Held-out OSS repos\n(106 questions)", (0.703, None, None), (0.840, None, None)),
            ("Private product repos\n(69, in-domain)", (0.684, None, None), (0.835, None, None))]
    fig, ax = plt.subplots(figsize=(8.6, 2.5), facecolor=t["bg"]); ax.set_facecolor(t["bg"])
    bars(ax, t, rows, "BM25", "defrost-ai (accurate)", fmt="{:.3f}")
    fig.text(0.02, 0.92, "Locked test, nDCG@10 (scored once)", fontsize=12.5, fontweight="semibold", color=t["ink"])
    fig.text(0.02, 0.82, "accurate vs BM25: +0.111 and +0.130, both 95% CIs above zero.", fontsize=9.5, color=t["muted"])
    fig.subplots_adjust(left=0.27, right=0.99, top=0.72, bottom=0.04)
    return fig


for name, make in (("chart-e2e", e2e), ("chart-locked-test", locked)):
    for theme, t in THEMES.items():
        fig = make(t)
        fig.savefig(OUT / f"{name}-{theme}.svg", facecolor=t["bg"])
        fig.savefig(f"/tmp/{name}-{theme}.png", facecolor=t["bg"], dpi=110)
        plt.close(fig)
