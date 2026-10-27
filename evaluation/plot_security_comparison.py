#!/usr/bin/env python3
"""Export a standalone four-panel figure from the measured summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    variants = summary["variants"]
    colors = ("#4C78A8", "#F58518")
    labels = ("Original", "Hardened")
    keys = ("original", "hardened")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    fig.suptitle("Soprano RAG: security, answer quality and latency", fontsize=17, weight="bold")

    def bars(ax, groups, get_values, title, ylabel, percent=False):
        x = np.arange(len(groups))
        maximum = 0.0
        for offset, (key, label, color) in enumerate(zip(keys, labels, colors)):
            values = get_values(variants[key])
            maximum = max(maximum, *values)
            rectangles = ax.bar(x + (offset - .5) * .35, values, .35, color=color, label=label)
            ax.bar_label(rectangles, labels=[f"{v:.1f}{'%' if percent else ''}" for v in values], padding=3, fontsize=10)
        ax.set_xticks(x, groups)
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=12)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=.18)
        if percent:
            ax.set_ylim(0, min(108, max(10, np.ceil(maximum * 1.35 / 5) * 5)))
        else:
            ax.margins(y=.22)

    bars(axes[0, 0], ["Direct attacks\n30 cases", "Poisoned context\n20 attempts; 19 injected"],
         lambda v: [v["cohorts"][f"{k}:all"]["success_rate"] * 100 for k in ("direct_attack", "indirect_attack")],
         "Attack success — lower is better", "Successful attacks (%)", True)
    bars(axes[0, 1], ["Korean\n20 questions", "English\n20 questions"],
         lambda v: [v["cohorts"][f"benign:{lang}"]["false_refusal_rate"] * 100 for lang in ("ko", "en")],
         "False refusals on answerable music questions", "False refusals (%)", True)
    bars(axes[1, 0], ["Delivered answers", "All 40 questions\n(refusals/errors score 0)"],
         lambda v: [v["cohorts"]["benign:all"][k] or 0 for k in ("conditional_quality", "effective_quality")],
         "Answer quality — blinded rubric review", "Rubric score / 100", False)
    axes[1, 0].set_ylim(0, 108)
    bars(axes[1, 1], ["Normal music", "Direct attacks", "Poisoned context"],
         lambda v: [v["latency"][k]["median_seconds"] or 0 for k in ("benign", "direct_attack", "indirect_attack")],
         "Warm request latency — median", "Seconds", False)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, legend_labels, loc="upper right", frameon=False, fontsize=11)
    fig.supxlabel("100 authored bilingual cases; 2 repeats; serial inference on one RTX A6000.\n"
                  "Rates/quality: first repeat; Codex review. Latency: per-case mean over repeats. Indirect attacks use forced exposure.",
                  fontsize=9)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.summary.parent / f"comparison.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
