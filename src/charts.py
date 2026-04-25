"""
Generate benchmark charts using matplotlib.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

from src.benchmarks import MODELS
from src.analysis import rank_models


PALETTE = [
    "#4C9BE8", "#E87C4C", "#4CE87C", "#E84C9B",
    "#9B4CE8", "#E8D44C", "#4CE8D4", "#E84C4C",
]

CATS = ["reasoning", "language", "factual_accuracy", "coding", "creativity", "safety", "speed", "cost"]
CAT_LABELS = ["Reasoning", "Language", "Factual\nAccuracy", "Coding", "Creativity", "Safety", "Speed", "Cost"]


def chart_overall_ranking(output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)
    ranked = rank_models()

    names = [m.name for _, m, _ in ranked]
    scores = [s for _, _, s in ranked]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(ranked))]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor="none", height=0.6)
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", ha="left", color="white", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 10.5)
    ax.set_xlabel("Overall Score (out of 10)", color="#888", fontsize=11)
    ax.set_title("LLM Overall Benchmark Rankings", color="white", fontsize=16, fontweight="bold", pad=16)
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.xaxis.label.set_color("#888")
    ax.grid(axis="x", color="#222", linewidth=0.8)
    ax.set_axisbelow(True)
    [t.set_color("white") for t in ax.get_yticklabels()]
    [t.set_color("#888") for t in ax.get_xticklabels()]

    plt.tight_layout()
    path = os.path.join(output_dir, "overall_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def chart_radar(output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)

    N = len(CATS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    for i, model in enumerate(MODELS):
        values = [model.scores.get(c, 0) or 0 for c in CATS]
        values += values[:1]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, values, color=color, linewidth=1.8, label=model.name)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CAT_LABELS, color="white", fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color="#888", fontsize=8)
    ax.spines["polar"].set_color("#333")
    ax.grid(color="#333", linewidth=0.8)
    ax.tick_params(colors="#888")

    legend = ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        facecolor="#1A1F2E", edgecolor="#333",
        labelcolor="white", fontsize=9,
    )

    ax.set_title("Multi-Dimensional Radar Comparison", color="white", fontsize=15, fontweight="bold", pad=24)
    plt.tight_layout()
    path = os.path.join(output_dir, "radar_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def chart_category_heatmap(output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)
    ranked = rank_models()
    model_names = [m.name for _, m, _ in ranked]

    data = np.array([
        [m.scores.get(c) or 0 for c in CATS]
        for _, m, _ in ranked
    ])

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    im = ax.imshow(data, cmap="RdYlGn", vmin=6, vmax=10, aspect="auto")

    ax.set_xticks(np.arange(len(CATS)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([l.replace("\n", " ") for l in CAT_LABELS], color="white", fontsize=10)
    ax.set_yticklabels(model_names, color="white", fontsize=10)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    for i in range(len(model_names)):
        for j in range(len(CATS)):
            val = data[i, j]
            text_color = "black" if val > 8.5 else "white"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color=text_color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Score (out of 10)", color="white", fontsize=10)

    ax.set_title("Benchmark Heatmap — All Models × All Categories",
                 color="white", fontsize=15, fontweight="bold", pad=16)
    ax.spines[:].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def chart_cost_vs_performance(output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    for i, model in enumerate(MODELS):
        cost = model.scores.get("cost", 5)
        overall = model.overall_score()
        color = PALETTE[i % len(PALETTE)]
        ax.scatter(cost, overall, color=color, s=160, zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(
            model.name,
            (cost, overall),
            textcoords="offset points", xytext=(8, 4),
            color=color, fontsize=8.5, fontweight="bold",
        )

    ax.set_xlabel("Cost Score (10 = cheapest)", color="#888", fontsize=11)
    ax.set_ylabel("Overall Score", color="#888", fontsize=11)
    ax.set_title("Cost vs. Performance Tradeoff", color="white", fontsize=15, fontweight="bold", pad=16)
    ax.tick_params(colors="#888")
    ax.spines[:].set_color("#333")
    ax.grid(color="#222", linewidth=0.8)
    [t.set_color("white") for t in ax.get_xticklabels() + ax.get_yticklabels()]

    plt.tight_layout()
    path = os.path.join(output_dir, "cost_vs_performance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def generate_all_charts(output_dir: str = "reports") -> list[str]:
    paths = []
    paths.append(chart_overall_ranking(output_dir))
    paths.append(chart_radar(output_dir))
    paths.append(chart_category_heatmap(output_dir))
    paths.append(chart_cost_vs_performance(output_dir))
    return paths
