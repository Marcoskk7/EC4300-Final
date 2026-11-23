"""Plot paper baselines vs a local Qwen model on HypoSpace.

Paper numbers are from docs/baselines.md (Tables 1 and Boolean).
Local numbers are read from results/legacy/<prefix>_*.json (2026-11-17 runs).

Usage:
  .venv/bin/python tools/plot_paper_vs_8b.py
  .venv/bin/python tools/plot_paper_vs_8b.py \
    --prefix local_27b --label "Local Qwen3.8-27B" \
    --out-prefix paper_vs_27b --three-d local_27b_3d_30.json
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
OUT_PNG = RESULTS / "paper_vs_8b_levels.png"
OUT_GAP = RESULTS / "paper_vs_8b_gaps.png"
OUT_3D = RESULTS / "paper_vs_8b_3d.png"
OUT_CSV = RESULTS / "paper_vs_8b_table.csv"

MODELS = [
    "GPT-5",
    "Gemini-2.5-Pro",
    "Grok-4",
    "DeepSeek-R1",
    "Claude-Opus-4",
    "GPT-4o",
    "LLaMA-3.3-70B",
]

PAPER = {
    "causal": {
        "title": "Causal n=4 (Level 1)",
        "VR": [100, 100, 100, 100, 89.3, 73.8, 30.0],
        "NR": [100, 100, 100, 98.2, 86.2, 83.2, 83.3],
        "RR": [100, 100, 100, 98.2, 77.5, 60.9, 24.0],
    },
    "boolean_basic": {
        "title": "Boolean basic (Level 1)",
        "VR": [100, 100, 89.3, 83.9, 88.3, 88.4, 95.6],
        "NR": [100, 100, 89.3, 82.9, 69.9, 38.1, 30.1],
        "RR": [100, 100, 89.3, 82.9, 69.9, 38.1, 29.0],
    },
    "boolean_extended": {
        "title": "Boolean extended (Level 2)",
        "VR": [74.9, 72.3, 52.2, 57.2, 52.7, 83.0, 55.8],
        "NR": [74.9, 72.3, 52.2, 57.2, 54.9, 31.6, 18.4],
        "RR": [72.3, 72.3, 50.8, 53.9, 48.4, 27.3, 14.9],
    },
    "boolean_full": {
        "title": "Boolean full (Level 3)",
        "VR": [65.1, 52.2, 40.6, 41.0, 36.9, 68.1, 66.6],
        "NR": [49.9, 48.9, 37.9, 36.6, 24.2, 21.1, 14.3],
        "RR": [48.0, 47.4, 36.5, 36.0, 23.8, 14.1, 11.0],
    },
}

LOCAL_FILES = {
    "causal": "local_8b_causal_n4_full.json",
    "boolean_basic": "local_8b_boolean_basic.json",
    "boolean_extended": "local_8b_boolean_extended.json",
    "boolean_full": "local_8b_boolean_full.json",
}

LOCAL_LABEL = "Local Qwen3-8B"
THREE_D_FILE = "local_8b_3d_30.json"


def _to_metrics(stats):
    keys = {"VR": "valid_rate", "NR": "novelty_rate", "RR": "recovery_rate"}
    return {metric: 100.0 * stats[stat_key]["mean"] for metric, stat_key in keys.items()}


def load_local(name):
    path = RESULTS / LOCAL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run the benchmark first")
    return _to_metrics(json.loads(path.read_text())["statistics"])


def load_3d():
    path = RESULTS / THREE_D_FILE
    if path.exists():
        return _to_metrics(json.loads(path.read_text())["statistics"])
    return None


def plot_levels(data, out_path):
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), constrained_layout=True)
    fig.suptitle(
        f"HypoSpace: paper baselines vs {LOCAL_LABEL}",
        fontsize=14,
        fontweight="bold",
    )
    colors = {"VR": "#B8B8B8", "NR": "#4C78A8", "RR": "#F58518"}
    labels = MODELS + [LOCAL_LABEL]
    for ax, key in zip(axes, ["causal", "boolean_basic", "boolean_extended", "boolean_full"]):
        paper = PAPER[key]
        local = data[key]
        y = np.arange(len(labels))
        height = 0.25
        for offset, metric in zip((-height, 0.0, height), ("VR", "NR", "RR")):
            values = paper[metric] + [local[metric]]
            ax.barh(
                y + offset,
                values,
                height=height,
                color=colors[metric],
                edgecolor="black" if metric == "RR" else "none",
                linewidth=0.4,
                label=metric if key == "causal" else None,
            )
            for yi, value in zip(y, values):
                ax.text(
                    value + 1.0,
                    yi + offset,
                    f"{value:.1f}",
                    va="center",
                    fontsize=7,
                    color="#222222",
                )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.get_yticklabels()[-1].set_fontweight("bold")
        ax.axhspan(len(labels) - 1.4, len(labels) - 0.6, color="#FFF3CD", zorder=0)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Percent")
        ax.set_title(paper["title"], loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    axes[0].legend(loc="lower right", ncol=3, frameon=False, fontsize=9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_gaps(data, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(
        f"VR/NR/RR gap: {LOCAL_LABEL} minus paper model (percentage points)",
        fontsize=14,
        fontweight="bold",
    )
    y = np.arange(len(MODELS))
    for ax, key in zip(axes.ravel(), ["causal", "boolean_basic", "boolean_extended", "boolean_full"]):
        paper = PAPER[key]
        local = data[key]
        vr_gap = local["VR"] - np.array(paper["VR"])
        nr_gap = local["NR"] - np.array(paper["NR"])
        rr_gap = local["RR"] - np.array(paper["RR"])
        ax.barh(y - 0.27, vr_gap, height=0.25, color="#B8B8B8", label="VR gap")
        ax.barh(y, nr_gap, height=0.25, color="#4C78A8", label="NR gap")
        ax.barh(y + 0.27, rr_gap, height=0.25, color="#F58518", label="RR gap")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(MODELS)
        ax.set_xlim(-100, 60)
        ax.set_xlabel("local - paper (pp)")
        ax.set_title(paper["title"], loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_csv(data, local_3d):
    rows = []
    for key in ["causal", "boolean_basic", "boolean_extended", "boolean_full"]:
        paper = PAPER[key]
        for i, model in enumerate(MODELS):
            rows.append(
                {
                    "setting": key,
                    "model": model,
                    "VR": paper["VR"][i],
                    "NR": paper["NR"][i],
                    "RR": paper["RR"][i],
                    "source": "paper",
                }
            )
        local = data[key]
        rows.append(
            {
                "setting": key,
                "model": LOCAL_LABEL,
                "VR": round(local["VR"], 3),
                "NR": round(local["NR"], 3),
                "RR": round(local["RR"], 3),
                "source": LOCAL_LABEL,
            }
        )
    if local_3d:
        rows.append(
            {
                "setting": "3d_g3_b3",
                "model": LOCAL_LABEL,
                "VR": round(local_3d["VR"], 3),
                "NR": round(local_3d["NR"], 3),
                "RR": round(local_3d["RR"], 3),
                "source": LOCAL_LABEL,
            }
        )
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", "model", "VR", "NR", "RR", "source"])
        writer.writeheader()
        writer.writerows(rows)


def plot_3d(local_3d):
    if not local_3d:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    labels = ["VR", "NR", "RR"]
    values = [local_3d[key] for key in labels]
    colors = ["#B8B8B8", "#4C78A8", "#F58518"]
    ax.barh(labels, values, color=colors, height=0.55)
    for yi, value in zip(range(len(labels)), values):
        ax.text(value + 1.0, yi, f"{value:.1f}", va="center", fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Percent")
    ax.set_title(
        f"3D g3/b3: {LOCAL_LABEL} (30/129 subset, |H_O|=27)",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.savefig(OUT_3D, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="local_8b")
    parser.add_argument("--label", default="Local Qwen3-8B")
    parser.add_argument("--out-prefix", default="paper_vs_8b")
    parser.add_argument("--three-d", default="local_8b_3d_30.json")
    args = parser.parse_args()

    global LOCAL_FILES, LOCAL_LABEL, THREE_D_FILE, OUT_PNG, OUT_GAP, OUT_3D, OUT_CSV
    LOCAL_LABEL = args.label
    THREE_D_FILE = args.three_d
    LOCAL_FILES = {
        "causal": f"{args.prefix}_causal_n4_full.json",
        "boolean_basic": f"{args.prefix}_boolean_basic.json",
        "boolean_extended": f"{args.prefix}_boolean_extended.json",
        "boolean_full": f"{args.prefix}_boolean_full.json",
    }
    OUT_PNG = RESULTS / f"{args.out_prefix}_levels.png"
    OUT_GAP = RESULTS / f"{args.out_prefix}_gaps.png"
    OUT_3D = RESULTS / f"{args.out_prefix}_3d.png"
    OUT_CSV = RESULTS / f"{args.out_prefix}_table.csv"

    RESULTS.mkdir(exist_ok=True)
    data = {key: load_local(key) for key in LOCAL_FILES}
    local_3d = load_3d()
    plot_levels(data, OUT_PNG)
    plot_gaps(data, OUT_GAP)
    plot_3d(local_3d)
    write_csv(data, local_3d)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_GAP}")
    if local_3d:
        print(f"wrote {OUT_3D}")
    print(f"wrote {OUT_CSV}")
    if local_3d:
        print(f"3D {LOCAL_LABEL}:", {k: round(v, 2) for k, v in local_3d.items()})


if __name__ == "__main__":
    main()
