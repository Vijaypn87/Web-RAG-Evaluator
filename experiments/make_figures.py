"""Draw the README / report figures from experiment summary CSVs.

    python -m experiments.make_figures \
        --retrieval experiments/results/rag_eval_45-retrieval \
        --full experiments/results/rag_eval_45-full \
        --out docs/figures

Needs matplotlib (pip install matplotlib); the rest of the project does not.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Chunk size is ordered, so it gets one hue from light to dark. Each line
# also has its own marker and a direct label, so color is never the only cue.
SERIES = {
    400: {"color": "#5bb57f", "marker": "o"},
    800: {"color": "#2f9e5f", "marker": "s"},
    1200: {"color": "#145a32", "marker": "^"},
}
INK = "#1f2a24"
MUTED = "#5d6b63"
GRID = "#e3e9e5"


def read_summary(folder: Path) -> list[dict]:
    with (folder / "summary.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    ax.set_xlabel("top-k (passages retrieved)", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


def _spread(values: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so no two are closer than `gap`.

    Works top-down and only moves labels downward, so the highest line keeps
    its label in place. Ties keep series order (larger chunk size on top).
    """
    order = sorted(range(len(values)), key=lambda i: (-values[i], -i))
    placed = list(values)
    for above, below in zip(order, order[1:]):
        if placed[above] - placed[below] < gap:
            placed[below] = placed[above] - gap
    return placed


def _panel(ax, rows: list[dict], metric: str, title: str, ylabel: str, ylim) -> None:
    ks = sorted({int(row["top_k"]) for row in rows})
    labels: list[tuple[str, float, float]] = []
    for size, style in SERIES.items():
        points = sorted(
            (int(row["top_k"]), _number(row[metric]), _number(row.get(f"{metric}_ci95", "")))
            for row in rows
            if int(row["chunk_size"]) == size and _number(row[metric]) is not None
        )
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cis = [p[2] or 0 for p in points]
        ax.fill_between(
            xs,
            [y - c for y, c in zip(ys, cis)],
            [min(1.0, y + c) for y, c in zip(ys, cis)],
            color=style["color"],
            alpha=0.10,
            linewidth=0,
        )
        ax.plot(xs, ys, color=style["color"], linewidth=2, marker=style["marker"],
                markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=f"{size} chars")
        labels.append((f"{size}", xs[-1], ys[-1]))
    gap = (ylim[1] - ylim[0]) * 0.06
    for (text, x, _), y in zip(labels, _spread([label[2] for label in labels], gap)):
        ax.annotate(text, (x, y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK, annotation_clip=False)
    ax.set_xticks(ks)
    ax.set_xlim(min(ks) - 0.5, max(ks) + 1.6)
    ax.set_ylim(*ylim)
    _style(ax, title, ylabel)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--full", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    retrieval = read_summary(args.retrieval)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    _panel(axes[0], retrieval, "hit_at_k", "Hit@k: evidence found in the top-k", "Hit@k", (0.4, 1.04))
    _panel(axes[1], retrieval, "reciprocal_rank", "MRR: how high the evidence ranks", "MRR", (0.4, 1.04))
    axes[1].legend(title="Chunk size", frameon=False, fontsize=8.5, title_fontsize=8.5,
                   loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.out / "retrieval.svg", bbox_inches="tight")
    fig.savefig(args.out / "retrieval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if args.full:
        full = read_summary(args.full)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        _panel(axes[0], full, "answer_correctness", "Answer correctness (LLM judge)", "Score (0–1)", (0.3, 1.04))
        _panel(axes[1], full, "citation_validity", "Citation validity", "Score (0–1)", (0.0, 1.04))
        axes[1].legend(title="Chunk size", frameon=False, fontsize=8.5, title_fontsize=8.5,
                       loc="lower right", labelcolor=INK)
        fig.tight_layout()
        fig.savefig(args.out / "answers.svg", bbox_inches="tight")
        fig.savefig(args.out / "answers.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"Figures written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
