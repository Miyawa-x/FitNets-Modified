#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TEACHER_RE = re.compile(r"teacher epoch\s+(\d+):.*?eval_acc=([0-9.eE+-]+)")
KD_RE = re.compile(r"kd epoch\s+(\d+) eval:.*?acc=([0-9.eE+-]+)")
HINT_RE = re.compile(
    r"stage1 epoch\s+(\d+) eval:\s+hint_mse=([0-9.eE+-]+)"
)
RELATION_RE = re.compile(
    r"relation epoch\s+(\d+) eval:\s+loss=([0-9.eE+-]+)"
)


@dataclass(frozen=True)
class RunCurves:
    method: str
    seed: str
    kd_accuracy: dict[int, float]
    stage1_loss: dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Original and Relation FitNets training logs."
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="METHOD:SEED=PATH",
        help="Add one student run. PATH can be a log file or a directory of logs.",
    )
    parser.add_argument(
        "--teacher-log",
        action="append",
        default=[],
        help="Teacher log file or directory. May be specified more than once.",
    )
    parser.add_argument("--output-dir", default="figures/cifar100")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.2, 0.3, 0.4],
        help="Accuracy thresholds, expressed as fractions.",
    )
    args = parser.parse_args()
    if not args.run:
        parser.error("at least one --run METHOD:SEED=PATH is required")
    return args


def read_path(path_text: str) -> str:
    path = Path(path_text)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.rglob("*.log"))
        if not files:
            raise ValueError(f"no .log files found under {path}")
    else:
        raise FileNotFoundError(path)
    return "\n".join(file.read_text(errors="replace") for file in files)


def parse_points(pattern: re.Pattern[str], text: str) -> dict[int, float]:
    return {int(epoch): float(value) for epoch, value in pattern.findall(text)}


def parse_run(spec: str) -> RunCurves:
    try:
        identity, path = spec.split("=", 1)
        method, seed = identity.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError(
            f"invalid run {spec!r}; expected METHOD:SEED=PATH"
        ) from exc
    if not method or not seed or not path:
        raise ValueError(f"invalid run {spec!r}; fields cannot be empty")

    text = read_path(path)
    hint = parse_points(HINT_RE, text)
    relation = parse_points(RELATION_RE, text)
    if hint and relation:
        raise ValueError(f"run {spec!r} contains both hint and relation Stage 1 logs")
    kd = parse_points(KD_RE, text)
    if not kd and not hint and not relation:
        raise ValueError(f"run {spec!r} contains no recognized training metrics")
    return RunCurves(method, seed, kd, hint or relation)


def grouped_runs(runs: list[RunCurves]) -> dict[str, list[RunCurves]]:
    grouped: dict[str, list[RunCurves]] = {}
    for run in runs:
        grouped.setdefault(run.method, []).append(run)
    return grouped


def aggregate_curve(
    runs: list[RunCurves], attribute: str
) -> tuple[list[int], list[float], list[float]]:
    curves = [getattr(run, attribute) for run in runs]
    epochs = sorted(set().union(*(curve.keys() for curve in curves)))
    means: list[float] = []
    stds: list[float] = []
    for epoch in epochs:
        values = [curve[epoch] for curve in curves if epoch in curve]
        means.append(statistics.fmean(values))
        stds.append(statistics.pstdev(values) if len(values) > 1 else 0.0)
    return epochs, means, stds


def plot_curves(
    grouped: dict[str, list[RunCurves]],
    attribute: str,
    output: Path,
    ylabel: str,
    teacher_best: float | None = None,
    log_scale: bool = False,
) -> None:
    colors = ["#247BA0", "#D1495B", "#2A9D8F", "#E09F3E", "#6D597A"]
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    plotted = False
    for index, (method, runs) in enumerate(grouped.items()):
        epochs, means, stds = aggregate_curve(runs, attribute)
        if not epochs:
            continue
        plotted = True
        color = colors[index % len(colors)]
        lower = [max(0.0, mean - std) for mean, std in zip(means, stds)]
        upper = [mean + std for mean, std in zip(means, stds)]
        ax.plot(epochs, means, color=color, linewidth=2, label=method)
        if len(runs) > 1:
            ax.fill_between(epochs, lower, upper, color=color, alpha=0.18)
    if teacher_best is not None and attribute == "kd_accuracy":
        ax.axhline(
            teacher_best,
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            label=f"Teacher best ({teacher_best:.3f})",
        )
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    if log_scale:
        ax.set_yscale("log")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def method_stats(runs: list[RunCurves]) -> tuple[list[float], list[float]]:
    final = [run.kd_accuracy[max(run.kd_accuracy)] for run in runs if run.kd_accuracy]
    best = [max(run.kd_accuracy.values()) for run in runs if run.kd_accuracy]
    return final, best


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values), (
        statistics.pstdev(values) if len(values) > 1 else 0.0
    )


def write_summary(
    grouped: dict[str, list[RunCurves]], output: Path, teacher_best: float | None
) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "seeds", "final_mean", "final_std", "best_mean", "best_std"]
        )
        if teacher_best is not None:
            writer.writerow(["Teacher", 1, teacher_best, 0.0, teacher_best, 0.0])
        for method, runs in grouped.items():
            final, best = method_stats(runs)
            final_mean, final_std = mean_std(final)
            best_mean, best_std = mean_std(best)
            writer.writerow(
                [method, len(best), final_mean, final_std, best_mean, best_std]
            )


def write_thresholds(
    grouped: dict[str, list[RunCurves]], thresholds: list[float], output: Path
) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "threshold", "reached_seeds", "epoch_mean", "epoch_std"]
        )
        for method, runs in grouped.items():
            for threshold in thresholds:
                reached: list[int] = []
                for run in runs:
                    epochs = [
                        epoch
                        for epoch, accuracy in run.kd_accuracy.items()
                        if accuracy >= threshold
                    ]
                    if epochs:
                        reached.append(min(epochs))
                epoch_mean, epoch_std = mean_std([float(value) for value in reached])
                writer.writerow(
                    [method, threshold, len(reached), epoch_mean, epoch_std]
                )


def plot_summary(
    grouped: dict[str, list[RunCurves]], output: Path, teacher_best: float | None
) -> None:
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    colors: list[str] = []
    if teacher_best is not None:
        labels.append("Teacher")
        means.append(teacher_best)
        stds.append(0.0)
        colors.append("#333333")
    palette = ["#247BA0", "#D1495B", "#2A9D8F", "#E09F3E"]
    for index, (method, runs) in enumerate(grouped.items()):
        _, best = method_stats(runs)
        if not best:
            continue
        mean, std = mean_std(best)
        labels.append(method)
        means.append(mean)
        stds.append(std)
        colors.append(palette[index % len(palette)])
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, width=0.68)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel("Best evaluation accuracy")
    ax.set_ylim(0.0, min(1.0, max(means) * 1.2))
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = [parse_run(spec) for spec in args.run]
    grouped = grouped_runs(runs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_accuracies: list[float] = []
    for path in args.teacher_log:
        teacher_accuracies.extend(parse_points(TEACHER_RE, read_path(path)).values())
    teacher_best = max(teacher_accuracies) if teacher_accuracies else None

    plot_curves(
        grouped,
        "kd_accuracy",
        output_dir / "kd_accuracy.png",
        "Evaluation accuracy",
        teacher_best=teacher_best,
    )
    plot_curves(
        grouped,
        "stage1_loss",
        output_dir / "stage1_loss.png",
        "Stage 1 evaluation loss",
        log_scale=True,
    )
    plot_summary(grouped, output_dir / "best_accuracy.png", teacher_best)
    write_summary(grouped, output_dir / "summary.csv", teacher_best)
    write_thresholds(grouped, args.thresholds, output_dir / "threshold_epochs.csv")
    print(f"wrote plots and CSV summaries to {output_dir}")


if __name__ == "__main__":
    main()
