#!/usr/bin/env python3
"""Plot Flood-JEPA pretraining curves from training_rank0.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def load_records(path: Path) -> tuple[list[dict], int]:
    """Load JSONL and remove records duplicated by restarted runs."""
    unique: dict[tuple[int, int, int], dict] = {}
    raw_count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            required = {
                "epoch",
                "iteration",
                "global_step",
                "loss",
                "prediction_loss",
                "context_loss",
                "context_weight",
                "learning_rate",
                "weight_decay",
                "ema",
            }
            missing = required.difference(record)
            if missing:
                raise ValueError(f"Missing fields at {path}:{line_number}: {sorted(missing)}")
            key = (
                int(record["epoch"]),
                int(record["iteration"]),
                int(record["global_step"]),
            )
            unique[key] = record

    records = sorted(unique.values(), key=lambda item: int(item["global_step"]))
    if not records:
        raise ValueError(f"No training records found in {path}")
    for record in records:
        for field in (
            "loss",
            "prediction_loss",
            "context_loss",
            "context_weight",
            "learning_rate",
            "weight_decay",
            "ema",
        ):
            if not math.isfinite(float(record[field])):
                raise ValueError(
                    f"Non-finite {field} at global step {record['global_step']}"
                )
    return records, raw_count - len(records)


def aggregate_epochs(records: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["epoch"])].append(record)

    summary = []
    for epoch in sorted(grouped):
        values = grouped[epoch]

        def mean(field: str) -> float:
            return sum(float(item[field]) for item in values) / len(values)

        summary.append(
            {
                "epoch": epoch,
                "logged_batches": len(values),
                "total_loss": mean("loss"),
                "prediction_loss": mean("prediction_loss"),
                "context_loss": mean("context_loss"),
                "weighted_context_loss": sum(
                    float(item["context_weight"]) * float(item["context_loss"])
                    for item in values
                )
                / len(values),
                "context_weight": mean("context_weight"),
                "learning_rate": mean("learning_rate"),
                "weight_decay": mean("weight_decay"),
                "ema": mean("ema"),
            }
        )
    return summary


def write_csv(summary: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def plot(summary: list[dict], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required for plotting; install it with: pip install matplotlib"
        ) from error
    epochs = [item["epoch"] for item in summary]
    figure, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(epochs, [item["total_loss"] for item in summary], label="Total loss")
    axes[0].plot(
        epochs,
        [item["prediction_loss"] for item in summary],
        label="Masked prediction loss",
    )
    axes[0].plot(
        epochs,
        [item["context_loss"] for item in summary],
        label="Raw context loss",
    )
    axes[0].plot(
        epochs,
        [item["weighted_context_loss"] for item in summary],
        label="Weighted context contribution",
        linestyle="--",
    )
    axes[0].set_ylabel("L1 latent loss")
    axes[0].set_title("Flood-JEPA pretraining losses (epoch means of logged rank-0 batches)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [item["context_weight"] for item in summary],
        color="tab:purple",
        label="Context weight",
    )
    axes[1].plot(
        epochs,
        [item["ema"] for item in summary],
        color="tab:green",
        label="EMA momentum",
    )
    axes[1].set_ylabel("Coefficient")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    learning_rate_axis = axes[2]
    weight_decay_axis = learning_rate_axis.twinx()
    learning_rate_axis.plot(
        epochs,
        [item["learning_rate"] for item in summary],
        color="tab:blue",
        label="Learning rate",
    )
    weight_decay_axis.plot(
        epochs,
        [item["weight_decay"] for item in summary],
        color="tab:orange",
        label="Weight decay",
    )
    learning_rate_axis.set_xlabel("Epoch")
    learning_rate_axis.set_ylabel("Learning rate", color="tab:blue")
    weight_decay_axis.set_ylabel("Weight decay", color="tab:orange")
    learning_rate_axis.grid(alpha=0.25)
    lines = learning_rate_axis.lines + weight_decay_axis.lines
    learning_rate_axis.legend(lines, [line.get_label() for line in lines])

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to training_rank0.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG destination (default: <log-directory>/pretraining_curves.png)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV destination (default: <log-directory>/pretraining_epoch_summary.csv)",
    )
    args = parser.parse_args()

    log_path = args.log.expanduser().resolve()
    output_path = args.output or log_path.with_name("pretraining_curves.png")
    csv_path = args.csv or log_path.with_name("pretraining_epoch_summary.csv")
    records, duplicate_count = load_records(log_path)
    summary = aggregate_epochs(records)
    plot(summary, output_path)
    write_csv(summary, csv_path)
    print(
        json.dumps(
            {
                "records": len(records),
                "duplicates_removed": duplicate_count,
                "epochs": len(summary),
                "last_global_step": int(records[-1]["global_step"]),
                "plot": str(output_path.resolve()),
                "csv": str(csv_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
