#!/usr/bin/env python3
"""Summarize HPO outputs into a CSV.

Reads all experiment subdirectories under output_debug_search/, extracts
config and final metrics, and writes a consolidated CSV report.

Fields included:
- lr (learning_rate)
- cw (contrastive_weight)
- do (dropout)
- nbl (num_blocks)
- best_val_accuracy
- final_train_accuracy
- final_val_accuracy
- final_train_loss
- final_val_loss

Usage:
  python summarize_results.py --root output_debug_search --out summary.csv
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd

def load_experiment(exp_dir: Path):
    """Load config and final metrics from an experiment directory."""
    config_path = exp_dir / "config.json"
    history_path = exp_dir / "training_history.json"
    if not config_path.exists() or not history_path.exists():
        return None

    try:
        with config_path.open("r") as f:
            config = json.load(f)
        with history_path.open("r") as f:
            history = json.load(f)
    except Exception:
        return None

    if not history:
        return None

    last_entry = history[-1]
    train_metrics = last_entry.get("train", {})
    val_metrics = last_entry.get("val", {}) or {}

    row = {
        "experiment": exp_dir.name,
        "lr": config.get("learning_rate"),
        "cw": config.get("contrastive_weight"),
        "do": config.get("dropout"),
        "nbl": config.get("num_blocks"),
        "best_val_accuracy": config.get("best_val_accuracy"),
        "final_train_accuracy": train_metrics.get("accuracy"),
        "final_val_accuracy": val_metrics.get("accuracy"),
        "final_train_loss": train_metrics.get("loss"),
        "final_val_loss": val_metrics.get("loss"),
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment outputs to CSV")
    parser.add_argument("--root", default="output_debug_search", help="Root directory containing experiment subdirs")
    parser.add_argument("--out", default="output_debug_search_summary.csv", help="Output CSV path")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []

    if not root.exists():
        print(f"Root directory not found: {root}")
        return

    for child in root.iterdir():
        if child.is_dir():
            row = load_experiment(child)
            if row:
                rows.append(row)

    if not rows:
        print("No experiments found to summarize.")
        return

    df = pd.DataFrame(rows)
    df.sort_values(by=["best_val_accuracy"], ascending=False, inplace=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote summary for {len(df)} experiments to {args.out}")


if __name__ == "__main__":
    main()
