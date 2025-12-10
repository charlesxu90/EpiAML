#!/usr/bin/env python3
"""Summarize HPO outputs into a CSV.

Reads all experiment subdirectories under output_debug_search/, extracts
config and final metrics, and writes a consolidated CSV report.

Fields included:
- lr (learning_rate)
- cw (contrastive_weight)
- do (dropout)
- nbl (num_blocks)
- input_dropout (input-level dropout for sparse feature learning)
- stride_config (pooling strategy: minimal or aggressive)
- use_attention (whether attention is enabled)
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
import numpy as np

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

    # Compute Pearson correlation between train and val loss across epochs
    train_losses = []
    val_losses = []
    for entry in history:
        t_loss = entry.get("train", {}).get("loss")
        v_loss = entry.get("val", {}).get("loss") if entry.get("val") else None
        if t_loss is not None and v_loss is not None:
            train_losses.append(t_loss)
            val_losses.append(v_loss)
    if len(train_losses) >= 2 and len(val_losses) == len(train_losses):
        corr_matrix = np.corrcoef(train_losses, val_losses)
        corr_train_val_loss = float(corr_matrix[0, 1])
    else:
        corr_train_val_loss = None

    row = {
        "experiment": exp_dir.name,
        "lr": config.get("learning_rate"),
        "cw": config.get("contrastive_weight"),
        "do": config.get("dropout"),
        "nbl": config.get("num_blocks"),
        "input_dropout": config.get("input_dropout"),
        "stride_config": config.get("stride_config"),
        "use_attention": config.get("use_attention"),
        "best_val_accuracy": config.get("best_val_accuracy"),
        "final_train_accuracy": train_metrics.get("accuracy"),
        "final_val_accuracy": val_metrics.get("accuracy"),
        "final_train_loss": train_metrics.get("loss"),
        "final_val_loss": val_metrics.get("loss"),
        "corr_train_val_loss": corr_train_val_loss,
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
