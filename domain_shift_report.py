#!/usr/bin/env python3
"""Domain shift report between bulk and nanopore HDF5 methylation matrices.

Expected H5 keys (preferred):
- data: (n_samples, n_features) float32/float64 in [0,1]
- feature_names: (n_features,) strings
- labels: (n_samples,) strings (optional)

This script is careful to stream across features to avoid loading the full matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


@dataclass(frozen=True)
class ComputeConfig:
    backend: str  # auto|cpu|torch
    device: str


def _select_compute(cfg: ComputeConfig) -> Tuple[str, str]:
    """Return (backend, device) after resolving 'auto' and availability."""

    backend = cfg.backend
    device = cfg.device

    if backend not in {"auto", "cpu", "torch"}:
        raise ValueError(f"Invalid backend: {backend}")

    if backend == "cpu":
        return "cpu", "cpu"

    if backend in {"auto", "torch"} and _torch_available():
        import torch

        # Default to cuda:0 if available, otherwise cpu.
        if torch.cuda.is_available():
            try:
                torch.device(device)  # validate
                return "torch", device
            except Exception:
                return "torch", "cuda:0"
        return "torch", "cpu"

    # Fallback: numpy CPU
    return "cpu", "cpu"


@dataclass(frozen=True)
class ProgressConfig:
    mode: str  # auto|tqdm|print|none
    log_every: int


def _progress_iter(
    iterable: Iterable[Tuple[int, int]],
    *,
    total: int,
    desc: str,
    cfg: ProgressConfig,
) -> Iterable[Tuple[int, int]]:
    """Wrap an iterable with progress reporting.

    - mode=auto: use tqdm if available and stderr is a TTY, else print.
    - mode=tqdm: require tqdm (falls back to print if not installed).
    - mode=print: periodic logs.
    - mode=none: no progress output.
    """

    if cfg.mode == "none":
        yield from iterable
        return

    use_tqdm = cfg.mode in {"auto", "tqdm"}
    if use_tqdm:
        try:
            from tqdm import tqdm  # type: ignore

            if cfg.mode == "tqdm" or (cfg.mode == "auto" and sys.stderr.isatty()):
                yield from tqdm(iterable, total=total, desc=desc, unit="chunk")
                return
        except Exception:
            # tqdm not installed or not usable; fall back to print.
            pass

    # Print fallback.
    log_every = max(1, int(cfg.log_every))
    start_t = time.time()
    last_print = 0
    for i, item in enumerate(iterable, start=1):
        if i == 1 or i == total or (i - last_print) >= log_every:
            elapsed = max(1e-9, time.time() - start_t)
            rate = i / elapsed
            eta = (total - i) / max(1e-9, rate)
            pct = 100.0 * i / max(1, total)
            print(f"[{desc}] {i}/{total} ({pct:.1f}%) eta={eta:.1f}s", file=sys.stderr)
            last_print = i
        yield item


def _as_str_array(x: np.ndarray) -> np.ndarray:
    # h5py often returns bytes for object/string datasets.
    if x.dtype.kind in {"S", "O", "U"}:
        out = x.astype(object, copy=False)
        def _coerce(v):
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            return str(v)
        return np.array([_coerce(v) for v in out], dtype=object)
    return x.astype(object)


def _ks_2samp_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic D (no p-value)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    all_vals = np.sort(np.concatenate([x_sorted, y_sorted]))
    # Empirical CDFs evaluated at combined support.
    cdf_x = np.searchsorted(x_sorted, all_vals, side="right") / x_sorted.size
    cdf_y = np.searchsorted(y_sorted, all_vals, side="right") / y_sorted.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


@dataclass
class H5View:
    path: str
    data_key: str
    feature_names_key: str
    labels_key: Optional[str]


def _open_h5_view(path: str) -> H5View:
    with h5py.File(path, "r") as f:
        _ = f.keys()

    # Prefer MARLIN fast keys; allow simple fallbacks.
    data_key = "data" if _h5_has(path, "data") else ("X" if _h5_has(path, "X") else None)
    feature_names_key = (
        "feature_names" if _h5_has(path, "feature_names") else ("features" if _h5_has(path, "features") else None)
    )
    labels_key = "labels" if _h5_has(path, "labels") else ("y" if _h5_has(path, "y") else None)

    if data_key is None:
        raise KeyError(f"{path}: missing dataset key for data (tried: data, X)")
    if feature_names_key is None:
        raise KeyError(f"{path}: missing dataset key for feature_names (tried: feature_names, features)")

    return H5View(path=path, data_key=data_key, feature_names_key=feature_names_key, labels_key=labels_key)


def _h5_has(path: str, key: str) -> bool:
    try:
        with h5py.File(path, "r") as f:
            return key in f
    except OSError:
        return False


def _read_feature_names(view: H5View) -> np.ndarray:
    with h5py.File(view.path, "r") as f:
        names = f[view.feature_names_key][...]
    return _as_str_array(names)


def _read_labels(view: H5View) -> Optional[np.ndarray]:
    if not view.labels_key:
        return None
    with h5py.File(view.path, "r") as f:
        labels = f[view.labels_key][...]
    return _as_str_array(labels)


def _iter_feature_chunks(n_features: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n_features, chunk_size):
        end = min(n_features, start + chunk_size)
        yield start, end


def _sample_means_streaming(
    view: H5View,
    chunk_size: int,
    progress: ProgressConfig,
    desc: str,
    compute: ComputeConfig,
) -> np.ndarray:
    with h5py.File(view.path, "r") as f:
        dset = f[view.data_key]
        n_samples, n_features = dset.shape
        backend, device = _select_compute(compute)

        acc: Optional[np.ndarray] = None
        acc_t = None
        dev = None

        if backend == "torch":
            import torch

            dev = torch.device(device)
            acc_t = torch.zeros((n_samples,), dtype=torch.float32, device=dev)
        else:
            acc = np.zeros((n_samples,), dtype=np.float64)
        total = int(math.ceil(n_features / float(chunk_size)))
        chunks = _iter_feature_chunks(n_features, chunk_size)
        for start, end in _progress_iter(chunks, total=total, desc=desc, cfg=progress):
            x_np = dset[:, start:end].astype(np.float32, copy=False)
            if backend == "torch":
                import torch

                assert acc_t is not None and dev is not None
                x_t = torch.from_numpy(x_np).to(dev, non_blocking=False)
                acc_t += torch.sum(x_t, dim=1)
            else:
                assert acc is not None
                acc += np.sum(x_np, axis=1)

        if backend == "torch":
            import torch

            assert acc_t is not None

            out = (acc_t / float(n_features)).detach().to("cpu")
            return out.to(dtype=torch.float64).numpy()

        assert acc is not None
        return (acc / float(n_features)).astype(np.float64)


def _compute_feature_stats(
    bulk_view: H5View,
    nano_view: H5View,
    threshold: float,
    chunk_size: int,
    progress: ProgressConfig,
    compute: ComputeConfig,
) -> Dict[str, np.ndarray]:
    """Compute per-feature mean/var and P(x>=threshold) for both datasets."""

    with h5py.File(bulk_view.path, "r") as fb, h5py.File(nano_view.path, "r") as fn:
        xb = fb[bulk_view.data_key]
        xn = fn[nano_view.data_key]

        if xb.shape[1] != xn.shape[1]:
            raise ValueError(f"Feature count mismatch: bulk={xb.shape[1]} nano={xn.shape[1]}")

        n_bulk, n_features = xb.shape
        n_nano = xn.shape[0]

        mean_bulk = np.zeros((n_features,), dtype=np.float64)
        mean_nano = np.zeros((n_features,), dtype=np.float64)
        m2_bulk = np.zeros((n_features,), dtype=np.float64)  # E[x^2]
        m2_nano = np.zeros((n_features,), dtype=np.float64)
        ppos_bulk = np.zeros((n_features,), dtype=np.float64)
        ppos_nano = np.zeros((n_features,), dtype=np.float64)

        backend, device = _select_compute(compute)
        if backend == "torch":
            import torch

            dev = torch.device(device)
            thr = float(threshold)

        total = int(math.ceil(n_features / float(chunk_size)))
        chunks = _iter_feature_chunks(n_features, chunk_size)
        for start, end in _progress_iter(chunks, total=total, desc="feature-stats", cfg=progress):
            b_np = xb[:, start:end].astype(np.float32, copy=False)
            n_np = xn[:, start:end].astype(np.float32, copy=False)

            if backend == "torch":
                import torch

                b_t = torch.from_numpy(b_np).to(dev, non_blocking=False)
                n_t = torch.from_numpy(n_np).to(dev, non_blocking=False)

                mb = torch.mean(b_t, dim=0)
                mn = torch.mean(n_t, dim=0)
                m2b = torch.mean(b_t * b_t, dim=0)
                m2n = torch.mean(n_t * n_t, dim=0)
                pb = torch.mean((b_t >= thr).to(torch.float32), dim=0)
                pn = torch.mean((n_t >= thr).to(torch.float32), dim=0)

                mean_bulk[start:end] = mb.detach().to("cpu").to(torch.float64).numpy()
                mean_nano[start:end] = mn.detach().to("cpu").to(torch.float64).numpy()
                m2_bulk[start:end] = m2b.detach().to("cpu").to(torch.float64).numpy()
                m2_nano[start:end] = m2n.detach().to("cpu").to(torch.float64).numpy()
                ppos_bulk[start:end] = pb.detach().to("cpu").to(torch.float64).numpy()
                ppos_nano[start:end] = pn.detach().to("cpu").to(torch.float64).numpy()
            else:
                mean_bulk[start:end] = np.mean(b_np, axis=0)
                mean_nano[start:end] = np.mean(n_np, axis=0)
                m2_bulk[start:end] = np.mean(b_np * b_np, axis=0)
                m2_nano[start:end] = np.mean(n_np * n_np, axis=0)
                ppos_bulk[start:end] = np.mean(b_np >= threshold, axis=0)
                ppos_nano[start:end] = np.mean(n_np >= threshold, axis=0)

        var_bulk = np.maximum(0.0, m2_bulk - mean_bulk * mean_bulk)
        var_nano = np.maximum(0.0, m2_nano - mean_nano * mean_nano)

        return {
            "mean_bulk": mean_bulk,
            "mean_nano": mean_nano,
            "var_bulk": var_bulk,
            "var_nano": var_nano,
            "ppos_bulk": ppos_bulk,
            "ppos_nano": ppos_nano,
        }


def _write_topk_csv(
    out_csv: str,
    feature_names: Sequence[str],
    stats: Dict[str, np.ndarray],
    topk: int,
) -> None:
    ppos_bulk = stats["ppos_bulk"]
    ppos_nano = stats["ppos_nano"]
    delta_ppos = ppos_nano - ppos_bulk
    score = np.abs(delta_ppos)

    k = min(int(topk), score.size)
    top_idx = np.argpartition(-score, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-score[top_idx])]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "feature",
                "abs_delta_ppos",
                "delta_ppos(nano-bulk)",
                "ppos_bulk",
                "ppos_nano",
                "mean_bulk",
                "mean_nano",
                "var_bulk",
                "var_nano",
            ]
        )
        for r, i in enumerate(top_idx, start=1):
            w.writerow(
                [
                    r,
                    feature_names[int(i)],
                    float(score[int(i)]),
                    float(delta_ppos[int(i)]),
                    float(ppos_bulk[int(i)]),
                    float(ppos_nano[int(i)]),
                    float(stats["mean_bulk"][int(i)]),
                    float(stats["mean_nano"][int(i)]),
                    float(stats["var_bulk"][int(i)]),
                    float(stats["var_nano"][int(i)]),
                ]
            )


def _summarize_labels(labels: Optional[np.ndarray]) -> Dict[str, object]:
    if labels is None:
        return {"present": False}
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    unique = unique[order]
    counts = counts[order]
    return {
        "present": True,
        "n_unique": int(unique.size),
        "top_counts": [{"label": str(unique[i]), "count": int(counts[i])} for i in range(min(10, unique.size))],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Bulk vs nanopore domain shift report (HDF5).")
    ap.add_argument("--bulk_h5", required=True)
    ap.add_argument("--nano_h5", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarization / P(x>=threshold).")
    ap.add_argument("--chunk_size", type=int, default=5000, help="Number of features per streamed chunk.")
    ap.add_argument("--topk", type=int, default=5000, help="Top-K features by |ΔP(x>=threshold)|.")
    ap.add_argument(
        "--progress",
        choices=["auto", "tqdm", "print", "none"],
        default="auto",
        help="Progress reporting mode for streamed loops.",
    )
    ap.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="For --progress=print, print every N chunks (also used as fallback when tqdm is unavailable).",
    )
    ap.add_argument(
        "--backend",
        choices=["auto", "cpu", "torch"],
        default="auto",
        help="Compute backend. 'auto' will use torch on GPU if available, else CPU.",
    )
    ap.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string when using GPU (default: cuda:0).",
    )
    args = ap.parse_args()

    progress = ProgressConfig(mode=str(args.progress), log_every=int(args.log_every))
    compute = ComputeConfig(backend=str(args.backend), device=str(args.device))

    bulk_view = _open_h5_view(args.bulk_h5)
    nano_view = _open_h5_view(args.nano_h5)

    bulk_names = _read_feature_names(bulk_view)
    nano_names = _read_feature_names(nano_view)

    feature_alignment = {
        "same_length": bool(bulk_names.shape == nano_names.shape),
        "exact_match": bool(bulk_names.shape == nano_names.shape and np.all(bulk_names == nano_names)),
        "head_bulk": [str(x) for x in bulk_names[:5]],
        "head_nano": [str(x) for x in nano_names[:5]],
    }

    bulk_labels = _read_labels(bulk_view)
    nano_labels = _read_labels(nano_view)

    # Sample-mean distribution (mean methylation across all CpGs per sample)
    backend_sel, device_sel = _select_compute(compute)
    print(f"[compute] backend={backend_sel} device={device_sel}", file=sys.stderr)

    bulk_sample_means = _sample_means_streaming(
        bulk_view,
        chunk_size=int(args.chunk_size),
        progress=progress,
        desc="sample-means bulk",
        compute=compute,
    )
    nano_sample_means = _sample_means_streaming(
        nano_view,
        chunk_size=int(args.chunk_size),
        progress=progress,
        desc="sample-means nanopore",
        compute=compute,
    )
    ks_d = _ks_2samp_statistic(bulk_sample_means, nano_sample_means)

    stats = _compute_feature_stats(
        bulk_view=bulk_view,
        nano_view=nano_view,
        threshold=float(args.threshold),
        chunk_size=int(args.chunk_size),
        progress=progress,
        compute=compute,
    )

    delta_ppos = stats["ppos_nano"] - stats["ppos_bulk"]
    abs_delta_ppos = np.abs(delta_ppos)

    summary = {
        "bulk_h5": bulk_view.path,
        "nano_h5": nano_view.path,
        "bulk_keys": {"data": bulk_view.data_key, "feature_names": bulk_view.feature_names_key, "labels": bulk_view.labels_key},
        "nano_keys": {"data": nano_view.data_key, "feature_names": nano_view.feature_names_key, "labels": nano_view.labels_key},
        "feature_alignment": feature_alignment,
        "labels_bulk": _summarize_labels(bulk_labels),
        "labels_nano": _summarize_labels(nano_labels),
        "threshold": float(args.threshold),
        "sample_mean": {
            "bulk": {
                "n": int(bulk_sample_means.size),
                "mean": float(np.mean(bulk_sample_means)),
                "std": float(np.std(bulk_sample_means)),
                "min": float(np.min(bulk_sample_means)),
                "max": float(np.max(bulk_sample_means)),
            },
            "nano": {
                "n": int(nano_sample_means.size),
                "mean": float(np.mean(nano_sample_means)),
                "std": float(np.std(nano_sample_means)),
                "min": float(np.min(nano_sample_means)),
                "max": float(np.max(nano_sample_means)),
            },
            "ks_D": float(ks_d),
        },
        "feature_drift": {
            "n_features": int(abs_delta_ppos.size),
            "abs_delta_ppos": {
                "mean": float(np.mean(abs_delta_ppos)),
                "p50": float(np.quantile(abs_delta_ppos, 0.50)),
                "p90": float(np.quantile(abs_delta_ppos, 0.90)),
                "p99": float(np.quantile(abs_delta_ppos, 0.99)),
                "max": float(np.max(abs_delta_ppos)),
            },
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out_dir, "sample_means.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "sample_index", "mean_methylation"])
        for i, v in enumerate(bulk_sample_means):
            w.writerow(["bulk", i, float(v)])
        for i, v in enumerate(nano_sample_means):
            w.writerow(["nanopore", i, float(v)])

    # Top drifted CpGs by |ΔP(x>=threshold)|
    _write_topk_csv(
        out_csv=os.path.join(args.out_dir, f"top{int(args.topk)}_drift_by_ppos.csv"),
        feature_names=[str(x) for x in bulk_names],
        stats=stats,
        topk=int(args.topk),
    )

    # Human-readable one-pager
    txt_path = os.path.join(args.out_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write("Bulk vs Nanopore Domain Shift Report\n")
        f.write("=================================\n\n")
        f.write(f"bulk_h5: {bulk_view.path}\n")
        f.write(f"nano_h5: {nano_view.path}\n\n")
        f.write("Feature alignment\n")
        f.write(f"- same_length: {feature_alignment['same_length']}\n")
        f.write(f"- exact_match: {feature_alignment['exact_match']}\n\n")
        f.write("Sample mean methylation (mean over all CpGs per sample)\n")
        f.write(
            f"- bulk: mean={summary['sample_mean']['bulk']['mean']:.6f} std={summary['sample_mean']['bulk']['std']:.6f} "
            f"min={summary['sample_mean']['bulk']['min']:.6f} max={summary['sample_mean']['bulk']['max']:.6f} n={summary['sample_mean']['bulk']['n']}\n"
        )
        f.write(
            f"- nano: mean={summary['sample_mean']['nano']['mean']:.6f} std={summary['sample_mean']['nano']['std']:.6f} "
            f"min={summary['sample_mean']['nano']['min']:.6f} max={summary['sample_mean']['nano']['max']:.6f} n={summary['sample_mean']['nano']['n']}\n"
        )
        f.write(f"- KS D: {summary['sample_mean']['ks_D']:.6f}\n\n")
        f.write(f"Per-feature drift using P(x>=threshold), threshold={float(args.threshold):.3f}\n")
        f.write(
            f"- |ΔP| mean={summary['feature_drift']['abs_delta_ppos']['mean']:.6f} "
            f"p50={summary['feature_drift']['abs_delta_ppos']['p50']:.6f} "
            f"p90={summary['feature_drift']['abs_delta_ppos']['p90']:.6f} "
            f"p99={summary['feature_drift']['abs_delta_ppos']['p99']:.6f} "
            f"max={summary['feature_drift']['abs_delta_ppos']['max']:.6f}\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
