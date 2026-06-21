#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RankCode-ICL D2 selective-screening analysis from qvec_main_clean_final.jsonl.

Purpose
-------
Compares flat TabPFN abstention signals against cumulative q-vector signals at the
same rejection fractions. The main question is whether cumulative threshold/RankCode
signals remove severe misranking cases better than flat softmax margin/confidence.

Default input
-------------
/content/drive/MyDrive/rankcode/colab_run/results/qvec_main_clean_final.jsonl

Default output directory
------------------------
/content/drive/MyDrive/rankcode/colab_run/results/d2_selective_screening

Runtime
-------
CPU is enough. GPU is not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    wilcoxon = None
    SCIPY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# Faster JSON parser if present; otherwise use stdlib.
try:
    import orjson  # type: ignore

    def loads_json(s: str) -> Dict[str, Any]:
        return orjson.loads(s)
except Exception:
    try:
        import ujson  # type: ignore

        def loads_json(s: str) -> Dict[str, Any]:
            return ujson.loads(s)
    except Exception:
        def loads_json(s: str) -> Dict[str, Any]:
            return json.loads(s)


DEFAULT_INPUT = "/content/drive/MyDrive/rankcode/colab_run/results/qvec_main_clean_final.jsonl"
DEFAULT_OUTDIR = "/content/drive/MyDrive/rankcode/colab_run/results/d2_selective_screening"

FLAT_MODEL = "TabPFN_flat"
THRESHOLD_MODEL = "TabPFN_vanilla_threshold"
RANKCODE_MODEL = "TabPFN_RankCode"

MODEL_TO_PRIMARY_METHOD = {
    FLAT_MODEL: "flat_margin",
    THRESHOLD_MODEL: "threshold_cumul",
    RANKCODE_MODEL: "rankcode_cumul",
}

REJECT_GRID_DEFAULT = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)


def maybe_mount_drive() -> None:
    """Mount Google Drive when running inside Colab; harmless otherwise."""
    if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive")
        except Exception as e:
            print(f"[WARN] Could not mount Drive automatically: {e}")


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def to_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def severe_from_record(r: Dict[str, Any]) -> int:
    """Severe misranking: |y_hat - y| >= ceil(K/2), using encoded ordinal positions."""
    k = to_int(r.get("K", 0), 0)
    y = to_int(r.get("y_true_enc", r.get("y_true", 0)), 0)
    yp = to_int(r.get("y_pred_enc", r.get("y_pred", 0)), 0)
    if k <= 0:
        return int(abs(yp - y) >= 2)
    return int(abs(yp - y) >= int(math.ceil(k / 2.0)))


def nonadj_from_record(r: Dict[str, Any]) -> int:
    y = to_int(r.get("y_true_enc", r.get("y_true", 0)), 0)
    yp = to_int(r.get("y_pred_enc", r.get("y_pred", 0)), 0)
    return int(abs(yp - y) >= 2)


def cum_to_class_probs(q: Iterable[Any]) -> np.ndarray:
    """Convert cumulative q_vec P(y>k) into class probabilities."""
    q_arr = np.clip(np.asarray(list(q), dtype=float), 0.0, 1.0)
    if q_arr.size == 0:
        return np.ones(1, dtype=float)
    k = q_arr.size + 1
    p = np.zeros(k, dtype=float)
    p[0] = 1.0 - q_arr[0]
    if k > 2:
        p[1:k-1] = q_arr[:-1] - q_arr[1:]
    p[k-1] = q_arr[-1]
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0 or not np.isfinite(s):
        return np.ones(k, dtype=float) / float(k)
    return p / s


def qvec_scores(r: Dict[str, Any]) -> Dict[str, float]:
    """
    Cumulative uncertainty scores.

    High score = reject earlier.
    Primary score = monotonicity violation + repaired class-margin uncertainty.
    Secondary scores are exported for robustness checks.
    """
    q_raw = r.get("q_vec_raw", None)
    q_rep = r.get("q_vec_repaired", q_raw)
    if q_raw is None or q_rep is None:
        return {
            "cumul": float("nan"),
            "cumul_margin": float("nan"),
            "cumul_conf": float("nan"),
            "cumul_violation": float("nan"),
        }

    raw = np.asarray(q_raw, dtype=float)
    if raw.size >= 2:
        # Cumulative probabilities should be non-increasing; upward jumps are violations.
        violation = float(np.sum(np.clip(raw[1:] - raw[:-1], 0.0, None)))
    else:
        violation = 0.0

    p = cum_to_class_probs(q_rep)
    if p.size >= 2:
        top2 = np.sort(p)[-2:]
        class_margin = float(top2[-1] - top2[-2])
    else:
        class_margin = float(p[0])
    max_prob = float(np.max(p))

    return {
        "cumul": float(violation + (1.0 - class_margin)),
        "cumul_margin": float(1.0 - class_margin),
        "cumul_conf": float(1.0 - max_prob),
        "cumul_violation": float(violation),
    }


class MethodDatasetStore:
    """Memory-efficient per-method, per-dataset score/error storage."""

    def __init__(self) -> None:
        self.scores: Dict[str, Dict[str, array]] = defaultdict(lambda: defaultdict(lambda: array("f")))
        self.severe: Dict[str, Dict[str, bytearray]] = defaultdict(lambda: defaultdict(bytearray))
        self.nonadj: Dict[str, Dict[str, bytearray]] = defaultdict(lambda: defaultdict(bytearray))
        self.raw_counts: Counter = Counter()
        self.model_counts: Counter = Counter()
        self.dataset_counts: Counter = Counter()

    def add(self, method: str, dataset: str, score: float, severe: int, nonadj: int) -> None:
        if not np.isfinite(score):
            return
        self.scores[method][dataset].append(float(score))
        self.severe[method][dataset].append(int(severe))
        self.nonadj[method][dataset].append(int(nonadj))
        self.raw_counts[(method, dataset)] += 1

    def methods(self) -> List[str]:
        return sorted(self.scores.keys())

    def datasets_for(self, method: str) -> List[str]:
        return sorted(self.scores.get(method, {}).keys())


def read_qvec_stream(path: Path, block: str = "main") -> MethodDatasetStore:
    t0 = time.time()
    store = MethodDatasetStore()
    total = 0
    kept = 0
    skipped_block = 0
    bad = 0
    unsupported = 0

    print("\nReading clean qvec JSONL:", path)
    print("This is CPU-only and may take several minutes for ~2 GB.")

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            s = line.strip()
            if not s:
                bad += 1
                continue
            try:
                r = loads_json(s)
            except Exception:
                bad += 1
                continue

            if str(r.get("block", "main")) != block:
                skipped_block += 1
                continue

            model = str(r.get("model", ""))
            ds = str(r.get("dataset", ""))
            sev = severe_from_record(r)
            nonadj = nonadj_from_record(r)
            store.model_counts[model] += 1
            store.dataset_counts[ds] += 1

            if model == FLAT_MODEL:
                margin = to_float(r.get("margin", 0.0), 0.0)
                max_prob = to_float(r.get("max_prob", 0.0), 0.0)
                store.add("flat_margin", ds, 1.0 - margin, sev, nonadj)
                store.add("flat_conf", ds, 1.0 - max_prob, sev, nonadj)
                kept += 1
            elif model == THRESHOLD_MODEL:
                sc = qvec_scores(r)
                store.add("threshold_cumul", ds, sc["cumul"], sev, nonadj)
                store.add("threshold_cumul_margin", ds, sc["cumul_margin"], sev, nonadj)
                store.add("threshold_cumul_conf", ds, sc["cumul_conf"], sev, nonadj)
                # violation alone is usually not enough, but exportable as diagnostic.
                store.add("threshold_violation", ds, sc["cumul_violation"], sev, nonadj)
                kept += 1
            elif model == RANKCODE_MODEL:
                sc = qvec_scores(r)
                store.add("rankcode_cumul", ds, sc["cumul"], sev, nonadj)
                store.add("rankcode_cumul_margin", ds, sc["cumul_margin"], sev, nonadj)
                store.add("rankcode_cumul_conf", ds, sc["cumul_conf"], sev, nonadj)
                store.add("rankcode_violation", ds, sc["cumul_violation"], sev, nonadj)
                kept += 1
            else:
                unsupported += 1

            if total % 1_000_000 == 0:
                elapsed = (time.time() - t0) / 60.0
                print(f"line={total:,} | parsed_model_rows={kept:,} | bad={bad:,} | elapsed={elapsed:.1f} min")

    elapsed = (time.time() - t0) / 60.0
    print("\nRead summary")
    print("- total lines       :", f"{total:,}")
    print("- usable model rows :", f"{kept:,}")
    print("- bad rows skipped  :", f"{bad:,}")
    print("- block skipped     :", f"{skipped_block:,}")
    print("- unsupported model :", f"{unsupported:,}")
    print("- elapsed min       :", round(elapsed, 2))
    print("- methods collected :", ", ".join(store.methods()))
    return store


def exact_rejection_curve(scores_arr: array, err_bytes: bytearray, reject_grid: Tuple[float, ...]) -> List[Dict[str, Any]]:
    scores = np.frombuffer(scores_arr, dtype=np.float32)
    errors = np.frombuffer(bytes(err_bytes), dtype=np.uint8)
    n = int(scores.shape[0])
    if n == 0:
        return []
    # Higher score = more uncertain/risky. Sort ascending so kept examples are prefix after rejecting high scores.
    order = np.argsort(scores, kind="mergesort")
    err_sorted = errors[order]
    csum = np.cumsum(err_sorted, dtype=np.int64)
    total_err = int(csum[-1])
    out: List[Dict[str, Any]] = []
    for frac in reject_grid:
        frac = float(frac)
        n_reject = int(round(frac * n))
        n_reject = max(0, min(n - 1 if n > 1 else 0, n_reject))
        n_keep = n - n_reject
        kept_err = int(csum[n_keep - 1]) if n_keep > 0 else 0
        smr = float(kept_err / n_keep) if n_keep > 0 else float("nan")
        out.append({
            "reject_frac": frac,
            "coverage": float(n_keep / n),
            "n_total": n,
            "n_kept": n_keep,
            "n_rejected": n_reject,
            "errors_total": total_err,
            "errors_kept": kept_err,
            "error_rate": smr,
        })
    return out


def compute_curves(store: MethodDatasetStore, reject_grid: Tuple[float, ...], error_kind: str = "severe") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    err_source = store.severe if error_kind == "severe" else store.nonadj
    for method in store.methods():
        for ds in store.datasets_for(method):
            curves = exact_rejection_curve(store.scores[method][ds], err_source[method][ds], reject_grid)
            for c in curves:
                rows.append({
                    "error_kind": error_kind,
                    "method": method,
                    "dataset": ds,
                    **c,
                })
    return pd.DataFrame(rows)


def pooled_curve(store: MethodDatasetStore, reject_grid: Tuple[float, ...], error_kind: str = "severe") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    err_source = store.severe if error_kind == "severe" else store.nonadj
    for method in store.methods():
        all_scores = array("f")
        all_errors = bytearray()
        for ds in store.datasets_for(method):
            all_scores.extend(store.scores[method][ds])
            all_errors.extend(err_source[method][ds])
        for c in exact_rejection_curve(all_scores, all_errors, reject_grid):
            rows.append({"error_kind": error_kind, "method": method, "dataset": "__POOLED__", **c})
    return pd.DataFrame(rows)


def safe_wilcoxon(deltas: np.ndarray, alternative: str = "greater") -> float:
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return float("nan")
    if np.allclose(deltas, 0):
        return 1.0
    if not SCIPY_AVAILABLE:
        return float("nan")
    try:
        return float(wilcoxon(deltas, alternative=alternative, zero_method="wilcox").pvalue)  # type: ignore[union-attr]
    except Exception:
        return float("nan")


def paired_gate(dataset_curves: pd.DataFrame, baseline: str, candidates: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    base = dataset_curves[dataset_curves["method"] == baseline]
    for cand in candidates:
        cdf = dataset_curves[dataset_curves["method"] == cand]
        merged = base.merge(
            cdf,
            on=["error_kind", "dataset", "reject_frac"],
            suffixes=("_flat", "_cand"),
            how="inner",
        )
        if merged.empty:
            continue
        for (err_kind, frac), g in merged.groupby(["error_kind", "reject_frac"], sort=True):
            delta = g["error_rate_flat"].to_numpy(float) - g["error_rate_cand"].to_numpy(float)
            improved = int(np.sum(delta > 1e-12))
            worsened = int(np.sum(delta < -1e-12))
            tied = int(np.sum(np.abs(delta) <= 1e-12))
            p_greater = safe_wilcoxon(delta, alternative="greater")
            p_two = safe_wilcoxon(delta, alternative="two-sided")
            rows.append({
                "error_kind": err_kind,
                "baseline": baseline,
                "candidate": cand,
                "reject_frac": float(frac),
                "n_datasets": int(len(g)),
                "flat_mean": float(g["error_rate_flat"].mean()),
                "candidate_mean": float(g["error_rate_cand"].mean()),
                "delta_mean_flat_minus_candidate": float(np.mean(delta)),
                "delta_median_flat_minus_candidate": float(np.median(delta)),
                "delta_std": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                "improved_datasets": improved,
                "worsened_datasets": worsened,
                "tied_datasets": tied,
                "wilcoxon_p_greater": p_greater,
                "wilcoxon_p_two_sided": p_two,
                "decision_one_sided": "GO" if (np.mean(delta) > 0 and improved > worsened and np.isfinite(p_greater) and p_greater < 0.05) else "STOP",
                "decision_two_sided": "GO" if (np.mean(delta) > 0 and improved > worsened and np.isfinite(p_two) and p_two < 0.05) else "STOP",
            })
    return pd.DataFrame(rows)


def plot_pooled_curves(pooled: pd.DataFrame, outdir: Path, error_kind: str = "severe") -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    sub = pooled[pooled["error_kind"] == error_kind].copy()
    if sub.empty:
        return
    # Keep the main methods readable.
    preferred = [
        "flat_margin",
        "flat_conf",
        "threshold_cumul",
        "threshold_cumul_margin",
        "rankcode_cumul",
        "rankcode_cumul_margin",
    ]
    sub = sub[sub["method"].isin(preferred)]
    if sub.empty:
        return

    plt.figure(figsize=(8, 5))
    for method, g in sub.groupby("method", sort=False):
        g = g.sort_values("reject_frac")
        plt.plot(g["reject_frac"], g["error_rate"], marker="o", label=method)
    plt.xlabel("Rejected fraction")
    ylabel = "Severe misranking rate among kept samples" if error_kind == "severe" else "Non-adjacent error rate among kept samples"
    plt.ylabel(ylabel)
    plt.title("D2 selective screening: pooled risk-coverage curve")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = outdir / f"D2_pooled_{error_kind}_curve.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved plot:", out)


def write_manifest(outdir: Path, args: argparse.Namespace, store: MethodDatasetStore,
                   dataset_curves: pd.DataFrame, gate: pd.DataFrame, elapsed_min: float) -> None:
    manifest = {
        "input": str(args.input),
        "outdir": str(outdir),
        "elapsed_min": elapsed_min,
        "reject_grid": list(args.reject_grid),
        "methods": store.methods(),
        "model_counts": dict(store.model_counts),
        "dataset_count": int(len(set(store.dataset_counts.keys()))),
        "dataset_curve_rows": int(len(dataset_curves)),
        "gate_rows": int(len(gate)),
        "scipy_available": SCIPY_AVAILABLE,
        "matplotlib_available": MATPLOTLIB_AVAILABLE,
    }
    with (outdir / "D2_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="RankCode q_vec D2 selective-screening analysis")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to qvec_main_clean_final.jsonl")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--block", default="main", help="Block name to analyze, default: main")
    parser.add_argument(
        "--reject-grid",
        type=float,
        nargs="+",
        default=list(REJECT_GRID_DEFAULT),
        help="Rejected fractions, e.g. --reject-grid 0 0.05 0.1 0.2 0.3",
    )
    parser.add_argument("--no-nonadjacent", action="store_true", help="Skip non-adjacent robustness curves")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plots")
    args = parser.parse_args()

    maybe_mount_drive()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Normalize grid and guarantee 0.0 is included.
    grid = tuple(sorted(set(float(x) for x in args.reject_grid)))
    if 0.0 not in grid:
        grid = (0.0,) + grid
    args.reject_grid = grid

    print("=" * 100)
    print("RANKCODE D2 SELECTIVE-SCREENING ANALYSIS")
    print("=" * 100)
    print("Input :", input_path)
    print("Outdir:", outdir)
    print("Input size MB:", round(input_path.stat().st_size / 1024**2, 2))
    print("Reject grid:", grid)
    print("Runtime: CPU-only. GPU is not used.")

    t0 = time.time()
    store = read_qvec_stream(input_path, block=args.block)

    print("\nComputing dataset-level exact rejection curves...")
    d_sev = compute_curves(store, grid, error_kind="severe")
    p_sev = pooled_curve(store, grid, error_kind="severe")

    frames_dataset = [d_sev]
    frames_pooled = [p_sev]
    if not args.no_nonadjacent:
        d_non = compute_curves(store, grid, error_kind="nonadjacent")
        p_non = pooled_curve(store, grid, error_kind="nonadjacent")
        frames_dataset.append(d_non)
        frames_pooled.append(p_non)

    dataset_curves = pd.concat(frames_dataset, ignore_index=True)
    pooled_curves = pd.concat(frames_pooled, ignore_index=True)

    candidates = [
        "threshold_cumul",
        "threshold_cumul_margin",
        "threshold_cumul_conf",
        "rankcode_cumul",
        "rankcode_cumul_margin",
        "rankcode_cumul_conf",
    ]
    candidates = [c for c in candidates if c in store.methods()]

    gate = paired_gate(dataset_curves, baseline="flat_margin", candidates=candidates)

    # Secondary comparison against flat confidence, useful for robustness.
    gate_conf = pd.DataFrame()
    if "flat_conf" in store.methods():
        gate_conf = paired_gate(dataset_curves, baseline="flat_conf", candidates=candidates)

    # Save outputs.
    dataset_path = outdir / "D2_dataset_level_curves.csv"
    pooled_path = outdir / "D2_pooled_curves.csv"
    gate_path = outdir / "D2_paired_gate_vs_flat_margin.csv"
    gate_conf_path = outdir / "D2_paired_gate_vs_flat_conf.csv"

    dataset_curves.to_csv(dataset_path, index=False)
    pooled_curves.to_csv(pooled_path, index=False)
    gate.to_csv(gate_path, index=False)
    if not gate_conf.empty:
        gate_conf.to_csv(gate_conf_path, index=False)

    if not args.no_plots:
        plot_pooled_curves(pooled_curves, outdir, "severe")
        if not args.no_nonadjacent:
            plot_pooled_curves(pooled_curves, outdir, "nonadjacent")

    elapsed_min = (time.time() - t0) / 60.0
    write_manifest(outdir, args, store, dataset_curves, gate, elapsed_min)

    # Compact console summary for the paper decision.
    print("\n" + "=" * 100)
    print("D2 MAIN GATE - severe misranking, baseline = flat_margin")
    print("=" * 100)
    sev_gate = gate[gate["error_kind"] == "severe"].copy()
    show_cols = [
        "candidate", "reject_frac", "flat_mean", "candidate_mean",
        "delta_mean_flat_minus_candidate", "improved_datasets", "worsened_datasets",
        "wilcoxon_p_greater", "wilcoxon_p_two_sided", "decision_one_sided", "decision_two_sided",
    ]
    if sev_gate.empty:
        print("No severe gate rows produced. Check method names in qvec file.")
    else:
        # Print only core reject fractions of interest first.
        core = sev_gate[sev_gate["reject_frac"].isin([0.10, 0.20, 0.30])]
        if core.empty:
            core = sev_gate
        with pd.option_context("display.max_rows", 100, "display.max_columns", 50, "display.width", 180):
            print(core[show_cols].to_string(index=False))

    print("\nOutputs saved:")
    print("-", dataset_path)
    print("-", pooled_path)
    print("-", gate_path)
    if not gate_conf.empty:
        print("-", gate_conf_path)
    print("-", outdir / "D2_manifest.json")
    print("\nElapsed min:", round(elapsed_min, 2))
    print("DONE.")


if __name__ == "__main__":
    main()
