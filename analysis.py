#!/usr/bin/env python3
"""
RankCode-ICL reproducible analysis pipeline.

This script reproduces the downstream tables and statistical tests from the
released experiment logs in ./data. It writes CSV files to ./results. The D2
selective-screening outputs are distributed as derived CSV files in ./results.

Usage:
    pip install -r requirements.txt
    python analysis.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Optional: scikit-posthocs for the Nemenyi matrix (falls back gracefully)
try:
    import scikit_posthocs as sp
    HAVE_SP = True
except Exception:
    HAVE_SP = False

RNG_SEED = 42
N_BOOT = 20000

DATA = "data"
OUT = "results"
os.makedirs(OUT, exist_ok=True)

# Metrics that are averaged / reported throughout
METRICS = [
    "qwk", "mae", "severe_misranking_rate",
    "adjacent_accuracy", "mean_extreme_recall",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load(name):
    """Load a JSONL block and keep only successful runs."""
    path = os.path.join(DATA, name)
    df = pd.read_json(path, lines=True)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    return df


def bootstrap_mean_ci(delta, n_boot=N_BOOT, seed=RNG_SEED, alpha=0.05):
    """Percentile bootstrap CI for the MEAN of a paired difference vector."""
    rng = np.random.default_rng(seed)
    delta = np.asarray(delta, dtype=float)
    boot = np.array([
        np.mean(rng.choice(delta, size=len(delta), replace=True))
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def nemenyi_cd(k, n, q_alpha=3.354):
    """Nemenyi critical difference at alpha=0.05 (q_alpha for k models)."""
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))


def banner(msg):
    print("\n" + "=" * 74)
    print(msg)
    print("=" * 74)


# ==========================================================================
# BLOCK 1 - MAIN BENCHMARK
# ==========================================================================
def analyse_main():
    banner("BLOCK 1 - Main benchmark (paper Table 3, Figure 2)")
    df = load("block1_main_fixed.jsonl")

    # ---- Table 3 : per-model means over all 46 datasets x 30 splits --------
    rows = []
    for m, g in df.groupby("model"):
        rows.append({
            "model": m,
            "QWK_mean": g["qwk"].mean(), "QWK_std": g["qwk"].std(),
            "MAE_mean": g["mae"].mean(), "MAE_std": g["mae"].std(),
            "SMR": g["severe_misranking_rate"].mean(),
            "AdjacentAcc": g["adjacent_accuracy"].mean(),
            "ExtremeRecall": g["mean_extreme_recall"].mean(),
        })
    table3 = pd.DataFrame(rows).sort_values("QWK_mean", ascending=False)

    # delta SMR vs the matching flat foundation model
    flat_smr = {
        "TabPFN": table3.loc[table3.model == "TabPFN_flat", "SMR"].values[0],
        "TabICL": table3.loc[table3.model == "TabICL_flat", "SMR"].values[0],
    }

    def dsmr(row):
        if "TabPFN" in row.model and row.model != "TabPFN_flat":
            return flat_smr["TabPFN"] - row.SMR
        if "TabICL" in row.model and row.model != "TabICL_flat":
            return flat_smr["TabICL"] - row.SMR
        return np.nan

    table3["dSMR_vs_flat"] = table3.apply(dsmr, axis=1)
    table3.to_csv(os.path.join(OUT, "table3_main_benchmark.csv"), index=False)
    print(table3.round(4).to_string(index=False))

    # ---- Dataset-level Friedman + Nemenyi (paper Figure 2) ----------------
    ds = df.groupby(["dataset", "model"])["qwk"].mean().reset_index()
    pivot = ds.pivot_table(index="dataset", columns="model",
                           values="qwk").dropna()
    chi, p = stats.friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    k, n = pivot.shape[1], pivot.shape[0]
    cd = nemenyi_cd(k, n)
    ranks = pivot.rank(axis=1, ascending=False).mean().sort_values()

    fr = pd.DataFrame({"model": ranks.index, "mean_rank": ranks.values})
    fr.attrs = {}
    fr_meta = pd.DataFrame([{
        "friedman_chi2": chi, "friedman_p": p,
        "n_datasets": n, "n_models": k, "nemenyi_CD_0.05": cd,
    }])
    fr.to_csv(os.path.join(OUT, "main_friedman_nemenyi.csv"), index=False)
    fr_meta.to_csv(os.path.join(OUT, "main_friedman_meta.csv"), index=False)
    print(f"\nDataset-level Friedman: chi2={chi:.2f}, p={p:.3e}, "
          f"n={n} datasets, CD={cd:.3f}")
    print("Mean ranks (QWK, lower=better):")
    print(fr.round(3).to_string(index=False))

    # ---- Dataset-level pairwise Wilcoxon on SMR (paper Sec 5, honest note) -
    smr_ds = (df.groupby(["dataset", "model"])["severe_misranking_rate"]
                .mean().reset_index()
                .pivot_table(index="dataset", columns="model",
                             values="severe_misranking_rate"))
    qwk_ds = pivot
    rows = []
    for base, rc in [("TabPFN_flat", "TabPFN_RankCode"),
                     ("TabICL_flat", "TabICL_RankCode")]:
        d = smr_ds[[base, rc]].dropna()
        delta = (d[base] - d[rc]).values          # +ve => RankCode better
        W, pw = stats.wilcoxon(d[base], d[rc])
        lo, hi = bootstrap_mean_ci(delta)
        t_p = stats.ttest_rel(d[base], d[rc]).pvalue
        qd = qwk_ds[[base, rc]].dropna()
        rows.append({
            "comparison": f"{base} vs {rc}",
            "n_datasets": len(d),
            "mean_dSMR": delta.mean(),
            "median_dSMR": np.median(delta),
            "n_improved": int((delta > 0).sum()),
            "n_worsened": int((delta < 0).sum()),
            "n_tied": int((delta == 0).sum()),
            "wilcoxon_W": W, "wilcoxon_p": pw,
            "mean_ci_lo": lo, "mean_ci_hi": hi,
            "paired_t_p": t_p,
            "mean_dQWK": (qd[rc] - qd[base]).mean(),
        })
    pw_df = pd.DataFrame(rows)
    pw_df.to_csv(os.path.join(OUT, "main_pairwise_smr_wilcoxon.csv"),
                 index=False)
    print("\nDataset-level pairwise SMR test (note the median ~ 0):")
    print(pw_df.round(4).to_string(index=False))

    # ---- Table 1 : distribution of K -------------------------------------
    if "K" in df.columns:
        kd = df.groupby("dataset")["K"].first()
        kdist = kd.value_counts().sort_index()
        kdf = pd.DataFrame({"K": kdist.index, "n_datasets": kdist.values})
        kdf.to_csv(os.path.join(OUT, "table1_dataset_K_distribution.csv"),
                   index=False)
        nrange = df.groupby("dataset")["n"].first() if "n" in df.columns else None
        print("\nK distribution:", dict(zip(kdf.K, kdf.n_datasets)))
        if nrange is not None:
            print(f"Sample size: min={nrange.min():.0f}, "
                  f"median={nrange.median():.0f}, max={nrange.max():.0f}")

    return table3


# ==========================================================================
# BLOCK 2 - FORMULATION ABLATION (paper Table 4)
# ==========================================================================
def analyse_formulation():
    banner("BLOCK 2 - Formulation ablation (paper Table 4)")
    df = load("block2_formulation_ablation.jsonl")
    rows = []
    for m, g in df.groupby("model"):
        rows.append({
            "variant": m,
            "QWK": g["qwk"].mean(),
            "MAE": g["mae"].mean(),
            "SMR": g["severe_misranking_rate"].mean(),
            "ExtremeRecall": g["mean_extreme_recall"].mean(),
        })
    t = pd.DataFrame(rows).sort_values("QWK", ascending=False)
    # deltas vs flat TabPFN
    base_smr = t.loc[t.variant.str.contains("flat", case=False)
                     & t.variant.str.contains("PFN", case=False), "SMR"]
    base_er = t.loc[t.variant.str.contains("flat", case=False)
                    & t.variant.str.contains("PFN", case=False), "ExtremeRecall"]
    if len(base_smr):
        t["dSMR_vs_flat"] = base_smr.values[0] - t["SMR"]
        t["dExtRec_vs_flat"] = t["ExtremeRecall"] - base_er.values[0]
    t.to_csv(os.path.join(OUT, "table4_formulation_ablation.csv"), index=False)
    print(t.round(4).to_string(index=False))
    return t


# ==========================================================================
# BLOCK 3 - CONTEXT ABLATION (paper Figure 4)
# ==========================================================================
def analyse_context():
    banner("BLOCK 3 - Context-design ablation (paper Figure 4)")
    df = load("block3_context_ablation.jsonl")
    rows = []
    for (mdl, b), g in df.groupby(["model", "budget"]):
        rows.append({
            "strategy": mdl, "budget": b,
            "QWK": g["qwk"].mean(),
            "SMR": g["severe_misranking_rate"].mean(),
            "ExtremeRecall": g["mean_extreme_recall"].mean()
            if "mean_extreme_recall" in g.columns else np.nan,
        })
    t = pd.DataFrame(rows).sort_values(["budget", "strategy"])
    t.to_csv(os.path.join(OUT, "context_ablation_budget_sweep.csv"),
             index=False)
    # print compact QWK pivot
    print("QWK by strategy x budget:")
    print(t.pivot_table(index="strategy", columns="budget",
                        values="QWK").round(4).to_string())
    return t


# ==========================================================================
# BLOCK 4 - IMBALANCE STRESS (paper Figure 5, seed-level Wilcoxon)
# ==========================================================================
def analyse_imbalance():
    banner("BLOCK 4 - Imbalance stress (paper Figure 5)")
    df = load("block4_imbalance_stress.jsonl")

    # per-scenario SMR means
    piv = df.pivot_table(index="model", columns="scenario",
                         values="severe_misranking_rate", aggfunc="mean")
    piv.to_csv(os.path.join(OUT, "imbalance_smr_by_scenario.csv"))
    print("SMR by model x scenario:")
    print(piv.round(4).to_string())

    # seed-level paired Wilcoxon at each scenario
    rows = []
    for sc in df["scenario"].unique():
        sub = df[df["scenario"] == sc]
        sp_piv = sub.pivot_table(index="seed", columns="model",
                                 values="severe_misranking_rate")
        for base, rc in [("TabPFN_flat", "TabPFN_RankCode"),
                         ("TabICL_flat", "TabICL_RankCode")]:
            if base in sp_piv and rc in sp_piv:
                d = sp_piv[[base, rc]].dropna()
                if len(d) > 5 and (d[base] - d[rc]).abs().sum() > 0:
                    W, pw = stats.wilcoxon(d[base], d[rc])
                    delta = (d[base] - d[rc])
                    rel = (delta.mean() / d[base].mean() * 100
                           if d[base].mean() > 0 else 0.0)
                    rows.append({
                        "scenario": sc,
                        "comparison": f"{base} vs {rc}",
                        "n_seeds": len(d),
                        "flat_SMR": d[base].mean(),
                        "rankcode_SMR": d[rc].mean(),
                        "mean_dSMR": delta.mean(),
                        "rel_reduction_pct": rel,
                        "wilcoxon_p": pw,
                    })
    wt = pd.DataFrame(rows)
    wt.to_csv(os.path.join(OUT, "imbalance_seed_wilcoxon.csv"), index=False)
    print("\nSeed-level paired Wilcoxon (key scenarios):")
    key = wt[wt.scenario.isin(["severe_ir30", "moderate_ir10"])]
    print(key.round(4).to_string(index=False))

    # monotone repair summary
    rc = df[df["model"].str.contains("RankCode")]
    if "monotone_violations_before_repair" in rc.columns:
        ms = pd.DataFrame([{
            "mean_violations_before_repair":
                rc["monotone_violations_before_repair"].mean(),
            "mean_violation_rate_after_repair":
                rc["monotone_violation_rate"].mean()
                if "monotone_violation_rate" in rc.columns else np.nan,
        }])
        ms.to_csv(os.path.join(OUT, "monotone_repair_summary.csv"),
                  index=False)
        print(f"\nMonotone repair: "
              f"{ms.mean_violations_before_repair.values[0]:.1f} violations "
              f"corrected per configuration (mean).")
    return wt


# ==========================================================================
# BLOCK 5 - FEW-SHOT (paper Figure 6, dataset-level Wilcoxon at 2 shots)
# ==========================================================================
def analyse_fewshot():
    banner("BLOCK 5 - Few-shot ordinal (paper Figure 6)")
    df = load("block5_few_shot.jsonl")

    qwk = df.pivot_table(index="model", columns="shots_per_class",
                         values="qwk", aggfunc="mean")
    smr = df.pivot_table(index="model", columns="shots_per_class",
                         values="severe_misranking_rate", aggfunc="mean")
    out = pd.concat({"QWK": qwk, "SMR": smr}, axis=1)
    out.to_csv(os.path.join(OUT, "fewshot_by_shots.csv"))
    print("QWK by model x shots:")
    print(qwk.round(4).to_string())
    print("\nSMR by model x shots:")
    print(smr.round(4).to_string())

    # dataset-level Wilcoxon at 2 shots (paper's reported test)
    rows = []
    sub = df[df["shots_per_class"] == 2]
    ds = (sub.groupby(["dataset", "model"])["severe_misranking_rate"]
             .mean().reset_index()
             .pivot_table(index="dataset", columns="model",
                          values="severe_misranking_rate"))
    for base, rc in [("TabPFN_flat", "TabPFN_RankCode")]:
        if base in ds and rc in ds:
            d = ds[[base, rc]].dropna()
            W, pw = stats.wilcoxon(d[base], d[rc])
            delta = (d[base] - d[rc]).values
            lo, hi = bootstrap_mean_ci(delta)
            rows.append({
                "shots": 2, "comparison": f"{base} vs {rc}",
                "n_datasets": len(d),
                "flat_SMR": d[base].mean(),
                "rankcode_SMR": d[rc].mean(),
                "mean_dSMR": delta.mean(),
                "rel_reduction_pct": delta.mean() / d[base].mean() * 100,
                "wilcoxon_p": pw,
                "mean_ci_lo": lo, "mean_ci_hi": hi,
            })
    ft = pd.DataFrame(rows)
    ft.to_csv(os.path.join(OUT, "fewshot_2shot_wilcoxon.csv"), index=False)
    print("\nDataset-level Wilcoxon at 2 shots:")
    print(ft.round(4).to_string(index=False))
    return ft


# ==========================================================================
# BLOCK 6 - DISTRIBUTION SHIFT (paper Figure 8, dataset-level Wilcoxon)
# ==========================================================================
def analyse_shift():
    banner("BLOCK 6 - Distribution shift (paper Figure 8)")
    df = load("block6_dist_shift.jsonl")

    qwk = df.pivot_table(index="model", columns="shift",
                         values="qwk", aggfunc="mean")
    smr = df.pivot_table(index="model", columns="shift",
                         values="severe_misranking_rate", aggfunc="mean")
    out = pd.concat({"QWK": qwk, "SMR": smr}, axis=1)
    out.to_csv(os.path.join(OUT, "distshift_by_shift.csv"))
    print("QWK by model x shift:")
    print(qwk.round(4).to_string())

    # degradation = QWK(none) - QWK(prior_shift), per dataset; paired Wilcoxon
    rows = []
    for base, rc in [("TabICL_flat", "TabICL_RankCode"),
                     ("TabPFN_flat", "TabPFN_RankCode")]:
        deg = {}
        for m in [base, rc]:
            sub = df[df["model"] == m]
            p = sub.pivot_table(index="dataset", columns="shift", values="qwk")
            if "none" in p and "prior_shift" in p:
                deg[m] = (p["none"] - p["prior_shift"]).dropna()
        if base in deg and rc in deg:
            common = deg[base].index.intersection(deg[rc].index)
            W, pw = stats.wilcoxon(deg[base][common], deg[rc][common])
            rows.append({
                "comparison": f"{base} vs {rc}",
                "n_datasets": len(common),
                "flat_degradation": deg[base][common].mean(),
                "rankcode_degradation": deg[rc][common].mean(),
                "wilcoxon_p": pw,
            })
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(OUT, "distshift_degradation.csv"), index=False)
    print("\nDataset-level degradation Wilcoxon:")
    print(st.round(4).to_string(index=False))
    return st


# ==========================================================================
# D2 - SELECTIVE-SCREENING AUDIT FROM Q-VECTORS
# ==========================================================================
def analyse_d2_selective_screening():
    """Print the completed D2 q-vector selective-screening audit.

    The full clean q-vector JSONL is large and is not required for the
    lightweight downstream reproducibility package. The derived gate table is
    included in ./results; the full audit can be recomputed from the clean
    q-vector file using qvec_tools/rankcode_d2_selective_screening.py.
    """
    path = os.path.join(OUT, "D2_paired_gate_vs_flat_margin.csv")
    if not os.path.exists(path):
        print("\nD2 selective-screening table not found; skipping optional D2 summary.")
        return None
    banner("D2 - Selective-screening audit from cumulative q-vectors")
    d2 = pd.read_csv(path)
    keep = d2[d2["candidate"].isin(["threshold_cumul", "rankcode_cumul"])].copy()
    keep.to_csv(os.path.join(OUT, "D2_selective_screening_summary_from_analysis.csv"), index=False)
    print(keep.round(6).to_string(index=False))
    return keep


# ==========================================================================
def main():
    print("RankCode-ICL - reproducing all paper statistics")
    print(f"(bootstrap seed = {RNG_SEED}, n_boot = {N_BOOT})")
    analyse_main()
    analyse_formulation()
    analyse_context()
    analyse_imbalance()
    analyse_fewshot()
    analyse_shift()
    analyse_d2_selective_screening()
    banner("DONE - all CSV artefacts written to ./results/")
    for f in sorted(os.listdir(OUT)):
        print("  results/" + f)


if __name__ == "__main__":
    main()
