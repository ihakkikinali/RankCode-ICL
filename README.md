# RankCode-ICL reproducibility package

This repository contains the reproducibility package for the manuscript:

**Severe Misranking in Tabular Foundation Models: Ordinal Reliability Modes for In-Context Learning**

The package contains released experiment logs, derived result tables, figure-generation code and the D2 selective-screening outputs. The full q-vector JSONL file is not included because it is large; derived D2 outputs are included.

## Repository contents

```text
rankcode_icl_reproducibility/
├── analysis.py                  Recomputes downstream tables and statistical tests
├── generate_figures.py          Recreates manuscript figures
├── requirements.txt             Python dependencies
├── environment.txt              Environment snapshot
├── data/                        Released JSONL logs for benchmark and stress blocks
├── results/                     Derived CSV tables, including D2 outputs
├── figures/                     Generated manuscript figures
└── qvec_tools/                  Optional D2 script for users with the clean q-vector file
```

## Quick start

```bash
pip install -r requirements.txt
python analysis.py
python generate_figures.py
```

`analysis.py` is deterministic for the released logs. Bootstrap confidence intervals use fixed settings in the script.

## Main interpretation

The manuscript does not claim that RankCode-ICL is a universal accuracy booster. The supported interpretation is narrower:

1. Flat tabular foundation models can obtain strong aggregate ordinal scores while still producing measurable severe misranking.
2. Threshold-mode cumulative decomposition reduces severe misranking on the natural benchmark with little change in aggregate QWK.
3. The balanced RankCode mode is most useful under scarcity and imbalance, where ordinal boundaries or extreme classes are under-represented.
4. The D2 audit shows that threshold-derived cumulative q-vector scores provide a significant low-rate selective-screening signal for severe misranking.

## D2 selective-screening audit

The D2 audit asks whether ordinal cumulative q-vector signals identify examples at risk of severe misranking better than the flat TabPFN softmax-margin signal.

Main result against `flat_margin`:

| Candidate | Reject fraction | Flat SMR | Candidate SMR | Delta SMR | Relative reduction | Two-sided Wilcoxon p | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| threshold_cumul | 0.10 | 0.031737 | 0.024424 | 0.007314 | 23.0% | 0.014539 | GO |
| threshold_cumul | 0.20 | 0.029521 | 0.023755 | 0.005766 | 19.5% | 0.005634 | GO |
| threshold_cumul | 0.30 | 0.027600 | 0.023344 | 0.004256 | 15.4% | 0.071592 | directional |
| rankcode_cumul | 0.10 | 0.031737 | 0.027842 | 0.003896 | 12.3% | 0.144984 | STOP |
| rankcode_cumul | 0.20 | 0.029521 | 0.027275 | 0.002246 | 7.6% | 0.387543 | STOP |

The supported claim is not that the full RankCode cumulative score is a strong abstention criterion. The supported claim is that threshold-derived cumulative ordinal scores provide a significant low-rate screening signal for severe misranking.

Relevant files:

- `results/D2_paired_gate_vs_flat_margin.csv`
- `results/D2_selective_screening_summary.csv`
- `results/D2_manifest.json`
- `figures/fig9_d2_selective_screening.pdf`
- `qvec_tools/rankcode_d2_selective_screening.py`

The full clean q-vector JSONL was hard-audited as complete before deriving the included D2 tables: 5,479,020 unique rows, 4,140/4,140 jobs and 1,380/1,380 split pairs. The clean q-vector JSONL itself is not included because it is approximately 2 GB.

## Output-to-paper mapping

| Output file | Paper element |
|---|---|
| `results/table3_main_benchmark.csv` | Main benchmark over 46 datasets and 30 splits |
| `results/main_friedman_meta.csv`, `results/main_friedman_nemenyi.csv` | Dataset-level Friedman/Nemenyi QWK comparison |
| `results/main_pairwise_smr_wilcoxon.csv` | Main benchmark paired SMR tests |
| `results/table1_dataset_K_distribution.csv` | Distribution of ordinal class count K |
| `results/table4_formulation_ablation.csv` | Formulation ablation |
| `results/context_ablation_budget_sweep.csv` | Context-budget and context-design ablation |
| `results/imbalance_smr_by_scenario.csv`, `results/imbalance_seed_wilcoxon.csv` | Controlled imbalance stress test |
| `results/monotone_repair_summary.csv` | Isotonic-repair diagnostics |
| `results/fewshot_by_shots.csv`, `results/fewshot_2shot_wilcoxon.csv` | Few-shot ordinal learning block |
| `results/distshift_by_shift.csv`, `results/distshift_degradation.csv` | Distribution-shift diagnostic block |
| `results/D2_paired_gate_vs_flat_margin.csv` | D2 q-vector selective-screening gate |
| `figures/fig9_d2_selective_screening.pdf` | Compact D2 selective-screening overview |

## Statistical protocol

- The primary unit for the main benchmark is the dataset. Scores are averaged over each dataset's fixed splits before Friedman/Nemenyi or paired Wilcoxon testing.
- Stress blocks use the replication unit appropriate to the block. Controlled imbalance is seed-level. Few-shot and distribution-shift analyses are dataset-level after split aggregation.
- The D2 selective-screening audit is a post-hoc reliability analysis from completed q-vector outputs. The main gate compares residual severe-misranking rate after rejecting the highest-risk fraction of examples according to each score.

## Data notes

The TOC-UCO datasets themselves are not redistributed. The JSONL logs in `data/` are sufficient to reproduce downstream tables, tests and figures without re-running foundation-model inference. Full q-vector regeneration is not required for ordinary reproduction from this package.
