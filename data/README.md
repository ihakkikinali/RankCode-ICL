# Data files

This directory contains released JSONL experiment logs for the benchmark and stress-test blocks. The original TOC-UCO datasets are not redistributed.

Files:

- `block1_main_fixed.jsonl`: main benchmark log with corrected ordinal label mapping
- `block2_formulation_ablation.jsonl`: formulation ablation log
- `block3_context_ablation.jsonl`: context-design and budget ablation log
- `block4_imbalance_stress.jsonl`: controlled imbalance stress-test log
- `block5_few_shot.jsonl`: few-shot log
- `block6_dist_shift.jsonl`: distribution-shift diagnostic log

The q-vector file used for the D2 selective-screening audit is not included because the clean file is approximately 2 GB. Derived D2 tables are included under `../results/`.
