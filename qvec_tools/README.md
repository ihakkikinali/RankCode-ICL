# q-vector selective-screening script

This folder contains the optional script used to reproduce the D2 selective-screening tables from a completed clean q-vector file.

The full clean q-vector file is not included because it is large. The derived D2 CSV outputs used by the manuscript are included in `../results/`.

To run the script, place `qvec_main_clean_final.jsonl` in the expected results directory or edit the path in the script, then run:

```bash
python rankcode_d2_selective_screening.py
```
