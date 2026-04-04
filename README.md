# StrictPRMBench

A strict counterfactual audit benchmark for stress-testing PRM aggregation rules beyond final-answer accuracy.

Anonymous submission artifact.

## Quick Start

```bash
pip install -r requirements.txt
python3 smoke_test.py
bash reproduce_all.sh
```

## Hardware

No GPU required. All reproduced outputs read from cached step scores and precomputed result tables. Included strict-trace files cover 500 GSM8K problems and 500 MATH-500 problems; BON candidate directories are also included for natural-pool analyses.

## Contents

| Directory | Description |
|-----------|-------------|
| `pools/` | Strict trace subset plus BON candidate pools for GSM8K and MATH-500 |
| `edits/` | Strict edit metadata and matched necessary/inert deletion pairs |
| `scores/` | Cached step scores for 7 PRMs across 2 benchmarks |
| `human_cards/` | Forced-choice, Likert, Qwen follow-up, and natural-pool human eval materials |
| `results/` | Precomputed paper-facing CSV/TXT/TEX outputs |
| `code/` | Sanitized reference scripts and lightweight reproduction helpers |

## Notes

- This artifact is replay-oriented. It does not require model weights or PRM inference.
- `code/configs/models.yaml` is included for reference only; the artifact reproduces from cached scores.
- The main reproduction entrypoints are `reproduce_table1.py`, `reproduce_table2.py`, `reproduce_table3.py`, and `reproduce_human.py`.
