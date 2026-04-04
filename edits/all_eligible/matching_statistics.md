# Matching Statistics

| Benchmark | Necessary pairs | Inert pairs | Matched pairs | Exact matches | Avg position diff | Avg token length diff |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 7737 | 1989 | 1989 | 1377 | 0.908 | 0.281 |
| math500 | 7916 | 3812 | 3812 | 1832 | 2.350 | 0.675 |

Construction notes:
- Necessary and inert pools are both mined from correct traces in the 32-candidate BoN pools, not from the legacy MATH-500 all-inc necessary-delete benchmark, to avoid the low clean-correct ceiling in that source.
- Exact matching enforces same benchmark, same difficulty bucket (based on correct-trace count quintiles), step position within ±1, and token length within ±30%.
- Remaining pairs are filled by same-bucket fallback with closest position/length match.
