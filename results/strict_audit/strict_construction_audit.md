# Strict Construction Audit

## Support Corrupt (strict)

| Metric | Value |
|---|---:|
| total_pairs | 1863 |
| intended_operator_swap | 720 |
| intended_numeric_perturb | 1143 |
| effective_operator_swap_ratio | 1.0000 |
| effective_numeric_perturb_ratio | 0.9965 |
| fallback_alt_rate_operator_swap | 0.0000 |
| fallback_alt_rate_numeric_perturb | 0.0000 |

## Canonical Diff Audit (scorer-input aligned)

| Subtype | n | pct_len_diff_eq_0 | pct_prefix32_equal | pct_char_diff_eq_1_when_len_equal |
|---|---:|---:|---:|---:|
| operator_swap | 720 | 0.8472 | 0.9958 | 1.0000 |
| numeric_perturb | 1143 | 0.5547 | 0.9536 | 0.8265 |

## Necessary Delete Sanity

| Metric | Value |
|---|---:|
| total_pairs | 2211 |
| step_count_drop_rate | 1.0000 |
| canonical_len_diff_rate | 1.0000 |
| canonical_char0_diff_rate | 0.1049 |
| canonical_prefix32_equal_rate | 0.7761 |

## Note

- Canonical text is reconstructed from step texts (`\n` join), matching scorer-facing step content semantics.
- Raw-text-only diffs are reported in JSON (`support_corrupt.by_subtype.*.raw_*`) for forensic checks.