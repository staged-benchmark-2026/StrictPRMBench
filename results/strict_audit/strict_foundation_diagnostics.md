# Strict Foundation Diagnostics

## 1) Family Format Integrity Audit (5 families)

| Family | n | raw_char0_diff | raw_len_diff | raw_think_removed | canonical_prefix32_equal | step_relation_pass | same_problem_id |
|---|---:|---:|---:|---:|---:|---:|---:|
| necessary_delete | 2211 | 0.1049 | 1.0000 | 0.0000 | 0.7761 | 1.0000 | 1.0000 |
| problem_mismatch | 500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| style_rewrite | 4450 | 1.0000 | 0.9944 | 1.0000 | 0.5582 | 1.0000 | 1.0000 |
| decorative_insert | 2000 | 1.0000 | 1.0000 | 1.0000 | 0.7500 | 1.0000 | 1.0000 |
| redundant_insert | 1486 | 1.0000 | 0.9865 | 1.0000 | 0.3324 | 1.0000 | 1.0000 |

## 2) InternLM2 CDS Decomposition

### internlm2_1_8b_reward

- recomputed_cds_overall: 0.409403

| Family | label | n | mean_delta | mean_signed_delta | prefer_original | prefer_modified | tie_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| decorative_insert | inert | 2000 | 0.000183 | -0.000002 | 0.3765 | 0.3675 | 0.2560 |
| necessary_delete | causal_important | 2211 | 0.088865 | -0.009460 | 0.3528 | 0.3921 | 0.2551 |
| problem_mismatch | causal_important | 500 | 0.048285 | 0.004405 | 0.5280 | 0.4720 | 0.0000 |
| redundant_insert | inert | 1486 | 0.026927 | -0.004197 | 0.4711 | 0.5262 | 0.0027 |
| style_rewrite | inert | 4450 | 0.040970 | -0.018468 | 0.3679 | 0.6292 | 0.0029 |
| support_corrupt | causal_important | 1863 | 0.001260 | -0.001260 | 0.0000 | 0.0161 | 0.9839 |

| Leave-one-family-out | CDS |
|---|---:|
| drop_decorative_insert | 0.364451 |
| drop_necessary_delete | 0.184884 |
| drop_problem_mismatch | 0.374078 |
| drop_redundant_insert | 0.412767 |
| drop_style_rewrite | 0.479722 |
| drop_support_corrupt | 0.658185 |

### internlm2_7b_reward

- recomputed_cds_overall: 0.806580

| Family | label | n | mean_delta | mean_signed_delta | prefer_original | prefer_modified | tie_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| decorative_insert | inert | 2000 | 0.000199 | 0.000003 | 0.3790 | 0.3710 | 0.2500 |
| necessary_delete | causal_important | 2211 | 0.156400 | 0.001824 | 0.5070 | 0.4604 | 0.0326 |
| problem_mismatch | causal_important | 500 | 0.030538 | -0.000119 | 0.5000 | 0.5000 | 0.0000 |
| redundant_insert | inert | 1486 | 0.030593 | -0.013148 | 0.3405 | 0.6568 | 0.0027 |
| style_rewrite | inert | 4450 | 0.029668 | -0.009938 | 0.3892 | 0.6070 | 0.0038 |
| support_corrupt | causal_important | 1863 | 0.182021 | 0.180227 | 0.9367 | 0.0612 | 0.0021 |

| Leave-one-family-out | CDS |
|---|---:|
| drop_decorative_insert | 0.752580 |
| drop_necessary_delete | 0.820298 |
| drop_problem_mismatch | 0.827786 |
| drop_redundant_insert | 0.820727 |
| drop_style_rewrite | 0.872357 |
| drop_support_corrupt | 0.762755 |

## 3) Unified CDS-Utility Correlation (strict)

| Dataset | Kendall tau-b (rank) | Spearman rho (CDS vs gain) |
|---|---:|---:|
| GSM8K | -0.195180 | +0.539618 |
| MATH500 | -0.523810 | -0.221448 |

