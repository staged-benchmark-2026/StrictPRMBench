# No-Preference Forced-Choice Reannotation Subset

This directory contains the 200-card no-preference reannotation subset used as an endpoint-robustness check for the original forced-choice human evaluation.

## Scope

- Source: 200 cards sampled from the original non-product-trained forced-choice blinded cards, not an independent new sample.
- Balanced cells: Skywork-7B / InternLM2-7B x GSM8K / MATH-500, 50 cards per cell.
- Endpoint: majority label over `GM`, `product`, and `no_preference` with 3 annotators per card.
- Blinding: independent A/B assignment per annotator; product position is balanced overall at 304/600 = 50.7% A.

## Headline Results

- GM majority: 113/200 = 56.5%.
- Product majority: 1/200 = 0.5%.
- No-preference majority: 84/200 = 42.0%.
- Split: 2/200 = 1.0%.
- Resolved-card GM preference: 113/114 = 99.1%; this is a secondary resolved-card diagnostic, not the primary headline.
- Fleiss kappa: 0.071. The low agreement is driven mainly by annotator-specific no-preference thresholds; product choices remain rare for every annotator.

## Files

- `completed/`: returned annotator workbooks.
- `per_annotator_manifest.csv`: card order and A/B assignment per annotator.
- `deblind_key.jsonl`: card metadata and A/B assignments for deblinding.
- `selected_cards.jsonl`: normalized card text and source metadata.
- `analysis/`: released analysis outputs.
- `protocol/`: annotation guide and timestamped analysis plan.
- `release_metadata/`: build and verification summaries with release-relative paths.

## Reproduction

Run from the repository root:

```bash
python3 reproduce_no_preference_reannotation.py --check
```

The script recomputes the analysis from the completed XLSX files, `per_annotator_manifest.csv`, and `deblind_key.jsonl`, then byte-compares the regenerated CSV/TXT outputs against `analysis/`.
