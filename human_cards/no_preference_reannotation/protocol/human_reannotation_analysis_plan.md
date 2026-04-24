# Human Replication Analysis Plan: No-Preference Forced-Choice

Timestamp: 2026-04-24 11:14:09 CST

## Objective

Estimate whether the original forced-choice GM preference persists when annotators may choose `No preference`.

## Sample

- Total cards: 200
- PRMs: Skywork-7B (`skywork_prm`) and InternLM2-7B (`internlm2_7b_reward`)
- Benchmarks: GSM8K and MATH-500
- Balanced cells: 50 cards per PRM x benchmark cell
- Inclusion rule: both product-selected and GM-selected traces have final answers matching the recorded correct answer
- Trace preprocessing: chat-template preambles are stripped before packaging; cards with residual chat markers near the start are excluded

## Annotation

- 3 independent annotators per card
- Response options: `A`, `B`, `No preference`
- Each annotator receives the same 200 cards with independently shuffled order and independently randomized A/B assignment

## Endpoints

1. Raw majority outcome: GM / product / no preference / split
2. Resolved-card GM preference among cards with GM or product majority
3. Majority no-preference rate

## Planned Analysis

- Deblind each annotator response using the recorded per-annotator assignment map
- Per card, compute majority over `GM`, `product`, and `No preference`
- If all three labels appear once, mark the card as `split`
- Report raw majority counts and percentages over all 200 cards
- Report resolved-card GM preference as `GM_majority / (GM_majority + product_majority)`
- Report no-preference majority rate as `No_preference_majority / 200`
- Stratify all headline counts by PRM and benchmark
- Use bootstrap over cards with seed 42 for 95% CIs

## Frozen Build Parameters

- Sampling seed: 42
- Cards per cell: 50
- Build directory: `human_cards/no_preference_reannotation`
