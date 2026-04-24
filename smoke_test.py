import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401
except Exception:
    print("ERROR: pip install -r requirements.txt")
    sys.exit(1)

for rel in ["pools", "edits", "scores", "human_cards", "results"]:
    assert (ROOT / rel).is_dir(), f"missing directory: {rel}"

gsm8k_pools = list((ROOT / "pools/gsm8k").glob("*.json"))
math500_pools = list((ROOT / "pools/math500").glob("*.json"))
assert len(gsm8k_pools) == 500, len(gsm8k_pools)
assert len(math500_pools) == 500, len(math500_pools)
assert len(list((ROOT / "scores").glob("*.jsonl"))) >= 28
assert len(list((ROOT / "scores/multiseed").glob("*.jsonl"))) == 28
assert (ROOT / "pools/bon_candidates/gsm8k_full_500").is_dir()
assert (ROOT / "pools/bon_candidates/math500_full_500").is_dir()

with open(ROOT / "human_cards/forced_choice/forced_choice_results.csv", newline="") as f:
    fc_rows = list(csv.DictReader(f))
assert len(fc_rows) == 5

with open(ROOT / "results/ni_gap/answer_preserved_ni.csv", newline="") as f:
    ni_rows = list(csv.DictReader(f))
assert len(ni_rows) == 14

with open(ROOT / "results/matched_trace/product_vs_gm_min_last_14cell.csv", newline="") as f:
    matched_ext_rows = list(csv.DictReader(f))
assert len(matched_ext_rows) == 14

pooled_obj = json.load((ROOT / "results/matched_trace/pooled_hierarchical_summary.json").open())
assert "all_14_cells" in pooled_obj and "native_only" in pooled_obj

with open(ROOT / "human_cards/expert_replication/expert_results.csv", newline="") as f:
    expert_rows = list(csv.DictReader(f))
assert len(expert_rows) == 3

with open(ROOT / "human_cards/natural_pool_reannotation/majority_summary.csv", newline="") as f:
    natural_rows = list(csv.DictReader(f))
assert len(natural_rows) == 7
assert any(r["prm"] == "pooled" and r["benchmark"] == "all" for r in natural_rows)

assert (ROOT / "human_cards/audits/annotator_temporal_audit.csv").is_file()
assert (ROOT / "human_cards/audits/annotator_temporal_audit.md").is_file()

with open(ROOT / "human_cards/no_preference_reannotation/analysis/majority_summary.csv", newline="") as f:
    no_pref_rows = list(csv.DictReader(f))
assert len(no_pref_rows) == 9
no_pref_overall = next(r for r in no_pref_rows if r["group"] == "overall")
assert int(no_pref_overall["n_cards"]) == 200
assert int(no_pref_overall["gm_majority"]) == 113
assert int(no_pref_overall["product_majority"]) == 1
assert int(no_pref_overall["no_preference_majority"]) == 84
assert len(list((ROOT / "human_cards/no_preference_reannotation/completed").glob("*.xlsx"))) == 3

sample_pool = gsm8k_pools[0]
obj = json.load(sample_pool.open())
assert isinstance(obj, dict)
assert "steps" in obj and "problem_id" in obj

print("PASSED: smoke test")
print("gsm8k strict traces:", len(gsm8k_pools))
print("math500 strict traces:", len(math500_pools))
print("score files:", len(list((ROOT / "scores").glob("*.jsonl"))))
print("matched-trace extension rows:", len(matched_ext_rows))
print("expert replication rows:", len(expert_rows))
print("natural-pool reannotation rows:", len(natural_rows))
print("no-preference reannotation rows:", len(no_pref_rows))
