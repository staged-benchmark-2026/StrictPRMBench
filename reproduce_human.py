import csv
from pathlib import Path

root = Path(__file__).resolve().parent
fc_rows = list(csv.DictReader((root / "human_cards/forced_choice/forced_choice_results.csv").open()))
print("=== Forced Choice ===")
for row in fc_rows:
    print(f"{row['prm']} | {row['benchmark']} | gm_rate={float(row['gm_rate']):.3f} | n={row['n']} | kappa={float(row['fleiss_kappa']):.3f}")

cluster_robust = root / "results/cluster_robust/summary.txt"
if cluster_robust.exists():
    print("\n=== Cluster Robust Summary ===")
    print(cluster_robust.read_text().strip())

expert_summary = root / "human_cards/expert_replication/expert_summary.txt"
if expert_summary.exists():
    print("\n=== Expert Mini Replication ===")
    print(expert_summary.read_text().strip())

natural_reannotation = root / "human_cards/natural_pool_reannotation/summary.txt"
if natural_reannotation.exists():
    print("\n=== Natural-Pool Reannotation ===")
    print(natural_reannotation.read_text().strip())

no_pref_summary = root / "human_cards/no_preference_reannotation/analysis/summary.txt"
no_pref_sanity = root / "human_cards/no_preference_reannotation/analysis/sanity_checks_no_pref_reannotation.txt"
if no_pref_summary.exists():
    print("\n=== No-Preference Forced-Choice Reannotation ===")
    print(no_pref_summary.read_text().strip())
if no_pref_sanity.exists():
    print("\n=== No-Preference Reannotation Sanity Checks ===")
    print(no_pref_sanity.read_text().strip())
