import csv
from pathlib import Path

root = Path(__file__).resolve().parent
fc_rows = list(csv.DictReader((root / 'human_cards/forced_choice/forced_choice_results.csv').open()))
print('=== Forced Choice ===')
for row in fc_rows:
    print(f"{row['prm']} | {row['benchmark']} | gm_rate={float(row['gm_rate']):.3f} | n={row['n']} | kappa={float(row['fleiss_kappa']):.3f}")

cluster_robust = root / 'results/cluster_robust/summary.txt'
if cluster_robust.exists():
    print('\n=== Cluster Robust Summary ===')
    print(cluster_robust.read_text().strip())
