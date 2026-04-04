import csv
from pathlib import Path

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / 'results/ni_gap/answer_preserved_ni.csv').open()))
print('=== Table 1: NI Gap ===')
for row in rows:
    print(
        f"{row['prm']} | {row['benchmark']} | n={row['n_answer_preserved']} | "
        f"raw={float(row['delta_ni_raw']):+.3f} [{float(row['raw_ci_lo']):+.3f}, {float(row['raw_ci_hi']):+.3f}] | "
        f"matched={float(row['delta_ni_matched']):+.3f} [{float(row['matched_ci_lo']):+.3f}, {float(row['matched_ci_hi']):+.3f}]"
    )
