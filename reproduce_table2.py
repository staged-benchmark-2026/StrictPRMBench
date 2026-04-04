import csv
from pathlib import Path

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / 'results/clustered_agg/primary_table.csv').open()))
print('=== Table 2: Aggregation Families ===')
for row in rows:
    print(
        f"{row['prm']} | {row['benchmark']} | Flip_prod={float(row['Flip_prod']):.3f} | "
        f"Flip_GM={float(row['Flip_GM']):.3f} | Flip_cluster={float(row['Flip_cluster']):.3f} | "
        f"Flip_official={float(row['Flip_official']):.3f} | Stab={row['Stab_Winner']} | Util={row['Util_Winner']}"
    )
