import csv
from pathlib import Path

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / 'results/optimization_bridge/hijack_results.csv').open()))
print('=== Table 3: Optimization Bridge ===')
for row in rows:
    print(
        f"{row['prm']} | {row['benchmark']} | optimize={row['optimize_for']} | n={row['n_nontrivial']} | "
        f"del={float(row['mean_n_deletions']):.2f} | gain={float(row['mean_score_gain']):+.4f} | "
        f"product={float(row['hijack_product']):.3f} | gm={float(row['hijack_gm']):.3f} | "
        f"cluster={float(row['hijack_cluster']):.3f} | official={float(row['hijack_official']):.3f} | "
        f"delta_pg={float(row['delta_product_minus_gm']):+.3f} [{float(row['delta_ci_lo']):+.3f}, {float(row['delta_ci_hi']):+.3f}]"
    )
