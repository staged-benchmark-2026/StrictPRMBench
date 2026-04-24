import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "results/matched_trace/product_vs_gm_min_last_14cell.csv").open()))

print("=== Matched-Trace Extension ===")
for key, lo, hi in [
    ("delta_prod_gm", "ci_lo_gm", "ci_hi_gm"),
    ("delta_prod_min", "ci_lo_min", "ci_hi_min"),
    ("delta_prod_last", "ci_lo_last", "ci_hi_last"),
]:
    n_pos = sum(float(r[key]) > 0 for r in rows)
    n_ci = sum(float(r[lo]) > 0 and float(r[hi]) > 0 for r in rows)
    print(f"{key}: positive in {n_pos}/{len(rows)} cells; CI>0 in {n_ci}/{len(rows)} cells")

pooled_json = root / "results/matched_trace/pooled_hierarchical_summary.json"
pooled_txt = root / "results/matched_trace/pooled_hierarchical_summary.txt"
if pooled_json.exists():
    data = json.load(pooled_json.open())
    print("\n=== Pooled Hierarchical Summary ===")
    for block in ["all_14_cells", "native_only"]:
        payload = data.get(block, {})
        if not payload:
            continue
        for label in ["delta_prod_gm", "delta_prod_min", "delta_prod_last"]:
            sub = payload.get(label, {})
            mean = sub.get("mean")
            ci_lo = sub.get("ci_lo")
            ci_hi = sub.get("ci_hi")
            if mean is not None:
                print(f"{block} {label} = {mean:+.6f} [{ci_lo:+.6f}, {ci_hi:+.6f}]")
if pooled_txt.exists():
    print("\n=== Pooled Hierarchical Summary (raw) ===")
    print(pooled_txt.read_text().strip())
