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
    print('ERROR: pip install -r requirements.txt')
    sys.exit(1)

for rel in ['pools', 'edits', 'scores', 'human_cards', 'results']:
    assert (ROOT / rel).is_dir(), f'missing directory: {rel}'

gsm8k_pools = list((ROOT / 'pools/gsm8k').glob('*.json'))
math500_pools = list((ROOT / 'pools/math500').glob('*.json'))
assert len(gsm8k_pools) == 500, len(gsm8k_pools)
assert len(math500_pools) == 500, len(math500_pools)
assert len(list((ROOT / 'scores').glob('*.jsonl'))) >= 28
assert len(list((ROOT / 'scores/multiseed').glob('*.jsonl'))) == 28
assert (ROOT / 'pools/bon_candidates/gsm8k_full_500').is_dir()
assert (ROOT / 'pools/bon_candidates/math500_full_500').is_dir()

with open(ROOT / 'human_cards/forced_choice/forced_choice_results.csv', newline='') as f:
    fc_rows = list(csv.DictReader(f))
assert len(fc_rows) == 5

with open(ROOT / 'results/ni_gap/answer_preserved_ni.csv', newline='') as f:
    ni_rows = list(csv.DictReader(f))
assert len(ni_rows) == 14

sample_pool = gsm8k_pools[0]
obj = json.load(sample_pool.open())
assert isinstance(obj, dict)
assert 'steps' in obj and 'problem_id' in obj

print('PASSED: smoke test')
print('gsm8k strict traces:', len(gsm8k_pools))
print('math500 strict traces:', len(math500_pools))
print('score files:', len(list((ROOT / 'scores').glob('*.jsonl'))))
print('bon candidate dirs:', len(list((ROOT / 'pools/bon_candidates').glob('*'))))
