#!/bin/bash
set -euo pipefail
python3 smoke_test.py
python3 reproduce_table1.py
python3 reproduce_table2.py
python3 reproduce_table3.py
python3 reproduce_human.py
python3 reproduce_no_preference_reannotation.py --check
python3 reproduce_extensions.py
