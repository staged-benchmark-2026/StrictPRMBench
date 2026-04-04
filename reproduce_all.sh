#!/bin/bash
set -euo pipefail
python3 smoke_test.py
python3 reproduce_table1.py
python3 reproduce_table2.py
python3 reproduce_table3.py
python3 reproduce_human.py
