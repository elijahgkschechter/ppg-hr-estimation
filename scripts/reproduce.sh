#!/usr/bin/env bash
# Rebuild every result in this repository from the raw PPG-DaLiA download, in order.
# Run from the project folder with the virtual environment active:   bash scripts/reproduce.sh
# Total time: roughly 1.5-2 hours (Step 8's nested evaluation is the slow part).
set -euo pipefail

echo "== Tests ==";                         pytest -q
echo "== Step 4: raw files -> small files =="; python scripts/make_dataset.py
echo "== Step 8: gradient boosting ==";      python scripts/run_step8.py
echo "== Step 9: neural network ==";         python scripts/run_step9.py
echo "== Step 9b: stability check ==";       python scripts/run_step9_seeds.py
echo "== Step 10: error analysis ==";        jupyter nbconvert --to notebook --execute --inplace notebooks/07_error_analysis.ipynb
echo "Done. Results are in results/."
