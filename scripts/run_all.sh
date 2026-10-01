#!/usr/bin/env bash
# Reproduce every table and figure in the paper (about 30 min on a 2-core laptop).
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1

bash scripts/download_data.sh
cd src
for seed in 0 1 2; do
  python run_natural.py --seed "$seed"        # E1, E3, E4, E5 raw results
done
python run_synthetic.py                        # E2 controlled shifts
python analysis.py                             # tables -> results/tables, figures -> figures/
python case_study.py                           # per-feature case-study figure
python ablation_samplesize.py                  # target sample-size ablation (adds a table)
python analysis.py                             # refresh tables
cd ../paper && cp ../figures/*.pdf figures/ && cp ../results/tables/*.tex tables/ && latexmk -pdf -quiet main.tex
