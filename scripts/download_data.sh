#!/usr/bin/env bash
# Download the seven public tabular datasets used in the paper into data/raw/.
# All files are public mirrors of the original UCI / ProPublica / MOA releases.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"

get() { [ -s "$1" ] || { echo "downloading $1"; curl -fsSL -o "$1" "$2"; }; }

get adult.csv    https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv
get compas.csv   https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
get housing.csv  https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv
get bank.csv     https://raw.githubusercontent.com/selva86/datasets/master/bank-additional-full.csv
get elec.csv     https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/elec.csv
get airlines.csv https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/airlines.csv
get covtype.csv  https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/covtype.csv

wc -l *.csv
