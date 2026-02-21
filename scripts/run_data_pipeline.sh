#!/usr/bin/env bash
set -euo pipefail

python -m src.data.load_raw
python -m src.data.clean_preprocess
python -m src.data.build_domains
python -m src.data.build_rps
python -m src.data.build_graphs
python -m src.data.build_samples
