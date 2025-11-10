# py-data-wrangling-stats
[![ci](https://github.com/kathleen-beirne/py-data-wrangling-stats/actions/workflows/ci.yml/badge.svg)](https://github.com/kathleen-beirne/py-data-wrangling-stats/actions/workflows/ci.yml)

End-to-end, reproducible demo: **clean → descriptives → GLM → ML** on a small synthetic dataset.

## Quick start
```bash
# (Optional) conda
conda env create -f environment.yml
conda activate wrangle

# Clean data (choose the raw file you uploaded)
python -m src.cli clean --raw data/raw/synthetic_dataset.csv
# or
# python -m src.cli clean --raw data/raw/synthetic_dataset.xlsx

# Descriptives + figure
python -m src.cli describe

# GLM
python -m src.cli glm

# ML
python -m src.cli ml
