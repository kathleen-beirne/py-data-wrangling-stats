# py-data-wrangling-stats

**End-to-end demo:** clean → descriptives → GLM → ML.  
CLI-driven, tested, and reproducible. Data: small synthetic CSV/Excel.

![CI](https://github.com/kathleen-beirne/py-data-wrangling-stats/actions/workflows/ci.yml/badge.svg)](https://github.com/kathleen-beirne/py-data-wrangling-stats/actions/workflows/ci.yml)

## How to run (locally)
```bash
# (Optional) create conda env
conda env create -f environment.yml
conda activate wrangle

# Clean data (choose the right raw file path)
python -m src.cli clean --raw data/raw/synthetic_dataset.csv

# Descriptives + plot
python -m src.cli describe

# GLM
python -m src.cli glm

# ML
python -m src.cli ml
