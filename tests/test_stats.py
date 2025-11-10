from src.etl import clean_data, load_cleaned
from src.stats import descriptives, glm
from pathlib import Path

def test_descriptives_and_glm(tmp_path: Path):
    clean_data("data/raw/synthetic_dataset.csv", "data/processed/clean.csv")
    df = load_cleaned()
    rp = descriptives(df, "reports/summary.md")
    gp = glm(df, "reports/summary.md")
    assert Path(rp).exists()
    assert Path(gp).exists()
