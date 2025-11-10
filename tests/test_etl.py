from pathlib import Path
from src.etl import clean_data, load_cleaned

def test_clean_and_load(tmp_path: Path):
    # Use the committed CSV path
    out = clean_data("data/raw/synthetic_dataset.csv", "data/processed/clean.csv")
    assert Path(out).exists()
    df = load_cleaned(out)
    assert len(df) > 0
    assert "score1_z" in df.columns
