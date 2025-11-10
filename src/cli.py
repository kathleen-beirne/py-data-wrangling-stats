import argparse
from src.etl import clean_data, load_cleaned
from src.stats import descriptives, glm
from src.ml import simple_classifier

def main():
    p = argparse.ArgumentParser(description="Wrangling/Stats demo CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_clean = sub.add_parser("clean", help="Clean raw data -> processed CSV")
    p_clean.add_argument("--raw", default="data/raw/synthetic_dataset.csv")
    p_clean.add_argument("--out", default="data/processed/clean.csv")

    sub.add_parser("describe", help="Descriptives + figure")
    sub.add_parser("glm", help="Fit GLM: score1 ~ age + C(group)")
    sub.add_parser("ml", help="Train simple classifier for target_asd")

    args = p.parse_args()

    if args.cmd == "clean":
        path = clean_data(args.raw, args.out)
        print(f"Wrote {path}")
    else:
        df = load_cleaned()
        if args.cmd == "describe":
            print(descriptives(df))
        elif args.cmd == "glm":
            print(glm(df))
        elif args.cmd == "ml":
            print(simple_classifier(df))

if __name__ == "__main__":
    main()
