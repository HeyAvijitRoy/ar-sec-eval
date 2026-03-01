# experiments/sanity_check.py
import pandas as pd
import sys

def main():
    path = sys.argv[1]
    df = pd.read_csv(path)

    # Drop the hyperparams/meta row if present
    df = df[df["image_id"] != "__HYPERPARAMS__"]

    success = df[df.success == 1]
    grouped = success.groupby("image_id").size()

    print("Total rows:", len(df))
    print("Unique images:", df.image_id.nunique())
    print("Images succeeded:", len(grouped))
    print("Max success per image:", int(grouped.max()) if len(grouped) else 0)

    # Extra checks (helpful)
    # 1) Any image has >1 success row? (should be 0)
    if len(grouped) and grouped.max() > 1:
        bad = grouped[grouped > 1].sort_values(ascending=False).head(10)
        print("\nWARNING: Some images have multiple success rows:")
        print(bad)

    # 2) Any images never reached success?
    failed = sorted(set(df["image_id"].unique()) - set(success["image_id"].unique()))
    print("Images failed:", len(failed))
    if failed:
        print("Example failed images:", failed[:10])

if __name__ == "__main__":
    main()