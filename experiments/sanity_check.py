import pandas as pd, sys

path = sys.argv[1]
df = pd.read_csv(path)

success = df[df.success == 1]
grouped = success.groupby("image_id").size()

print("Total rows:", len(df))
print("Unique images:", df.image_id.nunique())
print("Images succeeded:", len(grouped))
print("Max success per image:", int(grouped.max()) if len(grouped) else 0)