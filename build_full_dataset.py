import os
import pandas as pd

DATA_DIR = "data/processed"

print("Loading static features...")

nodes = pd.read_csv(os.path.join(DATA_DIR, "nodes.csv"))
ndvi = pd.read_csv(os.path.join(DATA_DIR, "node_ndvi.csv"))
dem = pd.read_csv(os.path.join(DATA_DIR, "node_dem.csv"))
slope = pd.read_csv(os.path.join(DATA_DIR, "node_slope.csv"))

# merge terrain features
static = nodes.merge(ndvi, on="node_id")
static = static.merge(dem, on="node_id")
static = static.merge(slope, on="node_id")

print("Static feature shape:", static.shape)

# load fire labels
print("Loading fire labels...")
fire_labels = pd.read_csv(os.path.join(DATA_DIR, "fire_labels.csv"))

print("Fire label shape:", fire_labels.shape)

# find weather files
weather_files = sorted([
    f for f in os.listdir(DATA_DIR) if f.startswith("weather_")
])

data_list = []

for wf in weather_files:

    print("Processing", wf)

    weather = pd.read_csv(os.path.join(DATA_DIR, wf))

    # merge static terrain
    merged = weather.merge(static, on="node_id")

    # merge fire labels
    merged = merged.merge(
        fire_labels,
        on=["node_id", "date"],
        how="left"
    )

    # replace missing fire labels with 0
    merged["fire"] = merged["fire"].fillna(0)

    data_list.append(merged)

# combine all months
full_data = pd.concat(data_list)

print("Final dataset shape:", full_data.shape)

# save dataset
output_path = os.path.join(DATA_DIR, "node_features_full_year.csv")

full_data.to_csv(output_path, index=False)

print("Dataset saved to:", output_path)