import pandas as pd
import matplotlib.pyplot as plt

nodes = pd.read_csv("data/processed/nodes.csv")

lat = nodes["lat"]
lon = nodes["lon"]

plt.figure(figsize=(8,8))

plt.scatter(
    lon,
    lat,
    s=1,
    color="red"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title("Spatial Distribution of Graph Nodes")

plt.grid()

plt.savefig("outputs/node_distribution_map.png", dpi=300)

print("Node spatial map saved")