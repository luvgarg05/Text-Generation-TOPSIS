import numpy as np
import pandas as pd

data = pd.DataFrame({
    "Quality": [7.0, 8.0, 6.0, 3.0, 8.5],
    "Perplexity": [29.17, 24.09, 71.45, 50.0, 4.21],
    "Time": [5.91, 10.42, 6.50, 0.32, 8.40],
    "Size": [500, 650, 330, 240, 500],
    "Consistency": [7.0, 8.0, 5.0, 3.0, 8.5]
}, index=[
    "GPT-2",
    "GPT-Neo 125M",
    "DistilGPT-2",
    "T5-Small",
    "GPT-2 Fine-Tuned"
])

# Weights (sum = 1)
weights = np.array([0.30, 0.25, 0.15, 0.10, 0.20])

# Impacts: + benefit, - cost
impacts = ["+", "-", "-", "-", "+"]

# Normalize
norm = data / np.sqrt((data**2).sum())

# Weighted normalized matrix
weighted = norm * weights

# Ideal best & worst
ideal_best = []
ideal_worst = []

for i, col in enumerate(weighted.columns):
    if impacts[i] == "+":
        ideal_best.append(weighted[col].max())
        ideal_worst.append(weighted[col].min())
    else:
        ideal_best.append(weighted[col].min())
        ideal_worst.append(weighted[col].max())

ideal_best = np.array(ideal_best)
ideal_worst = np.array(ideal_worst)

# Distances
dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

# TOPSIS score
scores = dist_worst / (dist_best + dist_worst)

data["TOPSIS Score"] = scores
data["Rank"] = data["TOPSIS Score"].rank(ascending=False)

print(data.sort_values("Rank"))
