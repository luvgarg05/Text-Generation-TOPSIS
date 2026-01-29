import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


models = [
    "GPT-2",
    "GPT-Neo 125M",
    "DistilGPT-2",
    "T5-Small",
    "GPT-2 Fine-Tuned"
]

topsis_scores = [
    0.618358,
    0.598966,
    0.289648,
    0.395952,
    0.735541
]

# -------------------------------
# BAR CHART: TOPSIS Scores
# -------------------------------
plt.figure()
plt.bar(models, topsis_scores)
plt.xticks(rotation=30)
plt.ylabel("TOPSIS Score")
plt.title("TOPSIS Scores of Text Generation Models")
plt.tight_layout()
plt.show()

# -------------------------------
# RADAR CHART: Criteria Comparison
# -------------------------------
criteria = ["Quality", "Perplexity", "Time", "Size", "Consistency"]

data = {
    "GPT-2": [7.0, 29.17, 5.91, 500, 7.0],
    "GPT-Neo 125M": [8.0, 24.09, 10.42, 650, 8.0],
    "DistilGPT-2": [6.0, 71.45, 6.50, 330, 5.0],
    "T5-Small": [3.0, 50.0, 0.32, 240, 3.0],
    "GPT-2 Fine-Tuned": [8.5, 4.21, 8.40, 500, 8.5]
}

df = pd.DataFrame(data, index=criteria)

# Normalize values for radar chart
df_norm = df / df.max()

angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False)
angles = np.concatenate([angles, [angles[0]]])

plt.figure()
ax = plt.subplot(111, polar=True)

for model in df_norm.columns:
    values = df_norm[model].values
    values = np.concatenate([values, [values[0]]])
    ax.plot(angles, values, label=model)

ax.set_thetagrids(angles[:-1] * 180 / np.pi, criteria)
plt.title("Radar Chart of Model Performance")
plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
plt.tight_layout()
plt.show()
