import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

df = pd.read_csv("olympics.csv", header=None, skiprows=2)
df.columns = ["Country", "Summer_Games", "Summer_Gold", "Summer_Silver", "Summer_Bronze",
              "Summer_Total", "Winter_Gold", "Winter_Silver", "Winter_Bronze", "Winter_Total",
              "Summer_Winter_Gold", "Summer_Winter_Silver", "Summer_Winter_Bronze",
              "Summer_Winter_Total", "Num_Games", "Combined_Total"]

numeric_cols = ["Summer_Gold", "Summer_Silver", "Summer_Bronze", "Summer_Total",
                "Winter_Gold", "Winter_Silver", "Winter_Bronze", "Winter_Total", "Combined_Total"]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df_clean = df[numeric_cols].dropna()

print(f"Data count before outlier detection: {len(df_clean)}")

lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
labels = lof.fit_predict(df_clean)

df_no_outliers = df_clean[labels == 1]

print(f"Data count after outlier detection:  {len(df_no_outliers)}")
print(f"Outliers removed: {len(df_clean) - len(df_no_outliers)}")

plt.figure(figsize=(8, 5))
plt.scatter(df_clean["Summer_Total"], df_clean["Combined_Total"],
            c=["red" if l == -1 else "blue" for l in labels], alpha=0.7)
plt.xlabel("Summer Total")
plt.ylabel("Combined Total")
plt.title("Outlier Detection using LOF (Distance-Based)")
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Outlier'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Normal')
])
plt.tight_layout()
plt.savefig("practical3_outliers.png")
plt.show()