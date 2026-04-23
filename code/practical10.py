import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes.csv")

print("Dataset Head:")
print(df.head())
print(f"\nShape: {df.shape}")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(14, 6))
df[numeric_cols].boxplot()
plt.title("Box Plot - Diabetes Dataset (All Features)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("practical10_boxplot_all.png")
plt.show()

print("\n=== Outliers Detected per Column (IQR Method) ===")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers (lower={lower:.2f}, upper={upper:.2f})")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(col)
plt.suptitle("Individual Box Plots - Diabetes Dataset")
plt.tight_layout()
plt.savefig("practical10_boxplots_individual.png")
plt.show()
