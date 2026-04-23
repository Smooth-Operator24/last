import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("diabetes.csv")

print("=== Correlation Matrix ===")
corr = df.corr()
print(corr)

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap - Diabetes Dataset")
plt.tight_layout()
plt.savefig("practical13_heatmap.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x="Glucose", y="BMI", hue="Outcome", data=df, palette="coolwarm", alpha=0.7)
plt.title("Scatter Plot: Glucose vs BMI")
plt.tight_layout()
plt.savefig("practical13_scatter_glucose_bmi.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x="Age", y="Glucose", hue="Outcome", data=df, palette="Set1", alpha=0.7)
plt.title("Scatter Plot: Age vs Glucose")
plt.tight_layout()
plt.savefig("practical13_scatter_age_glucose.png")
plt.show()

r, p = stats.pearsonr(df["Glucose"], df["BMI"])
print(f"\nPearson Correlation (Glucose vs BMI): r = {r:.4f}, p = {p:.4f}")
print("Strong positive correlation" if r > 0.5 else ("Weak correlation" if abs(r) < 0.3 else "Moderate correlation"))

plt.figure(figsize=(14, 10))
sns.pairplot(df, hue="Outcome", palette="husl", plot_kws={"alpha": 0.5})
plt.suptitle("Pair Plot - Diabetes Dataset", y=1.02)
plt.savefig("practical13_pairplot.png")
plt.show()
