import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=== Iris Dataset Head ===")
print(df.head())
print(f"\nShape: {df.shape}")
print("\nSpecies Distribution:")
print(df["species"].value_counts())

plt.figure(figsize=(7, 5))
colors = {"setosa": "red", "versicolor": "green", "virginica": "blue"}
for species, group in df.groupby("species"):
    plt.scatter(group["sepal length (cm)"], group["petal length (cm)"],
                label=species, color=colors[species], alpha=0.7, edgecolors="k", s=60)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length - Iris Dataset")
plt.legend(title="Species")
plt.tight_layout()
plt.savefig("practical14_scatter.png")
plt.show()

r, p = stats.pearsonr(df["sepal length (cm)"], df["petal length (cm)"])
print(f"\nPearson Correlation (Sepal Length vs Petal Length):")
print(f"r = {r:.4f}, p-value = {p:.6f}")
print("Strong positive correlation" if r > 0.7 else "Moderate correlation")

slope, intercept, r_val, p_val, std_err = stats.linregress(
    df["sepal length (cm)"], df["petal length (cm)"])
plt.figure(figsize=(7, 5))
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"],
            c=iris.target, cmap="Set1", alpha=0.7, edgecolors="k", s=60)
x_line = df["sepal length (cm)"].sort_values()
plt.plot(x_line, slope * x_line + intercept, color="black", linewidth=2, label="Regression Line")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length with Regression Line")
plt.legend()
plt.tight_layout()
plt.savefig("practical14_scatter_regression.png")
plt.show()

print(f"\nRegression Line: y = {slope:.4f}x + {intercept:.4f}")
print(f"R² = {r_val**2:.4f}")
