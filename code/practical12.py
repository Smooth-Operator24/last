import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Automobile_data.csv")
df.replace("?", np.nan, inplace=True)

numeric_cols = ["normalized-losses", "wheel-base", "length", "width", "height",
                "curb-weight", "engine-size", "bore", "stroke",
                "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("=== Shape ===")
print(df.shape)

print("\n=== Head ===")
print(df.head())

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Descriptive Statistics ===")
print(df[numeric_cols].describe())

plt.figure(figsize=(6, 4))
df["fuel-type"].value_counts().plot(kind="bar", color=["steelblue", "coral"], edgecolor="black")
plt.title("Fuel Type Distribution")
plt.tight_layout()
plt.savefig("practical12_fueltype.png")
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(df["price"].dropna(), bins=20, color="skyblue", edgecolor="black")
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("practical12_price_hist.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x="fuel-type", y="price", data=df, palette="Set2")
plt.title("Price by Fuel Type")
plt.tight_layout()
plt.savefig("practical12_boxplot.png")
plt.show()

plt.figure(figsize=(10, 7))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap - Automobile")
plt.tight_layout()
plt.savefig("practical12_heatmap.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x="engine-size", y="price", data=df, hue="fuel-type", palette="deep")
plt.title("Engine Size vs Price")
plt.tight_layout()
plt.savefig("practical12_scatter.png")
plt.show()

print("\n=== Skewness ===")
print(df[numeric_cols].skew())
print("\n=== Kurtosis ===")
print(df[numeric_cols].kurt())
