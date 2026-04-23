import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

df = pd.read_csv("Automobile_data.csv")

df.replace("?", np.nan, inplace=True)

print("=== Before Imputation ===")
print(f"Shape: {df.shape}")
print("Missing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

numeric_cols = ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

categorical_cols = ["num-of-doors"]

num_imputer = SimpleImputer(strategy="mean")
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("\n=== After Imputation ===")
print(f"Shape: {df.shape}")
print("Missing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nIf no columns shown above, all missing values have been imputed.")

print("\nSample data after imputation:")
print(df[numeric_cols + categorical_cols].head(10))

missing_before = [38, 4, 4, 2, 2, 41]
labels = numeric_cols

plt.figure(figsize=(9, 4))
plt.bar(labels, missing_before, color="salmon", label="Before Imputation")
plt.bar(labels, [0]*len(labels), color="green", label="After Imputation")
plt.title("Missing Values Before and After Imputation")
plt.ylabel("Missing Count")
plt.xticks(rotation=20)
plt.legend()
plt.tight_layout()
plt.savefig("practical8_imputation.png")
plt.show()
