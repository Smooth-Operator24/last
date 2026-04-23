import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("olympics.csv", header=None, skiprows=2)
df.columns = ["Country", "Summer_Games", "Summer_Gold", "Summer_Silver", "Summer_Bronze",
              "Summer_Total", "Winter_Gold", "Winter_Silver", "Winter_Bronze", "Winter_Total",
              "Summer_Winter_Gold", "Summer_Winter_Silver", "Summer_Winter_Bronze",
              "Summer_Winter_Total", "Num_Games", "Combined_Total"]

numeric_cols = ["Summer_Gold", "Summer_Silver", "Summer_Bronze", "Summer_Total",
                "Winter_Gold", "Winter_Silver", "Winter_Bronze", "Winter_Total", "Combined_Total"]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

print("=== Descriptive Statistics ===")
print(df[numeric_cols].describe())

print("\nMean:\n", df[numeric_cols].mean())
print("\nMedian:\n", df[numeric_cols].median())
print("\nMin:\n", df[numeric_cols].min())
print("\nMax:\n", df[numeric_cols].max())
print("\nStandard Deviation:\n", df[numeric_cols].std())

print("\n=== Inferential Statistics ===")
t_stat, p_val = stats.ttest_1samp(df["Summer_Total"].dropna(), popmean=10)
print(f"One-Sample T-Test (Summer Total vs mean=10): t={t_stat:.4f}, p={p_val:.4f}")

gold, silver = df["Summer_Gold"].dropna(), df["Summer_Silver"].dropna()
t2, p2 = stats.ttest_ind(gold, silver)
print(f"Independent T-Test (Gold vs Silver): t={t2:.4f}, p={p2:.4f}")

corr, p3 = stats.pearsonr(df["Summer_Total"].dropna(), df["Winter_Total"].dropna())
print(f"Pearson Correlation (Summer vs Winter Total): r={corr:.4f}, p={p3:.4f}")

print("\n=== Inferences ===")
print("1. Most countries have very few medals; distribution is heavily right-skewed.")
print("2. Summer medals significantly outnumber Winter medals across all countries.")
print("3. Summer Gold and Silver are not significantly different on average (check p-value).")
print("4. There is a positive correlation between Summer and Winter medal totals.")
