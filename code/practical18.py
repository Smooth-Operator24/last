import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("Student_Marks.csv")

print("=== Dataset Head ===")
print(df.head(10))
print(f"\nShape: {df.shape}")
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

print("\n=== Descriptive Statistics ===")
print(df.describe())

for col in ["Marks", "time_study", "number_courses"]:
    mean   = df[col].mean()
    median = df[col].median()
    mode   = df[col].mode()[0]
    std    = df[col].std()
    skew   = df[col].skew()
    kurt   = df[col].kurt()
    print(f"\n--- {col} ---")
    print(f"  Mean:     {mean:.4f}")
    print(f"  Median:   {median:.4f}")
    print(f"  Mode:     {mode:.4f}")
    print(f"  Std Dev:  {std:.4f}")
    print(f"  Skewness: {skew:.4f}")
    print(f"  Kurtosis: {kurt:.4f}")

print("\n=== Normality Check (Shapiro-Wilk Test) ===")
for col in ["Marks", "time_study"]:
    stat, p = stats.shapiro(df[col])
    print(f"{col}: W={stat:.4f}, p={p:.4f} => {'Normal distribution' if p > 0.05 else 'Not normal (skewed)'}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(["Marks", "time_study", "number_courses"]):
    axes[i].hist(df[col], bins=15, color="skyblue", edgecolor="black")
    axes[i].axvline(df[col].mean(), color="red", linestyle="--", label="Mean")
    axes[i].axvline(df[col].median(), color="green", linestyle="-.", label="Median")
    axes[i].set_title(f"{col} Distribution")
    axes[i].legend()
plt.tight_layout()
plt.savefig("practical18_histograms.png")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
stats.probplot(df["Marks"], dist="norm", plot=axes[0])
axes[0].set_title("Q-Q Plot - Marks")
stats.probplot(df["time_study"], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot - Time Study")
plt.tight_layout()
plt.savefig("practical18_qqplot.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x="time_study", y="Marks", data=df, color="darkorange", alpha=0.7)
plt.title("Study Time vs Marks")
plt.xlabel("Time Studied (hours)")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("practical18_scatter.png")
plt.show()

plt.figure(figsize=(8, 4))
df[["Marks", "time_study"]].boxplot()
plt.title("Box Plot - Marks and Study Time")
plt.tight_layout()
plt.savefig("practical18_boxplot.png")
plt.show()

print("\n=== Inference ===")
skew_val = df["Marks"].skew()
if abs(skew_val) < 0.5:
    print("Marks distribution is approximately Normal (symmetric).")
elif skew_val > 0:
    print(f"Marks distribution is Positively Skewed (skew={skew_val:.4f}): more students scored lower marks.")
else:
    print(f"Marks distribution is Negatively Skewed (skew={skew_val:.4f}): more students scored higher marks.")
