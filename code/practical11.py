import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("diabetes.csv")

print("=== 1. One-Sample T-Test ===")
t_stat, p_val = stats.ttest_1samp(df["Glucose"], popmean=120)
print(f"Test: Is mean Glucose significantly different from 120?")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
print("Result:", "Reject H0 (significant)" if p_val < 0.05 else "Fail to Reject H0")

print("\n=== 2. Two-Sample Independent T-Test ===")
group0 = df[df["Outcome"] == 0]["BMI"]
group1 = df[df["Outcome"] == 1]["BMI"]
t2, p2 = stats.ttest_ind(group0, group1)
print(f"Test: Is BMI different between diabetic and non-diabetic groups?")
print(f"t-statistic: {t2:.4f}, p-value: {p2:.4f}")
print("Result:", "Reject H0 (significant)" if p2 < 0.05 else "Fail to Reject H0")

print("\n=== 3. Chi-Square Test ===")
df["Age_Group"] = pd.cut(df["Age"], bins=[0, 30, 50, 100], labels=["Young", "Middle", "Senior"])
contingency = pd.crosstab(df["Age_Group"], df["Outcome"])
chi2, p3, dof, expected = stats.chi2_contingency(contingency)
print(f"Test: Is Age Group associated with Diabetes Outcome?")
print(f"Chi2: {chi2:.4f}, p-value: {p3:.4f}, Degrees of Freedom: {dof}")
print("Result:", "Reject H0 (association exists)" if p3 < 0.05 else "Fail to Reject H0")

print("\n=== 4. Pearson Correlation ===")
corr, p4 = stats.pearsonr(df["Glucose"], df["Outcome"])
print(f"Test: Is there a linear correlation between Glucose and Outcome?")
print(f"Pearson r: {corr:.4f}, p-value: {p4:.4f}")
print("Result:", "Significant correlation" if p4 < 0.05 else "No significant correlation")

print("\n=== 5. ANOVA (One-Way) ===")
young = df[df["Age_Group"] == "Young"]["Glucose"]
middle = df[df["Age_Group"] == "Middle"]["Glucose"]
senior = df[df["Age_Group"] == "Senior"]["Glucose"]
f_stat, p5 = stats.f_oneway(young, middle, senior)
print(f"Test: Is mean Glucose different across Age Groups?")
print(f"F-statistic: {f_stat:.4f}, p-value: {p5:.4f}")
print("Result:", "Reject H0 (group means differ)" if p5 < 0.05 else "Fail to Reject H0")

print("\n=== 6. Confidence Interval (95%) for Glucose ===")
mean = df["Glucose"].mean()
se = stats.sem(df["Glucose"])
ci = stats.t.interval(0.95, len(df["Glucose"])-1, loc=mean, scale=se)
print(f"Mean Glucose: {mean:.4f}")
print(f"95% Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})")
