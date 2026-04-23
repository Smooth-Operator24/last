import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("2018-2019_Daily_Attendance_20240429.csv")

print("=== Dataset Head ===")
print(df.head())
print(f"\nShape: {df.shape}")
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())

df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")

daily = df.groupby("Date")[["Enrolled", "Absent", "Present"]].sum().reset_index()
daily = daily.sort_values("Date")
daily["Attendance_Rate"] = (daily["Present"] / daily["Enrolled"]) * 100

print("\n=== Daily Attendance Summary ===")
print(daily.head(10))

plt.figure(figsize=(14, 5))
plt.plot(daily["Date"], daily["Attendance_Rate"], color="steelblue", linewidth=1)
plt.title("Daily Attendance Rate Over Time (2018-2019)")
plt.xlabel("Date")
plt.ylabel("Attendance Rate (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("practical17_attendance_rate.png")
plt.show()

monthly = daily.set_index("Date").resample("ME")[["Present", "Absent", "Enrolled"]].sum()
monthly["Attendance_Rate"] = (monthly["Present"] / monthly["Enrolled"]) * 100

plt.figure(figsize=(12, 5))
plt.plot(monthly.index, monthly["Attendance_Rate"], marker="o", color="darkorange", linewidth=2)
plt.title("Monthly Attendance Rate")
plt.xlabel("Month")
plt.ylabel("Attendance Rate (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("practical17_monthly_rate.png")
plt.show()

plt.figure(figsize=(12, 5))
plt.bar(monthly.index, monthly["Absent"], color="salmon", label="Absent", width=15)
plt.bar(monthly.index, monthly["Present"], color="steelblue", label="Present",
        bottom=monthly["Absent"], width=15)
plt.title("Monthly Present vs Absent Students")
plt.xlabel("Month")
plt.ylabel("Count")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("practical17_present_vs_absent.png")
plt.show()

ts = daily.set_index("Date")["Attendance_Rate"]
ts = ts[ts > 0].dropna()

if len(ts) >= 20:
    decompose_result = seasonal_decompose(ts, model="additive", period=5)
    fig, axes = plt.subplots(4, 1, figsize=(14, 8))
    decompose_result.observed.plot(ax=axes[0], title="Observed")
    decompose_result.trend.plot(ax=axes[1], title="Trend")
    decompose_result.seasonal.plot(ax=axes[2], title="Seasonal")
    decompose_result.resid.plot(ax=axes[3], title="Residual")
    plt.suptitle("Time Series Decomposition - Attendance Rate", fontsize=13)
    plt.tight_layout()
    plt.savefig("practical17_decomposition.png")
    plt.show()

print("\n=== Key Observations ===")
print(f"Average Daily Attendance Rate: {daily['Attendance_Rate'].mean():.2f}%")
print(f"Lowest Attendance Rate:  {daily['Attendance_Rate'].min():.2f}%  on {daily.loc[daily['Attendance_Rate'].idxmin(), 'Date'].date()}")
print(f"Highest Attendance Rate: {daily['Attendance_Rate'].max():.2f}%  on {daily.loc[daily['Attendance_Rate'].idxmax(), 'Date'].date()}")
