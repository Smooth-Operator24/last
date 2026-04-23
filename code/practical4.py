import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv("international-airline-passengers.csv")
df.columns = ["Month", "Passengers"]
df = df.dropna()
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df["Passengers"] = pd.to_numeric(df["Passengers"], errors="coerce")
df = df.dropna()

decomposition = seasonal_decompose(df["Passengers"], model="multiplicative", period=12)

fig, axes = plt.subplots(4, 1, figsize=(12, 8))
decomposition.observed.plot(ax=axes[0], title="Observed")
decomposition.trend.plot(ax=axes[1], title="Trend")
decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
decomposition.resid.plot(ax=axes[3], title="Residual")
plt.tight_layout()
plt.savefig("practical4_decomposition.png")
plt.show()

model = ExponentialSmoothing(df["Passengers"], trend="add", seasonal="mul", seasonal_periods=12)
fit = model.fit()
forecast = fit.forecast(24)

plt.figure(figsize=(12, 5))
plt.plot(df["Passengers"], label="Actual")
plt.plot(forecast, label="Forecast (24 months)", linestyle="--", color="red")
plt.title("Holt-Winters Forecasting - International Airline Passengers")
plt.xlabel("Month")
plt.ylabel("Passengers (thousands)")
plt.legend()
plt.tight_layout()
plt.savefig("practical4_forecast.png")
plt.show()

print("Forecasted values for next 24 months:")
print(forecast)
