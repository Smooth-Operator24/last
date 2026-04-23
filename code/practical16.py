import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("AirPassengers.csv")
df.columns = ["Month", "Passengers"]
df["Passengers"] = pd.to_numeric(df["Passengers"], errors="coerce")
df = df.dropna()

print("=== Dataset Head ===")
print(df.head())
print(f"Shape: {df.shape}")

series = df["Passengers"].values

def create_lag_features(data, n_lags=3):
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i - n_lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)

n_lags = 3
X, y = create_lag_features(series, n_lags)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\n=== Autoregression Model (Lags = {n_lags}) ===")
print(f"Coefficients: {model.coef_}")
print(f"Intercept:    {model.intercept_:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(range(len(series)), series, label="Actual", color="steelblue")
plt.plot(range(split + n_lags, len(series)), y_pred, label="AR Forecast", color="red", linestyle="--")
plt.axvline(x=split + n_lags, color="gray", linestyle=":", label="Train/Test Split")
plt.title(f"Autoregression Forecast (Lags={n_lags}) - Air Passengers")
plt.xlabel("Time Step")
plt.ylabel("Passengers")
plt.legend()
plt.tight_layout()
plt.savefig("practical16_ar_forecast.png")
plt.show()

future_input = series[-n_lags:].tolist()
future_preds = []
for _ in range(12):
    pred = model.predict([future_input[-n_lags:]])[0]
    future_preds.append(pred)
    future_input.append(pred)

plt.figure(figsize=(10, 5))
plt.plot(series, label="Historical", color="steelblue")
plt.plot(range(len(series), len(series) + 12), future_preds,
         label="Next 12 Months Forecast", color="orange", marker="o")
plt.title("Autoregression - Future 12 Months Forecast")
plt.xlabel("Time Step")
plt.ylabel("Passengers")
plt.legend()
plt.tight_layout()
plt.savefig("practical16_future_forecast.png")
plt.show()

print("\nNext 12 months predicted values:")
for i, val in enumerate(future_preds, 1):
    print(f"Month +{i}: {val:.2f}")
