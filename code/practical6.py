import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("housing.csv", header=None)
df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
              "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

print("Dataset Head:")
print(df.head())
print(f"\nShape: {df.shape}")
print("\nMissing values:\n", df.isnull().sum())

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== Performance Evaluation Metrics ===")
print(f"Mean Absolute Error (MAE):  {mae:.4f}")
print(f"Mean Squared Error (MSE):   {mse:.4f}")
print(f"Root Mean Squared Error:    {rmse:.4f}")
print(f"R² Score:                   {r2:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="steelblue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("practical6_actual_vs_pred.png")
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=30, color="salmon", edgecolor="black")
plt.title("Residuals Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("practical6_residuals.png")
plt.show()
