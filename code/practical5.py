import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

print("=== STEP 1: Data Collection ===")
df = pd.read_csv("placement.csv")
print(df.head())
print(f"Shape: {df.shape}")

print("\n=== STEP 2: Data Understanding / EDA ===")
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

plt.figure(figsize=(6, 4))
df["placed"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
plt.title("Placement Distribution")
plt.xticks([0, 1], ["Not Placed", "Placed"], rotation=0)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("practical5_distribution.png")
plt.show()

print("\n=== STEP 3: Data Preprocessing ===")
df = df.dropna()
scaler = StandardScaler()
X = df[["cgpa", "placement_exam_marks"]]
y = df["placed"]
X_scaled = scaler.fit_transform(X)
print("Data after scaling (first 5 rows):")
print(pd.DataFrame(X_scaled, columns=["cgpa", "placement_exam_marks"]).head())

print("\n=== STEP 4: Train-Test Split ===")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

print("\n=== STEP 5: Model Building ===")
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model trained: Logistic Regression")

print("\n=== STEP 6: Model Evaluation ===")
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("practical5_confusion.png")
plt.show()

print("\n=== STEP 7: Insights / Deployment ===")
print("CGPA and placement exam marks are strong predictors of placement.")
print("Model can be deployed to predict placement chances for new students.")
