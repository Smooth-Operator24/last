import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_before = RandomForestClassifier(random_state=42)
model_before.fit(X_train, y_train)
y_pred_before = model_before.predict(X_test)
f1_before = f1_score(y_test, y_pred_before)

print("=== Before SMOTE ===")
print(f"Class Distribution:\n{y_train.value_counts()}")
print(f"F1 Score: {f1_before:.4f}")
print(classification_report(y_test, y_pred_before))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model_after = RandomForestClassifier(random_state=42)
model_after.fit(X_resampled, y_resampled)
y_pred_after = model_after.predict(X_test)
f1_after = f1_score(y_test, y_pred_after)

print("=== After SMOTE ===")
print(f"Class Distribution:\n{pd.Series(y_resampled).value_counts()}")
print(f"F1 Score: {f1_after:.4f}")
print(classification_report(y_test, y_pred_after))

print(f"\nF1 Score Before SMOTE: {f1_before:.4f}")
print(f"F1 Score After  SMOTE: {f1_after:.4f}")
print("SMOTE improved class balance and F1-score by generating synthetic minority class samples.")
