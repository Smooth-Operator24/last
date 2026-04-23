import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc, classification_report)
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=== 1. Hold-Out Validation (80-20 Split) ===")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

print("=== 2. K-Fold Cross Validation (k=5) ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr, X_scaled, y, cv=kf, scoring="accuracy")
print(f"Fold Accuracies: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")

print("\n=== 3. Stratified K-Fold (k=5) ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_scores = cross_val_score(lr, X_scaled, y, cv=skf, scoring="f1")
print(f"Fold F1 Scores: {skf_scores}")
print(f"Mean F1: {skf_scores.mean():.4f}, Std: {skf_scores.std():.4f}")

print("\n=== 4. Comparing Multiple Models ===")
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    results[name] = scores
    print(f"{name}: Mean={scores.mean():.4f}, Std={scores.std():.4f}")

plt.figure(figsize=(8, 5))
plt.boxplot(results.values(), labels=results.keys())
plt.title("Model Comparison - 5-Fold Cross Validation Accuracy")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("practical15_model_comparison.png")
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("practical15_confusion.png")
plt.show()

y_prob = lr.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Diabetes Dataset")
plt.legend()
plt.tight_layout()
plt.savefig("practical15_roc.png")
plt.show()
