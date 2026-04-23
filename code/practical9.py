import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("placement.csv")
df = df.dropna()

print("Dataset Head:")
print(df.head())

plt.figure(figsize=(6, 4))
df["placed"].value_counts().plot(kind="bar", color=["coral", "steelblue"], edgecolor="black")
plt.title("Placement Count")
plt.xticks([0, 1], ["Not Placed (0)", "Placed (1)"], rotation=0)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("practical9_bar.png")
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(df["cgpa"], bins=20, color="skyblue", edgecolor="black")
plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("practical9_hist_cgpa.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x="placed", y="cgpa", data=df, palette="Set2")
plt.title("CGPA vs Placement Status")
plt.xticks([0, 1], ["Not Placed", "Placed"])
plt.tight_layout()
plt.savefig("practical9_boxplot.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x="cgpa", y="placement_exam_marks", hue="placed", data=df, palette="coolwarm")
plt.title("CGPA vs Placement Exam Marks")
plt.legend(title="Placed", labels=["Not Placed", "Placed"])
plt.tight_layout()
plt.savefig("practical9_scatter.png")
plt.show()

plt.figure(figsize=(5, 4))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("practical9_heatmap.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.violinplot(x="placed", y="placement_exam_marks", data=df, palette="muted")
plt.title("Placement Exam Marks by Placement Status")
plt.xticks([0, 1], ["Not Placed", "Placed"])
plt.tight_layout()
plt.savefig("practical9_violin.png")
plt.show()
