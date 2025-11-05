import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print("🔹 First 5 rows (train):")
print(train.head(), "\n")

print("🔹 Info (train):")
print(train.info(), "\n")

print("🔹 Missing values in train set:\n", train.isnull().sum(), "\n")

train["Age"].fillna(train["Age"].median(), inplace=True)

train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)

train.drop(columns=["Cabin"], inplace=True)

train["HasFamily"] = ((train["SibSp"] + train["Parch"]) > 0).astype(int)

train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
train = pd.get_dummies(train, columns=["Embarked", "Pclass"], drop_first=True)

print("✅ Missing values after cleaning:\n", train.isnull().sum(), "\n")

print("🔹 Summary statistics:\n", train.describe(), "\n")

survival_rate = train["Survived"].mean() * 100
print(f"🔹 Overall Survival Rate: {survival_rate:.2f}%\n")

sns.countplot(x="Survived", data=train)
plt.title("Survival Count (0 = Died, 1 = Survived)")
plt.show()

sns.countplot(x="Sex", hue="Survived", data=train)
plt.xticks([0,1], ["male", "female"])
plt.title("Survival by Sex")
plt.show()

sns.countplot(x="Pclass_2", hue="Survived", data=train)
plt.title("Survival by Pclass = 2 (vs class1 baseline)")
plt.show()

sns.countplot(x="Pclass_1", hue="Survived", data=train)
plt.title("Survival by Pclass = 1 (first class)")
plt.show()

sns.histplot(train["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x="Survived", y="Age", data=train)
plt.title("Age vs Survival")
plt.show()

corr_mat = train.corr()
sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

print("""
✅ Key Insights:
1. Female passengers had much higher survival chances than male passengers.
2. First-class passengers (Pclass = 1) survived at much higher rates than lower classes.
3. Younger passengers showed a modestly higher survival probability.
4. The presence of family (“HasFamily”) may show interesting patterns (you can explore further).
5. Features like fare, class and sex correlate with survival; “Embarked” and “HasFamily” might need more nuanced analysis.
""")
