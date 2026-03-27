# =============================
# 📦 IMPORT LIBRARIES
# =============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================
# 📂 LOAD DATASET
# =============================
df = pd.read_csv("insurance.csv")

# =============================
# 🧹 DATA CLEANING
# =============================
df.drop_duplicates(inplace=True)

# =============================
# 📊 EDA
# =============================

# Distribution
sns.histplot(df['charges'], kde=True)
plt.title("Distribution of Charges")
plt.show()

# Smoker vs Charges
sns.boxplot(x='smoker', y='charges', data=df)
plt.title("Smoker vs Charges")
plt.show()

# Age vs Charges
sns.scatterplot(x='age', y='charges', data=df)
plt.title("Age vs Charges")
plt.show()

# BMI vs Charges
sns.scatterplot(x='bmi', y='charges', data=df)
plt.title("BMI vs Charges")
plt.show()

# =============================
# 🔧 FEATURE ENGINEERING
# =============================
df_encoded = pd.get_dummies(df, drop_first=True)

# =============================
# 🎯 SPLIT DATA
# =============================
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

feature_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 🤖 TRAIN MODELS
# =============================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# =============================
# 📏 EVALUATION
# =============================
print("\n--- Linear Regression ---")
lr_pred = lr_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("MSE:", mean_squared_error(y_test, lr_pred))
print("R2:", r2_score(y_test, lr_pred))

print("\n--- Random Forest ---")
rf_pred = rf_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("MSE:", mean_squared_error(y_test, rf_pred))
print("R2:", r2_score(y_test, rf_pred))

# =============================
# 🏆 SELECT BEST MODEL
# =============================
best_model = rf_model

# =============================
# 💾 SAVE MODEL + FEATURES
# =============================
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("✅ Model and columns saved!")

# =============================
# 📈 ACTUAL VS PREDICTED
# =============================
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
