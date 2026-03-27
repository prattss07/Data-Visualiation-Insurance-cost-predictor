# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------
# 1. Load dataset
# ---------------------------
df = pd.read_csv("D:\\PACKAGES\\SEM4\\DV\\insurance.csv")

# ---------------------------
# 2. Basic Data Exploration
# ---------------------------
print(df.head())
print(df.info())

# ---------------------------
# 3. Data Cleaning
# ---------------------------
df.drop_duplicates(inplace=True)

# ---------------------------
# 4. Exploratory Data Analysis (EDA)
# ---------------------------

# Distribution of charges
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

# ---------------------------
# 5. Feature Engineering
# ---------------------------
df_encoded = pd.get_dummies(df, drop_first=True)

# Save column structure
feature_columns = df_encoded.drop("charges", axis=1).columns

# ---------------------------
# 6. Train Model
# ---------------------------
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------
# 7. Evaluation
# ---------------------------
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ---------------------------
# 8. USER INPUT
# ---------------------------
print("\n--- Enter Details for Prediction ---")

age = int(input("Enter Age: "))
sex = input("Enter Sex (male/female): ").lower()
bmi = float(input("Enter BMI: "))
children = int(input("Enter Number of Children: "))
smoker = input("Smoker? (yes/no): ").lower()
region = input("Region (northeast/northwest/southeast/southwest): ").lower()

# Create input dictionary
input_dict = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex": sex,
    "smoker": smoker,
    "region": region
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Encode input
input_encoded = pd.get_dummies(input_df)

# Match training columns
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# ---------------------------
# 9. Prediction
# ---------------------------
prediction = model.predict(input_encoded)

print(f"\n💰 Predicted Insurance Cost: {prediction[0]:.2f}")

# ---------------------------
# 10. Save User Input + Prediction
# ---------------------------

# Add prediction to dictionary
input_dict["predicted_charges"] = prediction[0]

# Convert to DataFrame
new_entry = pd.DataFrame([input_dict])

file_name = "user_inputs.csv"

try:
    existing_data = pd.read_csv(file_name)
    updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
    updated_data.to_csv(file_name, index=False)
except FileNotFoundError:
    new_entry.to_csv(file_name, index=False)

print("✅ Data saved to user_inputs.csv")

# ---------------------------
# 11. Visualization: Actual vs Predicted
# ---------------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted")
plt.show()