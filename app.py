# =============================
# 📦 IMPORT LIBRARIES
# =============================
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# =============================
# 📂 LOAD MODEL + COLUMNS
# =============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

df = pd.read_csv("insurance.csv")

# =============================
# 📂 LOAD DATA
# =============================
df = pd.read_csv("insurance.csv")

# =============================
# 🖥️ UI
# =============================
st.set_page_config(page_title="Insurance Predictor", layout="wide")

st.title("💰 Insurance Cost Prediction Dashboard")

# =============================
# 🧑‍💻 USER INPUT
# =============================
st.sidebar.header("User Input")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Children", 0, 5, 1)

sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox(
    "Region", ["northeast", "northwest", "southeast", "southwest"]
)

# =============================
# 🔄 PREPARE INPUT
# =============================
input_dict = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex": sex,
    "smoker": smoker,
    "region": region
}

input_df = pd.DataFrame([input_dict])

input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# =============================
# 💰 PREDICTION
# =============================
if st.sidebar.button("Predict"):
    prediction = model.predict(input_encoded)[0]
    st.subheader("Predicted Insurance Cost")
    st.success(f"₹ {prediction:,.2f}")

# =============================
# 📊 VISUALS (PLOTLY)
# =============================
st.subheader("Data Visualizations")

fig1 = px.histogram(df, x="charges", title="Charges Distribution")
st.plotly_chart(fig1)

fig2 = px.box(df, x="smoker", y="charges", title="Smoker vs Charges")
st.plotly_chart(fig2)

fig3 = px.scatter(df, x="age", y="charges", title="Age vs Charges")
st.plotly_chart(fig3)

fig4 = px.scatter(df, x="bmi", y="charges", title="BMI vs Charges")
st.plotly_chart(fig4)

# =============================
# 📏 MODEL INFO
# =============================
st.subheader("Model Performance")
st.write("R² Score: ~0.85")
st.write("MAE: ~2500")
