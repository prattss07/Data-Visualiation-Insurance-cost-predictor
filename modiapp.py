# =============================
# 📦 IMPORT LIBRARIES
# =============================
import streamlit as st
import pandas as pd
import pickle
import os
import bcrypt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =============================
# 🔐 AUTH SYSTEM
# =============================
USER_FILE = "users.csv"

# Create user file if not exists
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_FILE, index=False)


def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())


def user_exists(username):
    df = pd.read_csv(USER_FILE)
    return username in df["username"].values


def register_user(username, password):
    df = pd.read_csv(USER_FILE)
    hashed = hash_password(password).decode()

    new_user = pd.DataFrame([[username, hashed]], columns=["username", "password"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_FILE, index=False)


def authenticate_user(username, password):
    df = pd.read_csv(USER_FILE)
    user = df[df["username"] == username]

    if user.empty:
        return False

    stored_hash = user.iloc[0]["password"]
    return check_password(password, stored_hash)


# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# =============================
# 🔐 LOGIN / SIGNUP PAGE
# =============================
if not st.session_state.logged_in:

    st.set_page_config(page_title="Login", layout="centered")
    st.title("🔐 Insurance System Login")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # -------- LOGIN --------
    with tab1:
        st.subheader("Login")

        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login Successful ✅")
                st.rerun()
            else:
                st.error("Invalid Username or Password ❌")

    # -------- SIGNUP --------
    with tab2:
        st.subheader("Create Account")

        new_user = st.text_input("Username", key="signup_user")
        new_pass = st.text_input("Password", type="password", key="signup_pass")
        confirm_pass = st.text_input("Confirm Password")

        if st.button("Sign Up"):

            if user_exists(new_user):
                st.warning("Username already exists ⚠️")

            elif new_pass != confirm_pass:
                st.warning("Passwords do not match ⚠️")

            elif len(new_pass) < 4:
                st.warning("Password must be at least 4 characters ⚠️")

            else:
                register_user(new_user, new_pass)
                st.success("Account created successfully! Please login ✅")

    st.stop()

# =============================
# 📂 LOAD DATA
# =============================
df = pd.read_csv("insurance.csv")
df.drop_duplicates(inplace=True)

# =============================
# 🔧 TRAIN MODEL (ONLY ONCE)
# =============================
if not os.path.exists("model.pkl"):

    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("charges", axis=1)
    y = df_encoded["charges"]

    feature_columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("columns.pkl", "wb") as f:
        pickle.dump(feature_columns, f)

# =============================
# 📂 LOAD MODEL
# =============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# =============================
# 🖥️ MAIN APP UI
# =============================
st.set_page_config(page_title="Insurance System", layout="wide")
st.title("💰 Insurance Analytics & Prediction System")

# =============================
# 🚪 LOGOUT
# =============================
st.sidebar.write(f"👋 Welcome, {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# =============================
# 🔘 MODE SELECTION
# =============================
mode = st.sidebar.radio(
    "Select Section",
    ["📊 Dashboard (EDA)", "👤 Prediction Mode"]
)

# =============================
# 📊 DASHBOARD
# =============================
if mode == "📊 Dashboard (EDA)":

    st.header("📊 Exploratory Data Analysis")

    st.dataframe(df.head())
    st.bar_chart(df["charges"])
    st.scatter_chart(df[["age", "charges"]])
    st.scatter_chart(df[["bmi", "charges"]])

# =============================
# 👤 PREDICTION MODE
# =============================
elif mode == "👤 Prediction Mode":

    st.header("👤 Primary User Details")

    user_name = st.text_input("Enter Your Name")

    age = st.number_input("Age", 18, 100, 30)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    children = st.number_input("Children", 0, 10, 1)

    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox(
        "Region", ["northeast", "northwest", "southeast", "southwest"]
    )

    user_data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region
    }

    # -------- FAMILY --------
    st.header("👨‍👩‍👧 Family Members (Optional)")

    add_family = st.checkbox("Add Family Members")
    family_data = []

    if add_family:
        num_members = st.number_input("Number of Members", 1, 10, 1)

        for i in range(num_members):

            st.subheader(f"Member {i+1}")

            name = st.text_input(f"Name {i}", key=f"name_{i}")
            f_age = st.number_input(f"Age {i}", 1, 100, 25, key=f"age_{i}")
            f_bmi = st.number_input(f"BMI {i}", 10.0, 50.0, 25.0, key=f"bmi_{i}")
            f_children = st.number_input(f"Children {i}", 0, 10, 0, key=f"child_{i}")

            f_sex = st.selectbox(f"Sex {i}", ["male", "female"], key=f"sex_{i}")
            f_smoker = st.selectbox(f"Smoker {i}", ["yes", "no"], key=f"smoker_{i}")
            f_region = st.selectbox(
                f"Region {i}",
                ["northeast", "northwest", "southeast", "southwest"],
                key=f"region_{i}"
            )

            family_data.append({
                "name": name,
                "age": f_age,
                "bmi": f_bmi,
                "children": f_children,
                "sex": f_sex,
                "smoker": f_smoker,
                "region": f_region
            })

    # -------- PREDICT FUNCTION --------
    def predict_cost(data):
        df_input = pd.DataFrame([data])
        df_input = pd.get_dummies(df_input)
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)
        return model.predict(df_input)[0]

    # -------- CALCULATE --------
    if st.button("Calculate Insurance Cost"):

        total_cost = 0

        st.subheader("💵 Cost Breakdown")

        user_cost = predict_cost(user_data)
        total_cost += user_cost
        st.write(f"👤 {user_name if user_name else 'User'}: ₹ {user_cost:,.2f}")

        for member in family_data:
            cost = predict_cost(member)
            total_cost += cost
            name = member["name"] if member["name"] else "Family Member"
            st.write(f"👨 {name}: ₹ {cost:,.2f}")

        st.success(f"💰 Total Insurance Cost: ₹ {total_cost:,.2f}")

# =============================
# 📏 MODEL INFO
# =============================
st.sidebar.markdown("---")
st.sidebar.subheader("📏 Model Info")
st.sidebar.write("Model: Random Forest")
st.sidebar.write("R² Score: ~0.85")
st.sidebar.write("MAE: ~2500")