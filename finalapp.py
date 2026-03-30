# =============================
# 📦 IMPORT LIBRARIES
# =============================
import streamlit as st
import pandas as pd
import pickle
import os
import bcrypt
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =============================
# 🔐 AUTH SYSTEM
# =============================
USER_FILE = "users.csv"

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

# =============================
# 🔐 PASSWORD VALIDATION
# =============================
def is_strong_password(password):
    return (
        len(password) >= 8 and
        re.search(r"[A-Z]", password) and
        re.search(r"[a-z]", password) and
        re.search(r"[0-9]", password) and
        re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)
    )

# =============================
# 🧠 SESSION STATE
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "prefill_username" not in st.session_state:
    st.session_state.prefill_username = ""

if "signup_success" not in st.session_state:
    st.session_state.signup_success = False

if "signup_message" not in st.session_state:
    st.session_state.signup_message = ""

# =============================
# 🔐 LOGIN PAGE (UNCHANGED)
# =============================
if not st.session_state.logged_in:

    st.set_page_config(page_title="Login", layout="centered")
    st.title("🔐 Insurance System Login")

    tabs = st.tabs(["Login", "Sign Up"])

    with tabs[0]:
        if st.session_state.signup_success:
            st.success(st.session_state.signup_message)
            st.session_state.signup_success = False

        username = st.text_input("Username", value=st.session_state.prefill_username)
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.prefill_username = ""
                st.rerun()
            else:
                st.error("Invalid Credentials ❌")

    with tabs[1]:
        with st.form("signup_form", clear_on_submit=True):
            new_user = st.text_input("Username")
            new_pass = st.text_input("Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")

            submitted = st.form_submit_button("Sign Up")

            if submitted:
                if user_exists(new_user):
                    st.warning("User already exists ⚠️")
                elif new_pass != confirm_pass:
                    st.warning("Passwords do not match ⚠️")
                elif not is_strong_password(new_pass):
                    st.warning("Weak password ⚠️")
                else:
                    register_user(new_user, new_pass)
                    st.session_state.signup_success = True
                    st.session_state.signup_message = "Account created successfully!"
                    st.session_state.prefill_username = new_user
                    st.rerun()

    st.stop()

# =============================
# 📂 LOAD DATA
# =============================
df = pd.read_csv("insurance.csv")
df.drop_duplicates(inplace=True)

# =============================
# 🤖 TRAIN MODEL
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

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(feature_columns, open("columns.pkl", "wb"))

model = pickle.load(open("model.pkl", "rb"))
feature_columns = pickle.load(open("columns.pkl", "rb"))

# =============================
# 🖥️ MAIN UI
# =============================
st.set_page_config(page_title="Insurance System", layout="wide")
st.title("💰 Insurance System")

st.sidebar.write(f"👋 Welcome, {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

mode = st.sidebar.radio("Select Section", ["📊 Dashboard", "👤 Insurance Calculator"])

# =============================
# 🧠 LOGIC
# =============================
def calculate_risk(data):
    risk = 0
    if data["age"] > 50: risk += 2
    if data["smoker"] == "yes": risk += 3
    if data["bmi"] > 30: risk += 2
    if "None" not in data["disease"]: risk += 4
    if data["alcohol"] == "frequent": risk += 2
    if data["exercise"] == "low": risk += 2
    return risk

def calculate_premium(data):
    base = 5000
    risk = calculate_risk(data)

    if data["age"] > 60:
        base += 5000
    elif data["age"] > 45:
        base += 3000
    elif data["age"] > 30:
        base += 1500

    if data["smoker"] == "yes":
        base += 4000

    if data["bmi"] > 30:
        base += 2000

    if "None" not in data["disease"]:
        base += 5000

    premium = base + (risk * 1000)

    if risk <= 3:
        coverage = premium * 200
        level = "Low Risk 🟢"
    elif risk <= 7:
        coverage = premium * 150
        level = "Medium Risk 🟠"
    else:
        coverage = premium * 100
        level = "High Risk 🔴"

    return premium, coverage, level

# =============================
# 📊 DASHBOARD
# =============================
if mode == "📊 Dashboard":
    st.dataframe(df.head(), use_container_width=True)
    st.bar_chart(df.groupby("smoker")["charges"].mean())
    st.bar_chart(df.groupby("region")["charges"].mean())
    st.scatter_chart(df[["bmi", "charges"]])
    st.scatter_chart(df[["age", "charges"]])

# =============================
# 👤 INSURANCE CALCULATOR
# =============================
else:

    st.header("👤 User Details")

    name = st.text_input("Name")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 100, 30)
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)

    with col2:
        smoker = st.selectbox("Smoker", ["yes", "no"])
        alcohol = st.selectbox("Alcohol", ["no", "occasional", "frequent"])
        exercise = st.selectbox("Exercise", ["low", "medium", "high"])

    disease = st.multiselect("Diseases", ["Diabetes", "BP", "Heart Disease", "None"])

    # FAMILY
    st.header("👨‍👩‍👧 Family Members")

    add_family = st.checkbox("Add Family Members")
    family_data = []

    if add_family:
        n = st.number_input("Members", 1, 10, 1)

        for i in range(n):
            st.subheader(f"Member {i+1}")

            m_name = st.text_input("Name", key=f"name{i}")
            m_age = st.number_input("Age", 1, 100, 25, key=f"age{i}")
            m_bmi = st.number_input("BMI", 10.0, 50.0, 25.0, key=f"bmi{i}")
            m_smoker = st.selectbox("Smoker", ["yes", "no"], key=f"smoker{i}")

            m_disease = st.multiselect(
                "Diseases",
                ["Diabetes", "BP", "Heart Disease", "None"],
                key=f"disease{i}"
            )

            family_data.append({
                "name": m_name,
                "age": m_age,
                "bmi": m_bmi,
                "smoker": m_smoker,
                "disease": m_disease,
                "alcohol": "no",
                "exercise": "medium"
            })

    if st.button("Calculate Insurance"):

        total_premium = 0
        total_coverage = 0

        user_data = {
            "name": name,
            "age": age,
            "bmi": bmi,
            "smoker": smoker,
            "disease": disease,
            "alcohol": alcohol,
            "exercise": exercise
        }

        p, c, r = calculate_premium(user_data)

        total_premium += p
        total_coverage += c

        st.write(f"👤 {name}: ₹{p:,} | Coverage ₹{c:,} | {r}")

        for m in family_data:
            p, c, r = calculate_premium(m)
            total_premium += p
            total_coverage += c
            st.write(f"👨 {m['name']}: ₹{p:,} | Coverage ₹{c:,} | {r}")

        if len(family_data) > 0:
            total_premium *= 0.9

        st.success(f"Total Premium: ₹{int(total_premium):,}/year")
        st.success(f"Total Coverage: ₹{int(total_coverage):,}")