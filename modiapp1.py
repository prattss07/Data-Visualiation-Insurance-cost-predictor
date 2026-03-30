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
# 🧠 SESSION STATE
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "user_profile" not in st.session_state:
    st.session_state.user_profile = None

# =============================
# 🔐 LOGIN PAGE
# =============================
if not st.session_state.logged_in:

    st.set_page_config(page_title="Login", layout="centered")
    st.title("🔐 Insurance System Login")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", key="login_btn"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login Successful ✅")
                st.rerun()
            else:
                st.error("Invalid Credentials ❌")

    with tab2:
        new_user = st.text_input("Username", key="signup_username")
        new_pass = st.text_input("Password", type="password", key="signup_password")
        confirm_pass = st.text_input("Confirm Password", type="password", key="signup_confirm")

        if st.button("Sign Up", key="signup_btn"):
            if user_exists(new_user):
                st.warning("User exists ⚠️")
            elif new_pass != confirm_pass:
                st.warning("Passwords do not match ⚠️")
            else:
                register_user(new_user, new_pass)
                st.success("Account created ✅")

    st.stop()

# =============================
# 📂 LOAD DATA
# =============================
df = pd.read_csv("insurance.csv")
df.drop_duplicates(inplace=True)

# =============================
# 🔧 TRAIN MODEL
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

# =============================
# 📂 LOAD MODEL
# =============================
model = pickle.load(open("model.pkl", "rb"))
feature_columns = pickle.load(open("columns.pkl", "rb"))

# =============================
# 🖥️ MAIN UI
# =============================
st.set_page_config(page_title="Insurance System", layout="wide")
st.title("💰 Insurance Analytics & Prediction System")

# Sidebar
st.sidebar.write(f"👋 Welcome, {st.session_state.username}")

if st.sidebar.button("Logout", key="logout_btn"):
    st.session_state.logged_in = False
    st.session_state.user_profile = None
    st.rerun()

mode = st.sidebar.radio(
    "Select Section",
    ["📊 Dashboard (EDA)", "👤 Prediction Mode"],
    key="mode_select"
)

# =============================
# 📊 DASHBOARD
# =============================
if mode == "📊 Dashboard (EDA)":
    st.header("📊 EDA")
    st.dataframe(df.head(), use_container_width=True)
    st.bar_chart(df["charges"])
    st.scatter_chart(df[["age", "charges"]])

# =============================
# 👤 PREDICTION MODE
# =============================
else:

    st.header("👤 User Details")

    if st.session_state.user_profile is None:

        user_name = st.text_input("Name", key="main_name")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 0, 100, 30, key="main_age")
            sex = st.selectbox("Sex", ["male", "female"], key="main_sex")

        with col2:
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0, key="main_bmi")
            region = st.selectbox(
                "Region",
                ["northeast", "northwest", "southeast", "southwest"],
                key="main_region"
            )

        if age >= 18:
            children = st.number_input("Children", 0, 10, 1, key="main_children")
            smoker = st.selectbox("Smoker", ["yes", "no"], key="main_smoker")
        else:
            children = 0
            smoker = "no"

        if st.button("Save Details", key="save_profile"):
            st.session_state.user_profile = {
                "name": user_name,
                "age": age,
                "bmi": bmi,
                "children": children,
                "sex": sex,
                "smoker": smoker,
                "region": region
            }
            st.success("Details saved ✅")
            st.rerun()

        st.stop()

    user_data = st.session_state.user_profile

    st.success(f"Using saved profile for {user_data['name']}")

    if st.button("Edit Details", key="edit_profile"):
        st.session_state.user_profile = None
        st.rerun()

    # =============================
    # 👨‍👩‍👧 FAMILY
    # =============================
    st.header("👨‍👩‍👧 Family Members")

    relation_gender_map = {
        "Father": "male",
        "Mother": "female",
        "Husband": "male",
        "Wife": "female",
        "Son": "male",
        "Daughter": "female",
    }

    add_family = st.checkbox("Add Family", key="add_family")

    family_data = []

    if add_family:
        n = st.number_input("Members", 1, 10, 1, key="family_count")

        for i in range(n):

            st.markdown(f"### Member {i+1}")

            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input("Name", key=f"name{i}")
                relation = st.selectbox(
                    "Relation",
                    ["Father", "Mother", "Wife", "Husband", "Son", "Daughter", "Other"],
                    key=f"rel{i}"
                )
                f_age = st.number_input("Age", 1, 100, 25, key=f"age{i}")

            auto_gender = relation_gender_map.get(relation, "male")

            with col2:
                f_bmi = st.number_input("BMI", 10.0, 50.0, 25.0, key=f"bmi{i}")
                f_sex = st.selectbox(
                    "Sex",
                    ["male", "female"],
                    index=0 if auto_gender == "male" else 1,
                    key=f"sex{i}"
                )

                if f_age >= 18:
                    f_children = st.number_input("Children", 0, 10, 0, key=f"child{i}")
                    f_smoker = st.selectbox("Smoker", ["yes", "no"], key=f"smoker{i}")
                else:
                    f_children = 0
                    f_smoker = "no"

            f_region = st.selectbox(
                "Region",
                ["northeast", "northwest", "southeast", "southwest"],
                key=f"region{i}"
            )

            family_data.append({
                "name": name,
                "relation": relation,
                "age": f_age,
                "bmi": f_bmi,
                "children": f_children,
                "sex": f_sex,
                "smoker": f_smoker,
                "region": f_region
            })

    # =============================
    # 🔮 PREDICTION
    # =============================
    def predict(data):
        df_input = pd.DataFrame([data])
        df_input = pd.get_dummies(df_input)
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)
        return model.predict(df_input)[0]

    if st.button("Calculate Insurance Cost", key="calc_btn"):

        total = 0

        st.subheader("💵 Cost Breakdown")

        cost = predict(user_data)
        total += cost
        st.write(f"👤 {user_data['name']}: ₹ {cost:,.2f}")

        for m in family_data:
            c = predict(m)
            total += c
            st.write(f"👨 {m['relation']} - {m['name']}: ₹ {c:,.2f}")

        st.success(f"💰 Total Cost: ₹ {total:,.2f}")