import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# -----------------------------
# Train models (only once)
# -----------------------------
if not os.path.exists("models"):
    os.makedirs("models")

if not os.path.exists("models/lr.pkl"):

    np.random.seed(42)

    size = 1000
    age = np.random.randint(20, 80, size)
    bp = np.random.randint(90, 180, size)
    chol = np.random.randint(150, 300, size)
    hr = np.random.randint(90, 200, size)
    oldpeak = np.random.uniform(0, 4, size)

    df = pd.DataFrame({
        'age': age,
        'trestbps': bp,
        'chol': chol,
        'thalach': hr,
        'oldpeak': oldpeak
    })

    df['target'] = (
        (df['age'] > 50).astype(int) +
        (df['trestbps'] > 140).astype(int) +
        (df['chol'] > 240).astype(int) +
        (df['thalach'] < 140).astype(int) +
        (df['oldpeak'] > 2).astype(int)
    )

    df['target'] = (df['target'] >= 3).astype(int)

    X = df[['age','trestbps','chol','thalach','oldpeak']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    lr = CalibratedClassifierCV(LogisticRegression(max_iter=1000), method='sigmoid')
    dt = CalibratedClassifierCV(DecisionTreeClassifier(max_depth=5), method='sigmoid')
    rf = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, max_depth=5), method='sigmoid')

    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    pickle.dump(lr, open("models/lr.pkl","wb"))
    pickle.dump(dt, open("models/dt.pkl","wb"))
    pickle.dump(rf, open("models/rf.pkl","wb"))
    pickle.dump(scaler, open("models/scaler.pkl","wb"))

# -----------------------------
# Load models
# -----------------------------
lr = pickle.load(open("models/lr.pkl","rb"))
dt = pickle.load(open("models/dt.pkl","rb"))
rf = pickle.load(open("models/rf.pkl","rb"))
scaler = pickle.load(open("models/scaler.pkl","rb"))

# -----------------------------
# UI Header
# -----------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.caption("Enter patient details to assess cardiovascular risk")

# -----------------------------
# Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 30)
    bp = st.number_input("Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)

with col2:
    hr = st.number_input("Max Heart Rate", 60, 220, 150)
    oldpeak = st.number_input("Oldpeak", 0.0, 5.0, 1.0)
    model_choice = st.selectbox("Select Model",
                               ["Logistic Regression", "Decision Tree", "Random Forest"])

st.write("")

# -----------------------------
# Center Button
# -----------------------------
col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
with col_btn2:
    predict_clicked = st.button("🔍 Predict Risk")

# -----------------------------
# Prediction
# -----------------------------
if predict_clicked:

    features = np.array([[age, bp, chol, hr, oldpeak]])
    features = scaler.transform(features)

    if model_choice == "Logistic Regression":
        model = lr
    elif model_choice == "Decision Tree":
        model = dt
    else:
        model = rf

    probability = model.predict_proba(features)[0][1] * 100

    # Result
    if probability < 40:
        st.success(f"Low Risk ✅ ({probability:.2f}%)")

    elif probability < 70:
        st.warning(f"Moderate Risk ⚠️ ({probability:.2f}%)")

    else:
        st.error(f"High Risk 🔴 ({probability:.2f}%)")