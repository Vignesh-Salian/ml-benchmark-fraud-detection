import streamlit as st
import numpy as np
import pickle
import pandas as pd

@st.cache_resource
def load_models():
    try:
        with open("models/fraud_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        return None, None

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Model Comparison", "Fraud Prediction"]
)

# ==========================
# PAGE 1 — MODEL COMPARISON
# ==========================

if page == "Model Comparison":

    st.title("📊 Machine Learning Model Benchmarking")

    data = {
        "Model":[
            "Random Forest",
            "XGBoost",
            "LightGBM",
            "KNN",
            "Decision Tree",
            "Logistic Regression",
            "Naive Bayes"
        ],

        "Accuracy":[
            0.999456,
            0.999245,
            0.998227,
            0.998051,
            0.997174,
            0.974264,
            0.973842
        ],

        "F1 Score":[
            0.837696,
            0.801843,
            0.624535,
            0.607774,
            0.485623,
            0.109356,
            0.103490
        ]
    }

    df = pd.DataFrame(data)

    st.subheader("Model Performance Table")
    st.dataframe(df)

    best_model = df.sort_values("F1 Score", ascending=False).iloc[0]

    st.success(
        f"🏆 Best Model: {best_model['Model']} (F1 Score: {best_model['F1 Score']:.3f})"
    )

    st.subheader("Model Comparison Chart")
    st.bar_chart(df.set_index("Model")["F1 Score"])


# ==========================
# PAGE 2 — FRAUD PREDICTION
# ==========================

if page == "Fraud Prediction":

    model, scaler = load_models()

    st.title("💳 Credit Card Fraud Detection System")

    if model is None or scaler is None:
        st.error("⚠️ Error loading models. Please ensure 'models/fraud_model.pkl' and 'models/scaler.pkl' exist.")
        st.stop()

    st.write("Enter transaction feature values to predict fraud.")

    # Transaction Time
    time = st.number_input("Transaction Time", value=0.0)

    # V1 to V28 features
    st.subheader("Transaction Features (V1 - V28)")
    features = []

    cols = st.columns(4)
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        with cols[col_idx]:
            value = st.number_input(f"V{i}", value=0.0)
            features.append(value)

    # Transaction Amount
    amount = st.number_input("Transaction Amount", value=0.0)

    if st.button("Predict Transaction"):

        # scale Time and Amount separately
        time_scaled = scaler.transform([[time]])[0][0]
        amount_scaled = scaler.transform([[amount]])[0][0]

        # correct feature order
        input_data = [time_scaled] + features + [amount_scaled]

        input_array = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]

        st.write(f"Fraud Probability: {probability*100:.2f}%")

        if prediction == 1:
            st.error("⚠️ Fraudulent Transaction Detected")
        else:
            st.success("✅ Normal Transaction")