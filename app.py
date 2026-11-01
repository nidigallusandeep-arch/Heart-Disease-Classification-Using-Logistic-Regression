import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

DATA_PATH = Path("heart_disease_prediction_2026.csv")
MODEL_PATH = Path("heart_disease_model.joblib")

st.title("❤️ Heart Disease Prediction")
st.caption("Educational machine-learning demonstration using Logistic Regression.")
st.warning(
    "This application is for educational purposes only. It is not a medical "
    "diagnostic tool and must not be used to make healthcare decisions."
)

if not DATA_PATH.exists():
    st.error("Dataset file not found. Add heart_disease_prediction_2026.csv to the project repository.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train_model.py` locally and upload heart_disease_model.joblib to GitHub.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()
feature_df = df.drop(columns=["heart_disease"])

st.subheader("Enter patient information")
st.write("Complete the fields below and click **Predict**.")

numeric_features = feature_df.select_dtypes(include=["number"]).columns.tolist()
categorical_features = feature_df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

with st.form("prediction_form"):
    values = {}
    left, right = st.columns(2)

    for idx, col in enumerate(feature_df.columns):
        container = left if idx % 2 == 0 else right
        with container:
            series = feature_df[col].dropna()

            if col in categorical_features:
                options = sorted(series.astype(str).unique().tolist())
                values[col] = st.selectbox(col.replace("_", " ").title(), options)
            else:
                minimum = float(series.min())
                maximum = float(series.max())
                default = float(series.median())
                is_integer = pd.api.types.is_integer_dtype(feature_df[col])

                if is_integer:
                    values[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(default),
                        step=1
                    )
                else:
                    values[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=0.1,
                        format="%.2f"
                    )

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([values], columns=feature_df.columns)
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    st.subheader("Prediction result")
    st.metric("Estimated model probability for class 1", f"{probability:.1%}")

    if prediction == 1:
        st.error("Model prediction: Heart disease class (1)")
    else:
        st.success("Model prediction: No heart disease class (0)")

    st.caption(
        "This is only the model's output for the supplied data. "
        "It does not establish whether a person has or does not have heart disease."
    )
