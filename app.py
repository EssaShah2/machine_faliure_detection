import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Predictive Maintenance App", page_icon="⚙️", layout="centered"
)

st.title("⚙️ Equipment Failure Prediction")
st.write(
    "Enter the operational parameters below to predict machine failure risk."
)


# Load Model and Transformer
@st.cache_resource
def load_assets():
    # Update paths if your file names are different
    transformer = joblib.load("transformer.joblib")
    model = joblib.load("xgboost_model.joblib")
    return transformer, model


try:
    transformer, model = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Input Form
with st.form("prediction_form"):
    st.subheader("Input Operational Parameters")

    # Categorical Feature
    type_input = st.selectbox(
        "Type",
        options=["L", "M", "H"],
        help="L: Low, M: Medium, H: High quality variant",
    )

    # Numerical Features
    air_temp = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=400.0,
        value=300.0,
        step=0.1,
    )
    process_temp = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.1,
    )
    rot_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=100,
        max_value=3000,
        value=1500,
        step=10,
    )
    torque = st.number_input(
        "Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0, step=0.1
    )
    tool_wear = st.number_input(
        "Tool Wear [min]", min_value=0, max_value=300, value=0, step=1
    )

    submit = st.form_submit_button("Predict Failure Risk")

# Prediction Logic
if submit:
    # 1. Create Dataframe with exact column names expected by transformer
    input_df = pd.DataFrame(
        [
            {
                "Type": type_input,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rot_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }
        ]
    )

    try:
        # 2. Transform raw features
        transformed_data = transformer.transform(input_df)

        # 3. Get prediction and probabilities from XGBoost model
        prediction = model.predict(transformed_data)[0]
        prob = model.predict_proba(transformed_data)[0]

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(
                f"🚨 **High Risk of Failure Detected!**\n\nProbability of failure: **{prob[1]*100:.1f}%**"
            )
        else:
            st.success(
                f"✅ **Normal Operation Expected.**\n\nProbability of normal operation: **{prob[0]*100:.1f}%**"
            )

    except Exception as err:
        st.error(f"Error during prediction: {err}")