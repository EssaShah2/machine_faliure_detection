from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Predictive Maintenance App",
    page_icon="⚙️",
    layout="centered",
)

# Application Header
st.title("⚙️ Predictive Maintenance Dashboard")
st.write(
    "Enter operational parameters or use sample presets to evaluate machine failure risk in real-time."
)


# Load Assets from Dictionary
@st.cache_resource
def load_assets():
    BASE_DIR = Path(__file__).resolve().parent
    file_path = BASE_DIR / "machine_failure_detection.joblib"

    # Load dictionary containing all objects
    pipeline_dict = joblib.load(file_path)

    # Extract components using your dictionary keys
    transformer = pipeline_dict["transfomer"]
    scaler = pipeline_dict["scaler"]
    model = pipeline_dict["xgboost"]

    return transformer, scaler, model


try:
    transformer, scaler, model = load_assets()
except Exception as e:
    st.error(f"⚠️ Error loading model file `machine_failure_detection.joblib`: {e}")
    st.stop()

# Auto-fill Sample Data Logic
st.sidebar.header("🧪 Test Presets")
st.sidebar.write("Quickly populate form parameters for testing:")

col_preset1, col_preset2 = st.sidebar.columns(2)
if col_preset1.button("Normal Operation"):
    st.session_state["type"] = "L"
    st.session_state["air_temp"] = 298.1
    st.session_state["process_temp"] = 308.6
    st.session_state["rot_speed"] = 1551
    st.session_state["torque"] = 42.8
    st.session_state["tool_wear"] = 0

if col_preset2.button("High Failure Risk"):
    st.session_state["type"] = "H"
    st.session_state["air_temp"] = 302.5
    st.session_state["process_temp"] = 311.2
    st.session_state["rot_speed"] = 1380
    st.session_state["torque"] = 68.4
    st.session_state["tool_wear"] = 215

# Input Form
with st.form("prediction_form"):
    st.subheader("📋 Operational Parameters")

    type_input = st.selectbox(
        "Equipment Quality Variant (Type)",
        options=["L", "M", "H"],
        index=["L", "M", "H"].index(st.session_state.get("type", "L")),
        help="L: Low, M: Medium, H: High quality variant",
    )

    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input(
            "Air Temperature [K]",
            min_value=250.0,
            max_value=400.0,
            value=st.session_state.get("air_temp", 300.0),
            step=0.1,
        )
        rot_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=100,
            max_value=3000,
            value=st.session_state.get("rot_speed", 1500),
            step=10,
        )
        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            max_value=300,
            value=st.session_state.get("tool_wear", 0),
            step=1,
        )

    with col2:
        process_temp = st.number_input(
            "Process Temperature [K]",
            min_value=250.0,
            max_value=400.0,
            value=st.session_state.get("process_temp", 310.0),
            step=0.1,
        )
        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.get("torque", 40.0),
            step=0.1,
        )

    st.markdown("---")
    submit = st.form_submit_button("🔍 Predict Failure Risk", use_container_width=True)

# Prediction Logic
if submit:
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
        # Step 1: Transform categorical/raw features
        transformed_data = transformer.transform(input_df)

        # Step 2: Scale numeric features
        scaled_data = scaler.transform(transformed_data)

        # Step 3: Run prediction with XGBoost
        prediction = model.predict(scaled_data)[0]
        prob = model.predict_proba(scaled_data)[0]

        st.subheader("Results Overview")
        m1, m2 = st.columns(2)

        failure_prob = prob[1] * 100
        normal_prob = prob[0] * 100

        m1.metric(label="Failure Probability", value=f"{failure_prob:.1f}%")
        m2.metric(label="Normal Operation Confidence", value=f"{normal_prob:.1f}%")

        if prediction == 1:
            st.error(
                "🚨 **HIGH RISK: Equipment Failure Detected!**\n\n"
                "Immediate maintenance inspection is recommended for this asset."
            )
        else:
            st.success(
                "✅ **NORMAL: Equipment Operating Within Safe Parameters.**\n\n"
                "No immediate maintenance required."
            )

    except Exception as err:
        st.error(f"Error during preprocessing or prediction: {err}")