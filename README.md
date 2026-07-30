# 🛠️ Industrial Machine Failure Prediction App

An interactive, end-to-end Machine Learning web application designed for **predictive maintenance**. The app analyzes sensor metrics—such as rotational speed, torque, and process temperatures—to accurately predict potential equipment failures in real time before they happen.

🚀 **Live Interactive App:** [Launch Streamlit App](https://machinefaliuredetection-f8bfmyful9z59kmanczqob.streamlit.app/)

---

## 📌 Project Overview

Unplanned machine downtime can cause significant financial loss and operational disruption in industrial settings. This project leverages an optimized **XGBoost Classifier** trained on industrial sensor data to classify whether a machine is likely to experience operational failure.

By embedding this model inside an intuitive Streamlit interface, operators and engineers can quickly input real-time sensor parameters to receive instant risk assessments and diagnostic feedback.

---

## ✨ Key Features

* **Real-Time Failure Prediction:** Instant binary classification (`Failure` vs. `No Failure`) based on physical machine telemetry.
* **XGBoost Engine:** Powered by a high-precision XGBoost model tuned specifically for unbalanced predictive maintenance data.
* **Interactive UI:** Clean, user-friendly interface built with Streamlit for seamless parameter adjustment.
* **End-to-End Pipeline:** Seamless integration from raw sensor data preprocessing to real-world model deployment.

---

## 🛠️ Tech Stack & Tools

* **Programming Language:** Python 3.x
* **Machine Learning:** XGBoost, Scikit-learn
* **Data Processing & Analytics:** Pandas, NumPy
* **Web Framework & Deployment:** Streamlit
* **Model Serialization:** Joblib

---

## 📊 Telemetry Features Analyzed

The predictive model evaluates several key operational parameters:

* **Air Temperature [K]:** Ambient operating temperature.
* **Process Temperature [K]:** Internal operational process temperature.
* **Rotational Speed [rpm]:** Motor shaft speed.
* **Torque [Nm]:** Rotational force output.
* **Tool Wear [min]:** Cumulative usage time of the machine tool.
* **Type:** Machine quality variant (Low, Medium, High).

---

## 🚀 Getting Started Locally

To run this project on your local machine, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/EssaShah2/machine_faliure_detection.git](https://github.com/EssaShah2/machine_faliure_detection.git)
cd machine_faliure_detection
