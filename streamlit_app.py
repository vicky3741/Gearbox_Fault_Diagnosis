"""
Gearbox Fault Diagnosis Dashboard

Author : Harshvardhan Nayakal
"""

import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from preprocessing.preprocessing import preprocess_dataset
from preprocessing.emd import emd_dataset
from preprocessing.segmentation import segment_dataset
from preprocessing.fft import fft_pipeline
from preprocessing.feature_extraction import extract_dataset_features

from models.model_utils import (
    load_rf_model,
    load_dl_model
)

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(

    page_title="Gearbox Fault Diagnosis",

    page_icon="⚙",

    layout="wide"

)

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("⚙ Gearbox Fault Diagnosis using EMD + FFT + AI")

st.markdown(
"""
This application predicts the condition of a gearbox using

- Empirical Mode Decomposition (EMD)
- Fast Fourier Transform (FFT)
- Random Forest
- 1D CNN
- LSTM
"""
)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("Settings")

model_name = st.sidebar.selectbox(

    "Select Model",

    [

        "Random Forest",

        "1D CNN",

        "LSTM"

    ]

)

st.sidebar.success(f"Selected Model : {model_name}")

# -------------------------------------------------------
# File Upload
# -------------------------------------------------------

uploaded_file = st.file_uploader(

    "Upload Signal File",

    type=[

        "csv",

        "xlsx",

        "xls"

    ]

)

# -------------------------------------------------------
# Class Names
# -------------------------------------------------------

CLASS_NAMES = {

    0: "Healthy",

    1: "BT25",

    2: "BT50",

    3: "BT75",

    4: "BT100"

}

# -------------------------------------------------------
# Load Signal
# -------------------------------------------------------

def load_signal(file):

    extension = os.path.splitext(file.name)[1].lower()

    if extension == ".csv":

        data = pd.read_csv(file)

    else:

        data = pd.read_excel(file)

    numeric = data.select_dtypes(include=np.number)

    signal = numeric.iloc[:,0].dropna().values

    return signal

# -------------------------------------------------------
# Show Signal
# -------------------------------------------------------

if uploaded_file is not None:

    signal = load_signal(uploaded_file)

    st.success("Signal Loaded Successfully")

    st.write("Total Samples :", len(signal))

    st.subheader("Signal Preview")

    st.dataframe(

        pd.DataFrame(

            signal,

            columns=["Amplitude"]

        ).head(20)

    )

    # ---------------------------------------

    fig, ax = plt.subplots(

        figsize=(12,4)

    )

    ax.plot(signal)

    ax.set_title(

        "Raw Vibration Signal"

    )

    ax.set_xlabel(

        "Sample"

    )

    ax.set_ylabel(

        "Amplitude"

    )

    st.pyplot(fig)

# -------------------------------------------------------
# Information
# -------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.info(
"""
Pipeline

Signal

↓

Preprocessing

↓

EMD

↓

FFT

↓

Machine Learning

↓

Prediction
"""
)

st.sidebar.markdown("---")

st.sidebar.write("Version : 1.0")

st.sidebar.write("Developed using Streamlit")

# -------------------------------------------------------
# Signal Processing Pipeline
# -------------------------------------------------------

def process_signal(signal):

    signals = np.array([signal])

    # Preprocessing
    signals = preprocess_dataset(signals)

    # EMD
    signals = emd_dataset(
        signals,
        number_of_imfs=3
    )

    # Segmentation
    X, _ = segment_dataset(
        signals,
        labels=[0],
        window_size=1024,
        overlap=0
    )

    # FFT
    X_fft = fft_pipeline(X)

    return X, X_fft


# -------------------------------------------------------
# Random Forest Prediction
# -------------------------------------------------------

def predict_random_forest(signal):

    _, X_fft = process_signal(signal)

    features = extract_dataset_features(X_fft)

    model = load_rf_model(
        "saved_models/random_forest.pkl"
    )

    probability = model.predict_proba(features)

    prediction = np.argmax(probability, axis=1)

    confidence = np.max(probability) * 100

    return prediction[0], confidence, X_fft


# -------------------------------------------------------
# CNN Prediction
# -------------------------------------------------------

def predict_cnn(signal):

    _, X_fft = process_signal(signal)

    X = X_fft.reshape(
        X_fft.shape[0],
        X_fft.shape[1],
        1
    )

    model = load_dl_model(
        "saved_models/cnn_model.keras"
    )

    probability = model.predict(
        X,
        verbose=0
    )

    prediction = np.argmax(
        probability,
        axis=1
    )

    confidence = np.max(probability) * 100

    return prediction[0], confidence, X_fft


# -------------------------------------------------------
# LSTM Prediction
# -------------------------------------------------------

def predict_lstm(signal):

    _, X_fft = process_signal(signal)

    X = X_fft.reshape(
        X_fft.shape[0],
        X_fft.shape[1],
        1
    )

    model = load_dl_model(
        "saved_models/lstm_model.keras"
    )

    probability = model.predict(
        X,
        verbose=0
    )

    prediction = np.argmax(
        probability,
        axis=1
    )

    confidence = np.max(probability) * 100

    return prediction[0], confidence, X_fft


# -------------------------------------------------------
# FFT Visualization
# -------------------------------------------------------

def plot_fft(fft_signal):

    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(fft_signal[0])

    ax.set_title("FFT Spectrum")

    ax.set_xlabel("Frequency Bin")

    ax.set_ylabel("Magnitude")

    st.pyplot(fig)


# -------------------------------------------------------
# IMF Visualization
# -------------------------------------------------------

def plot_imfs(signal):

    try:

        from PyEMD import EMD

        emd = EMD()

        imfs = emd(signal)

        st.subheader("Intrinsic Mode Functions (IMFs)")

        number = min(3, len(imfs))

        for i in range(number):

            fig, ax = plt.subplots(figsize=(12,3))

            ax.plot(imfs[i])

            ax.set_title(f"IMF {i+1}")

            st.pyplot(fig)

    except Exception as error:

        st.warning(f"Unable to display IMF plots.\n{error}")

# -------------------------------------------------------
# Prediction Section
# -------------------------------------------------------

if uploaded_file is not None:

    st.markdown("---")

    if st.button("🚀 Predict Gear Condition"):

        with st.spinner("Processing Signal..."):

            try:

                # -----------------------------
                # Model Selection
                # -----------------------------

                if model_name == "Random Forest":

                    prediction, confidence, fft_signal = predict_random_forest(signal)

                elif model_name == "1D CNN":

                    prediction, confidence, fft_signal = predict_cnn(signal)

                else:

                    prediction, confidence, fft_signal = predict_lstm(signal)

                # -----------------------------
                # Results
                # -----------------------------

                st.success("Prediction Completed Successfully!")

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(

                        "Gear Condition",

                        CLASS_NAMES[prediction]

                    )

                with col2:

                    st.metric(

                        "Confidence",

                        f"{confidence:.2f}%"

                    )

                # -----------------------------
                # FFT Plot
                # -----------------------------

                st.markdown("---")

                st.subheader("FFT Spectrum")

                plot_fft(fft_signal)

                # -----------------------------
                # IMF Plot
                # -----------------------------

                st.markdown("---")

                plot_imfs(signal)

                # -----------------------------
                # Result Table
                # -----------------------------

                st.markdown("---")

                st.subheader("Prediction Summary")

                result_df = pd.DataFrame({

                    "Model":[model_name],

                    "Prediction":[CLASS_NAMES[prediction]],

                    "Confidence (%)":[round(confidence,2)]

                })

                st.dataframe(

                    result_df,

                    use_container_width=True

                )

                # -----------------------------
                # Download CSV
                # -----------------------------

                csv = result_df.to_csv(

                    index=False

                ).encode("utf-8")

                st.download_button(

                    label="📥 Download Result",

                    data=csv,

                    file_name="prediction_result.csv",

                    mime="text/csv"

                )

            except Exception as e:

                st.error(f"Prediction Failed\n\n{e}")

# -------------------------------------------------------
# Footer
# -------------------------------------------------------

st.markdown("---")

st.markdown(
"""
### About

**Gearbox Fault Diagnosis System**

This application detects gearbox faults using vibration signals.

### Processing Pipeline

Signal

↓

Preprocessing

↓

EMD (Intrinsic Mode Functions)

↓

Segmentation

↓

FFT

↓

Feature Extraction

↓

Machine Learning / Deep Learning

↓

Prediction

---

**Models Used**

- Random Forest
- 1D CNN
- LSTM

---

Developed using:

- Python
- Streamlit
- TensorFlow
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- PyEMD
"""
)