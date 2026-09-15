"""
Gearbox Fault Prediction

Pipeline

Signal File
      │
      ▼
Load Signal
      │
      ▼
Preprocessing
      │
      ▼
EMD
      │
      ▼
Segmentation
      │
      ▼
FFT
      │
      ▼
Feature Extraction
      │
      ▼
Load Trained Model
      │
      ▼
Prediction

Author : Harshvardhan Nayakal
"""

import os
import numpy as np
import pandas as pd

from preprocessing.preprocessing import preprocess_dataset
#from preprocessing.emd import emd_dataset
from preprocessing.segmentation import segment_dataset
from preprocessing.fft import fft_pipeline
from preprocessing.feature_extraction import extract_dataset_features

from models.model_utils import (
    load_rf_model,
    load_dl_model
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

def load_signal(file_path):

    if not os.path.exists(file_path):

        raise FileNotFoundError(

            f"\nFile not found:\n{file_path}"

        )

    extension = os.path.splitext(file_path)[1].lower()

    # -----------------------------------

    if extension == ".csv":

        data = pd.read_csv(file_path)

    elif extension in [".xlsx", ".xls"]:

        data = pd.read_excel(file_path)

    else:

        raise ValueError(

            "Only CSV and Excel files are supported."

        )

    # -----------------------------------
    # Keep numeric columns only
    # -----------------------------------

    numeric = data.select_dtypes(include=[np.number])

    if numeric.empty:

        raise ValueError(

            "No numeric signal found."

        )

    signal = numeric.iloc[:, 0].dropna().values

    print()

    print("=" * 60)

    print("Signal Loaded")

    print("=" * 60)

    print("File :", file_path)

    print("Samples :", len(signal))

    print("=" * 60)

    return signal

# -------------------------------------------------------
# Signal Processing
# -------------------------------------------------------

def process_signal(signal):

    signals = np.array([signal])

    print("\nPreprocessing...")

    signals = preprocess_dataset(signals)

    print("Applying EMD...")

    signals = emd_dataset(

        signals,

        number_of_imfs=3

    )

    print("Segmenting...")

    X, _ = segment_dataset(

        signals,

        labels=[0],

        window_size=1024,

        overlap=0

    )

    print("Applying FFT...")

    X = fft_pipeline(X)

    return X

# -------------------------------------------------------
# Random Forest Prediction
# -------------------------------------------------------

def predict_random_forest(file_path):

    signal = load_signal(file_path)

    X = process_signal(signal)

    print("Extracting Features...")

    features = extract_dataset_features(

        X

    )

    print("Loading Random Forest Model...")

    model = load_rf_model(

        "saved_models/random_forest.pkl"

    )

    prediction = model.predict(

        features

    )

    probability = model.predict_proba(

        features

    )

    confidence = np.max(

        probability

    ) * 100

    return prediction[0], confidence

# -------------------------------------------------------
# CNN Prediction
# -------------------------------------------------------

def predict_cnn(file_path):

    print("\nLoading Signal...")

    signal = load_signal(file_path)

    X = process_signal(signal)

    print("Preparing CNN Input...")

    X = X.reshape(

        X.shape[0],

        X.shape[1],

        1

    )

    print("Loading CNN Model...")

    model = load_dl_model(

        "saved_models/cnn_model.keras"

    )

    print("Predicting...")

    probability = model.predict(

        X,

        verbose=0

    )

    prediction = np.argmax(

        probability,

        axis=1

    )

    confidence = np.max(

        probability

    ) * 100

    return prediction[0], confidence


# -------------------------------------------------------
# LSTM Prediction
# -------------------------------------------------------

def predict_lstm(file_path):

    print("\nLoading Signal...")

    signal = load_signal(file_path)

    X = process_signal(signal)

    print("Preparing LSTM Input...")

    X = X.reshape(

        X.shape[0],

        X.shape[1],

        1

    )

    print("Loading LSTM Model...")

    model = load_dl_model(

        "saved_models/lstm_model.keras"

    )

    print("Predicting...")

    probability = model.predict(

        X,

        verbose=0

    )

    prediction = np.argmax(

        probability,

        axis=1

    )

    confidence = np.max(

        probability

    ) * 100

    return prediction[0], confidence

# -------------------------------------------------------
# Main Program
# -------------------------------------------------------

if __name__ == "__main__":

    print()

    print("=" * 70)
    print("      Gearbox Fault Diagnosis Using EMD + FFT")
    print("=" * 70)

    file_path = input("\nEnter Signal File (.csv/.xlsx): ").strip()

    print()

    print("Select Prediction Model")
    print("-----------------------")
    print("1. Random Forest")
    print("2. 1D CNN")
    print("3. LSTM")

    choice = input("\nEnter Choice (1/2/3): ").strip()

    try:

        # --------------------------------------------
        # Random Forest
        # --------------------------------------------

        if choice == "1":

            prediction, confidence = predict_random_forest(
                file_path
            )

            model_name = "Random Forest"

        # --------------------------------------------
        # CNN
        # --------------------------------------------

        elif choice == "2":

            prediction, confidence = predict_cnn(
                file_path
            )

            model_name = "1D CNN"

        # --------------------------------------------
        # LSTM
        # --------------------------------------------

        elif choice == "3":

            prediction, confidence = predict_lstm(
                file_path
            )

            model_name = "LSTM"

        else:

            raise ValueError(
                "Invalid model selection."
            )

        print()

        print("=" * 70)

        print("Prediction Completed")

        print("=" * 70)

        print(f"Model Used      : {model_name}")

        print(f"Gear Condition  : {CLASS_NAMES[prediction]}")

        print(f"Confidence      : {confidence:.2f}%")

        print("=" * 70)

    except FileNotFoundError as error:

        print()

        print("ERROR")

        print(error)

    except ValueError as error:

        print()

        print("ERROR")

        print(error)

    except Exception as error:

        print()

        print("=" * 70)

        print("Unexpected Error")

        print("=" * 70)

        print(error)

        print("=" * 70)