"""
Train Random Forest Model

Pipeline

1. Load Dataset
2. Preprocess Signals
3. Apply EMD
4. Segment Signals
5. Apply FFT
6. Extract Features
7. Train/Test Split
8. Train Random Forest
9. Evaluate
10. Save Model

Author: Harshvardhan Nayakal
"""

import os

from sklearn.model_selection import train_test_split

from preprocessing.dataset_loader import load_all
from preprocessing.preprocessing import preprocess_dataset

# NEW
from preprocessing.emd import emd_dataset

from preprocessing.segmentation import segment_dataset
from preprocessing.fft import fft_pipeline

from preprocessing.feature_extraction import (

    extract_dataset_features,

    FEATURE_NAMES

)

from models.random_forest import RandomForestModel

from evaluation.metrics import evaluate_model
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.roc_curve import plot_roc_curve
from evaluation.plots import plot_feature_importance


# -----------------------------------------------------
# Create folders
# -----------------------------------------------------

os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

print("\nLoading Dataset...")

signals, labels, files = load_all()

print("Signals :", len(signals))


# -----------------------------------------------------
# Preprocessing
# -----------------------------------------------------

print("\nPreprocessing Signals...")

signals = preprocess_dataset(signals)


# -----------------------------------------------------
# Apply EMD (NEW)
# -----------------------------------------------------

print("\nApplying EMD...")

signals = emd_dataset(

    signals,

    number_of_imfs=3

)

print("EMD Completed")


# -----------------------------------------------------
# Segmentation
# -----------------------------------------------------

print("\nSegmenting Signals...")

X, y = segment_dataset(

    signals,

    labels,

    window_size=1024,

    overlap=0.5

)

print("Segments :", X.shape)


# -----------------------------------------------------
# FFT
# -----------------------------------------------------

print("\nApplying FFT...")

fft_features = fft_pipeline(X)

print("FFT Shape :", fft_features.shape)


# -----------------------------------------------------
# Feature Extraction
# -----------------------------------------------------

print("\nExtracting Statistical Features...")

features = extract_dataset_features(

    fft_features

)

print("Feature Matrix :", features.shape)


# -----------------------------------------------------
# Train Test Split
# -----------------------------------------------------

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    features,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("Training Samples :", len(X_train))

print("Testing Samples :", len(X_test))


# -----------------------------------------------------
# Build Model
# -----------------------------------------------------

print("\nTraining Random Forest...")

rf = RandomForestModel(

    n_estimators=200,

    random_state=42

)

rf.train(

    X_train,

    y_train

)


# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

print("\nPredicting...")

y_pred = rf.predict(

    X_test

)

y_prob = rf.predict_probability(

    X_test

)


# -----------------------------------------------------
# Evaluation
# -----------------------------------------------------

print("\nEvaluating Model...")

results = evaluate_model(

    y_test,

    y_pred

)


# -----------------------------------------------------
# Confusion Matrix
# -----------------------------------------------------

plot_confusion_matrix(

    y_test,

    y_pred,

    model_name="RandomForest"

)


# -----------------------------------------------------
# ROC Curve
# -----------------------------------------------------

plot_roc_curve(

    y_test,

    y_prob,

    model_name="RandomForest"

)


# -----------------------------------------------------
# Feature Importance
# -----------------------------------------------------

importance = rf.feature_importance()

plot_feature_importance(

    importance,

    FEATURE_NAMES

)


# -----------------------------------------------------
# Save Model
# -----------------------------------------------------

rf.save_model(

    "saved_models/random_forest.pkl"

)


# -----------------------------------------------------
# Save Metrics
# -----------------------------------------------------

with open(

    "results/random_forest_results.txt",

    "w"

) as file:

    for key, value in results.items():

        file.write(

            f"{key}: {value:.4f}\n"

        )


# -----------------------------------------------------
# Print Results
# -----------------------------------------------------

print()

print("="*70)

print("Random Forest Training Completed Successfully")

print("="*70)

for key, value in results.items():

    print(f"{key:<15}: {value:.4f}")

print("="*70)