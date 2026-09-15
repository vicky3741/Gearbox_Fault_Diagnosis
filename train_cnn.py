"""
Train 1D CNN Model using EMD + FFT

Pipeline:
1. Load Dataset
2. Preprocess Signals
3. Apply EMD
4. Segment Signals
5. Apply FFT
6. Train CNN
7. Evaluate Model
8. Save Model

Author: Harshvardhan Nayakal
"""

import os
import numpy as np

from sklearn.model_selection import train_test_split

from preprocessing.dataset_loader import load_all
from preprocessing.preprocessing import preprocess_dataset
from preprocessing.emd import emd_dataset
from preprocessing.segmentation import segment_dataset
from preprocessing.fft import fft_pipeline

from models.cnn1d import (
    build_cnn,
    train_model
)

from evaluation.metrics import evaluate_model
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.roc_curve import plot_roc_curve
from evaluation.plots import (
    plot_accuracy,
    plot_loss
)

# ----------------------------------------------------
# Create folders
# ----------------------------------------------------

os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

print("\nLoading Dataset...")

signals, labels, files = load_all()

print("Signals :", len(signals))

# ----------------------------------------------------
# Preprocessing
# ----------------------------------------------------

print("\nPreprocessing Signals...")

signals = preprocess_dataset(signals)

# ----------------------------------------------------
# Apply EMD
# ----------------------------------------------------

print("\nApplying EMD...")

signals = emd_dataset(
    signals,
    number_of_imfs=3
)

print("EMD Completed")

# ----------------------------------------------------
# Segmentation
# ----------------------------------------------------

print("\nSegmenting Signals...")

X, y = segment_dataset(

    signals,

    labels,

    window_size=1024,

    overlap=0.5

)

print("Segment Shape :", X.shape)

# ----------------------------------------------------
# FFT
# ----------------------------------------------------

print("\nApplying FFT...")

X = fft_pipeline(X)

print("FFT Shape :", X.shape)

# ----------------------------------------------------
# Reshape for CNN
# ----------------------------------------------------

X = X.reshape(

    X.shape[0],

    X.shape[1],

    1

)

print("CNN Input Shape :", X.shape)

# ----------------------------------------------------
# Train Test Split
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

X_train, X_val, y_train, y_val = train_test_split(

    X_train,

    y_train,

    test_size=0.20,

    random_state=42,

    stratify=y_train

)

print()

print("Training Samples :", X_train.shape)

print("Validation Samples :", X_val.shape)

print("Testing Samples :", X_test.shape)

# ----------------------------------------------------
# Build CNN
# ----------------------------------------------------

print("\nBuilding CNN...")

model = build_cnn(

    input_length=X.shape[1],

    num_classes=5

)

model.summary()

# ----------------------------------------------------
# Train CNN
# ----------------------------------------------------

print("\nTraining CNN...")

history = train_model(

    model,

    X_train,

    y_train,

    X_val,

    y_val,

    epochs=30,

    batch_size=32

)

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

print("\nPredicting...")

probability = model.predict(

    X_test,

    verbose=0

)

prediction = np.argmax(

    probability,

    axis=1

)

# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------

print("\nEvaluating...")

results = evaluate_model(

    y_test,

    prediction

)

# ----------------------------------------------------
# Accuracy Curve
# ----------------------------------------------------

plot_accuracy(

    history,

    "CNN"

)

# ----------------------------------------------------
# Loss Curve
# ----------------------------------------------------

plot_loss(

    history,

    "CNN"

)

# ----------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------

plot_confusion_matrix(

    y_test,

    prediction,

    "CNN"

)

# ----------------------------------------------------
# ROC Curve
# ----------------------------------------------------

plot_roc_curve(

    y_test,

    probability,

    "CNN"

)

# ----------------------------------------------------
# Save Model
# ----------------------------------------------------

model.save(

    "saved_models/cnn_model.keras"

)

# ----------------------------------------------------
# Save Metrics
# ----------------------------------------------------

with open(

    "results/cnn_results.txt",

    "w"

) as file:

    for key, value in results.items():

        file.write(

            f"{key}: {value:.4f}\n"

        )

# ----------------------------------------------------
# Results
# ----------------------------------------------------

print()

print("="*70)

print("CNN Training Completed Successfully")

print("="*70)

for key, value in results.items():

    print(f"{key:<15}: {value:.4f}")

print("="*70)