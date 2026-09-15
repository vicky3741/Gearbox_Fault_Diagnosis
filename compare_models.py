"""
Compare Machine Learning and Deep Learning Models

Models:
1. Random Forest
2. 1D CNN
3. LSTM

Pipeline:
Saved Results
      ↓
Read Metrics
      ↓
Create Comparison Table
      ↓
Generate Graphs
      ↓
Find Best Model

Author: Harshvardhan Nayakal
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Create Results Folder
# ----------------------------------------------------

os.makedirs("results", exist_ok=True)

# ----------------------------------------------------
# Function to Read Result File
# ----------------------------------------------------

def load_results(filename):

    metrics = {}

    with open(filename, "r") as file:

        for line in file:

            if ":" in line:

                key, value = line.strip().split(":")

                metrics[key.strip()] = float(value.strip())

    return metrics


# ----------------------------------------------------
# Load Results
# ----------------------------------------------------

print("\nLoading Model Results...\n")

results = {

    "Random Forest": load_results(
        "results/random_forest_results.txt"
    ),

    "1D CNN": load_results(
        "results/cnn_results.txt"
    ),

    "LSTM": load_results(
        "results/lstm_results.txt"
    )

}

# ----------------------------------------------------
# Create DataFrame
# ----------------------------------------------------

df = pd.DataFrame(results).T

print("=" * 70)

print(df)

print("=" * 70)

# ----------------------------------------------------
# Save CSV
# ----------------------------------------------------

csv_file = "results/model_comparison.csv"

df.to_csv(csv_file)

print("\nComparison CSV Saved")

# ----------------------------------------------------
# Accuracy Comparison
# ----------------------------------------------------

plt.figure(figsize=(8,6))

plt.bar(

    df.index,

    df["Accuracy"]

)

plt.title("Accuracy Comparison")

plt.ylabel("Accuracy")

plt.ylim(0.80,1.00)

plt.grid(axis="y")

plt.savefig(

    "results/accuracy_comparison.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ----------------------------------------------------
# Precision Comparison
# ----------------------------------------------------

plt.figure(figsize=(8,6))

plt.bar(

    df.index,

    df["Precision"]

)

plt.title("Precision Comparison")

plt.ylabel("Precision")

plt.ylim(0.80,1.00)

plt.grid(axis="y")

plt.savefig(

    "results/precision_comparison.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ----------------------------------------------------
# Recall Comparison
# ----------------------------------------------------

plt.figure(figsize=(8,6))

plt.bar(

    df.index,

    df["Recall"]

)

plt.title("Recall Comparison")

plt.ylabel("Recall")

plt.ylim(0.80,1.00)

plt.grid(axis="y")

plt.savefig(

    "results/recall_comparison.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ----------------------------------------------------
# F1 Comparison
# ----------------------------------------------------

plt.figure(figsize=(8,6))

plt.bar(

    df.index,

    df["F1"]

)

plt.title("F1 Score Comparison")

plt.ylabel("F1 Score")

plt.ylim(0.80,1.00)

plt.grid(axis="y")

plt.savefig(

    "results/f1_comparison.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ----------------------------------------------------
# Kappa Comparison
# ----------------------------------------------------

plt.figure(figsize=(8,6))

plt.bar(

    df.index,

    df["Kappa"]

)

plt.title("Kappa Comparison")

plt.ylabel("Kappa")

plt.ylim(0.80,1.00)

plt.grid(axis="y")

plt.savefig(

    "results/kappa_comparison.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ----------------------------------------------------
# MCC Comparison
# ----------------------------------------------------

plt.figure(figsize=(8,6))

plt.bar(

    df.index,

    df["MCC"]

)

plt.title("MCC Comparison")

plt.ylabel("MCC")

plt.ylim(0.80,1.00)

plt.grid(axis="y")

plt.savefig(

    "results/mcc_comparison.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ----------------------------------------------------
# Overall Comparison
# ----------------------------------------------------

metrics = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1",

    "Kappa",

    "MCC"

]

plt.figure(figsize=(12,6))

for metric in metrics:

    plt.plot(

        df.index,

        df[metric],

        marker="o",

        linewidth=2,

        label=metric

    )

plt.title("Performance Comparison of Models")

plt.ylabel("Score")

plt.ylim(0.80,1.00)

plt.grid(True)

plt.legend()

plt.savefig(

    "results/all_metrics.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ----------------------------------------------------
# Best Model
# ----------------------------------------------------

best_model = df["Accuracy"].idxmax()

best_accuracy = df["Accuracy"].max()

print()

print("=" * 70)

print("Best Performing Model")

print("=" * 70)

print("Model    :", best_model)

print("Accuracy :", round(best_accuracy,4))

print("=" * 70)

# ----------------------------------------------------
# Save Summary
# ----------------------------------------------------

summary_file = "results/best_model.txt"

with open(summary_file, "w") as file:

    file.write("Best Model\n")

    file.write("=====================\n\n")

    file.write(f"Model : {best_model}\n")

    file.write(f"Accuracy : {best_accuracy:.4f}\n")

print("\nSummary Saved :", summary_file)