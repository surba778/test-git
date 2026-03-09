"""
D7054E Lab 3 – Part A: Naive Bayes Classification on the UCI Wine Dataset.

Uses scikit-learn's GaussianNB to classify wines into three cultivar classes.
Outputs accuracy, a classification report, a confusion-matrix heatmap, and a
pair of feature-distribution plots so the model outcomes are clearly visible.

References
----------
[3] DataCamp Naive Bayes tutorial with scikit-learn:
    https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
[4] UCI Wine dataset:
    https://archive.ics.uci.edu/ml/datasets/wine
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------

def load_data():
    """Load the UCI Wine dataset and return feature matrix and target vector."""
    wine = load_wine()
    return wine.data, wine.target, wine.feature_names, wine.target_names


# ---------------------------------------------------------------------------
# 2. Pre-process
# ---------------------------------------------------------------------------

def preprocess(x_data, y_data, test_size=0.3, random_state=42):
    """Split data into training and test sets and apply StandardScaler."""
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size, random_state=random_state, stratify=y_data
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


# ---------------------------------------------------------------------------
# 3. Train model
# ---------------------------------------------------------------------------

def train_model(x_train, y_train):
    """Fit and return a GaussianNB classifier."""
    model = GaussianNB()
    model.fit(x_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 4. Evaluate model
# ---------------------------------------------------------------------------

def evaluate_model(model, x_test, y_test, target_names):
    """Print accuracy and a full classification report."""
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    return y_pred


# ---------------------------------------------------------------------------
# 5. Visualisations
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_test, y_pred, target_names):
    """Display a labelled confusion-matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Naive Bayes – Confusion Matrix (UCI Wine)", fontsize=13)
    plt.tight_layout()
    plt.savefig("confusion_matrix_naive_bayes.png", dpi=150)
    plt.show()
    print("Confusion matrix saved to confusion_matrix_naive_bayes.png")


def plot_feature_distributions(x_data, y_data, feature_names, target_names):
    """
    Plot the distribution of the two most informative features split by class.

    Only two features are plotted to keep the visualisation readable; the two
    chosen features are 'alcohol' (index 0) and 'flavanoids' (index 6), which
    are known to be highly discriminative for this dataset.
    """
    selected_indices = [0, 6]
    selected_names = [feature_names[i] for i in selected_indices]
    colors = ["steelblue", "tomato", "seagreen"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, feat_idx, feat_name in zip(axes, selected_indices, selected_names):
        for class_idx, (class_name, color) in enumerate(zip(target_names, colors)):
            values = x_data[y_data == class_idx, feat_idx]
            ax.hist(values, bins=15, alpha=0.65, label=class_name, color=color, edgecolor="white")
        ax.set_xlabel(feat_name.replace("_", " ").capitalize(), fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"Distribution of '{feat_name}'", fontsize=12)
        ax.legend()

    fig.suptitle("Feature Distributions by Wine Class (UCI Wine Dataset)", fontsize=13)
    plt.tight_layout()
    plt.savefig("feature_distributions_naive_bayes.png", dpi=150)
    plt.show()
    print("Feature distribution plot saved to feature_distributions_naive_bayes.png")


def plot_prior_probabilities(model, target_names):
    """Bar chart of the class prior probabilities learned by Naive Bayes."""
    priors = model.class_prior_
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(target_names, priors, color=["steelblue", "tomato", "seagreen"], edgecolor="black")
    ax.set_xlabel("Wine Class", fontsize=11)
    ax.set_ylabel("Prior Probability", fontsize=11)
    ax.set_title("Naive Bayes – Class Prior Probabilities", fontsize=13)
    ax.set_ylim(0, 1)
    for i, prob in enumerate(priors):
        ax.text(i, prob + 0.01, f"{prob:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("prior_probabilities_naive_bayes.png", dpi=150)
    plt.show()
    print("Prior probabilities plot saved to prior_probabilities_naive_bayes.png")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    """End-to-end Naive Bayes pipeline for the UCI Wine dataset."""
    print("=" * 60)
    print("D7054E Lab 3 – Part A: Naive Bayes on UCI Wine Dataset")
    print("=" * 60)

    # Load and inspect
    x_data, y_data, feature_names, target_names = load_data()
    print(f"\nDataset shape : {x_data.shape}  (samples × features)")
    print(f"Classes       : {list(target_names)}")
    print(f"Class counts  : {np.bincount(y_data).tolist()}\n")

    # Pre-process
    x_train, x_test, y_train, y_test = preprocess(x_data, y_data)
    print(f"Training samples : {len(x_train)}")
    print(f"Test samples     : {len(x_test)}\n")

    # Train
    model = train_model(x_train, y_train)
    print("Model trained successfully.\n")

    # Evaluate
    y_pred = evaluate_model(model, x_test, y_test, target_names)

    # Visualise
    plot_confusion_matrix(y_test, y_pred, target_names)
    plot_feature_distributions(x_data, y_data, feature_names, target_names)
    plot_prior_probabilities(model, target_names)

    print("\nPart A complete.")


if __name__ == "__main__":
    main()
