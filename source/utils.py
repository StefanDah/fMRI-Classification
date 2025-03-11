import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, y_pred, save_path="figures/confusion_matrix.png"):
    """
    Plots and saves the confusion matrix.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        save_path (str, optional): Path to save the confusion matrix plot.
    """

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📸 Confusion matrix saved at: {save_path}")
