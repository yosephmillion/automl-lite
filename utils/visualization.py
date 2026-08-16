import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


PLOT_FOLDER = "static/plots"

os.makedirs(PLOT_FOLDER, exist_ok=True)


def plot_confusion_matrix(cm, labels):

    cm = np.array(cm)

    path = os.path.join(PLOT_FOLDER, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(ax=ax, cmap="Blues", values_format="d")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

    return path

def plot_actual_vs_predicted(y_true, predictions):

    path = os.path.join(PLOT_FOLDER, "actual_vs_predicted.png")

    plt.figure(figsize=(7,5))

    plt.scatter(
        y_true,
        predictions,
        alpha=0.7
    )

    minimum = min(min(y_true), min(predictions))
    maximum = max(max(y_true), max(predictions))

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--"
    )

    plt.xlabel("Actual")

    plt.ylabel("Predicted")

    plt.title("Actual vs Predicted")

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

    return path


def plot_feature_importance(feature_importance):

    if feature_importance is None:

        return None

    path = os.path.join(PLOT_FOLDER, "feature_importance.png")

    names = list(feature_importance.keys())

    values = list(feature_importance.values())

    order = sorted(
        range(len(values)),
        key=lambda i: values[i],
        reverse=True
    )

    names = [names[i] for i in order]

    values = [values[i] for i in order]

    plt.figure(figsize=(8,5))

    plt.barh(names, values)

    plt.gca().invert_yaxis()

    plt.xlabel("Importance")

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

    return path


def plot_distribution(df):
    numeric = df.select_dtypes(include="number")

    paths = []

    numeric_columns = list(numeric.columns)

    for col in numeric_columns:
        # Remove missing and infinite values
        data = df[col].replace([float("inf"), float("-inf")], float("nan")).dropna()

        # Skip empty columns
        if data.empty:
            # skip columns without valid numeric data
            continue

        plt.figure(figsize=(6, 4))

        plt.hist(data, bins=20)

        plt.title(col)

        plt.tight_layout()

        filename = f"distribution_{col}.png"
        # sanitize filename
        filename = filename.replace(" ", "_")
        path = os.path.join(PLOT_FOLDER, filename)

        plt.savefig(path)
        plt.close()

        paths.append(path)

    return paths