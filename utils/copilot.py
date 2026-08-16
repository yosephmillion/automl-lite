def generate_copilot(
    task,
    metrics,
    model_name,
    dataset_info=None
):

    if not metrics:

        return (
            "No evaluation metrics are available yet. "
            "Train a model to receive recommendations."
        )

    lines = []

    lines.append(
        "MODEL ANALYSIS"
    )

    lines.append(
        f"Task: {task}"
    )

    lines.append(
        f"Model: {model_name}"
    )

    lines.append("")

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if task == "Classification":

        accuracy = metrics.get(
            "Accuracy"
        )

        precision = metrics.get(
            "Precision"
        )

        recall = metrics.get(
            "Recall"
        )

        f1 = metrics.get(
            "F1 Score"
        )

        if accuracy is not None:

            lines.append(
                f"Accuracy: {accuracy:.2%}"
            )

        if precision is not None:

            lines.append(
                f"Precision: {precision:.2%}"
            )

        if recall is not None:

            lines.append(
                f"Recall: {recall:.2%}"
            )

        if f1 is not None:

            lines.append(
                f"F1 Score: {f1:.2%}"
            )

        lines.append("")

        lines.append(
            "INTERPRETATION"
        )

        if accuracy is not None:

            if accuracy >= 0.90:

                lines.append(
                    "The model shows very strong predictive performance."
                )

            elif accuracy >= 0.75:

                lines.append(
                    "The model shows reasonable predictive performance, "
                    "but there is room for improvement."
                )

            elif accuracy >= 0.60:

                lines.append(
                    "The model has moderate performance. "
                    "Feature engineering or a different model may help."
                )

            else:

                lines.append(
                    "The model performance is relatively weak. "
                    "Consider improving preprocessing, feature selection, "
                    "or trying another model."
                )

        if recall is not None and precision is not None:

            if recall > precision + 0.10:

                lines.append(
                    "Recall is noticeably higher than precision, "
                    "so the model may be producing more false positives."
                )

            elif precision > recall + 0.10:

                lines.append(
                    "Precision is noticeably higher than recall, "
                    "so the model may be missing some positive cases."
                )

        lines.append("")

        lines.append(
            "RECOMMENDATION"
        )

        lines.append(
            "Compare this model against Random Forest and SVM. "
            "Use the model that provides the best balance between "
            "accuracy, precision, recall and F1 score."
        )

    # ========================================================
    # REGRESSION
    # ========================================================

    else:

        mae = metrics.get(
            "MAE"
        )

        rmse = metrics.get(
            "RMSE"
        )

        r2 = metrics.get(
            "R² Score"
        )

        if mae is not None:

            lines.append(
                f"MAE: {mae:.4f}"
            )

        if rmse is not None:

            lines.append(
                f"RMSE: {rmse:.4f}"
            )

        if r2 is not None:

            lines.append(
                f"R² Score: {r2:.4f}"
            )

        lines.append("")

        lines.append(
            "INTERPRETATION"
        )

        if r2 is not None:

            if r2 >= 0.90:

                lines.append(
                    "The model explains most of the variation in the target."
                )

            elif r2 >= 0.70:

                lines.append(
                    "The model has a strong relationship with the target, "
                    "although some unexplained variation remains."
                )

            elif r2 >= 0.40:

                lines.append(
                    "The model has moderate explanatory power."
                )

            elif r2 >= 0:

                lines.append(
                    "The model has limited explanatory power."
                )

            else:

                lines.append(
                    "The model performs poorly compared with a simple "
                    "baseline prediction."
                )

        lines.append("")

        lines.append(
            "RECOMMENDATION"
        )

        lines.append(
            "Compare this model with Random Forest Regressor, "
            "Ridge and SVR. Examine RMSE and R² together rather "
            "than relying on a single metric."
        )

    return "\n".join(lines)