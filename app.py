from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import pandas as pd
import numpy as np
import os
import traceback
import re

from utils.data_handler import (
    detect_header,
    detect_task,
    detect_target,
    dataset_summary,
    make_json_serializable,
    load_dataset
)

from utils.ml_models import train_model

from utils.copilot import generate_copilot

from utils.visualization import (
    plot_confusion_matrix,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_distribution
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# CURRENT DATASET
# ============================================================

CURRENT_DATAFRAME = None


# ============================================================
# JSON SERIALIZATION
# ============================================================

def make_json_serializable(obj):

    # Dictionary
    if isinstance(obj, dict):

        return {
            str(k): make_json_serializable(v)
            for k, v in obj.items()
        }

    # List
    if isinstance(obj, list):

        return [
            make_json_serializable(v)
            for v in obj
        ]

    # Tuple
    if isinstance(obj, tuple):

        return [
            make_json_serializable(v)
            for v in obj
        ]

    # NumPy integer
    if isinstance(obj, np.integer):

        return int(obj)

    # NumPy float
    if isinstance(obj, np.floating):

        if np.isfinite(obj):

            return float(obj)

        return None

    # NumPy array
    if isinstance(obj, np.ndarray):

        return [
            make_json_serializable(v)
            for v in obj.tolist()
        ]

    # Pandas Series
    if isinstance(obj, pd.Series):

        return [
            make_json_serializable(v)
            for v in obj.tolist()
        ]

    # Pandas DataFrame
    if isinstance(obj, pd.DataFrame):

        return make_json_serializable(
            obj.to_dict(orient="records")
        )

    # Pandas NA / NaN
    try:

        if pd.isna(obj):

            return None

    except Exception:

        pass

    return obj


# ============================================================
# SAFE FILENAME
# ============================================================

def safe_filename(filename):

    """
    Prevent problematic filenames from being used directly.
    """

    filename = os.path.basename(filename)

    filename = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        filename
    )

    return filename


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# UPLOAD DATASET
# ============================================================

@app.route("/upload", methods=["POST"])
def upload_file():

    global CURRENT_DATAFRAME

    try:

        # ----------------------------------------------------
        # Check file
        # ----------------------------------------------------

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No file was uploaded."
            }), 400


        file = request.files["file"]


        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No file was selected."
            }), 400


        # ----------------------------------------------------
        # Check extension
        # ----------------------------------------------------

        if not file.filename.lower().endswith(".csv"):

            return jsonify({
                "success": False,
                "error": "Only CSV files are supported."
            }), 400


        # ----------------------------------------------------
        # Save file
        # ----------------------------------------------------

        filename = safe_filename(file.filename)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)


        # ----------------------------------------------------
        # Header option
        # ----------------------------------------------------

        header_value = request.form.get(
            "has_header",
            "true"
        )

        has_header = (
            str(header_value).lower()
            in ["true", "1", "yes", "on"]
        )


        # ----------------------------------------------------
        # Load dataset
        #
        # IMPORTANT:
        # Use load_dataset() instead of pd.read_csv()
        # so UTF-8, UTF-8-SIG, CP1252 and Latin-1
        # are supported.
        # ----------------------------------------------------

        df = load_dataset(
            filepath,
            use_header=has_header
        )


        # ----------------------------------------------------
        # Validate dataset
        # ----------------------------------------------------

        if df is None:

            return jsonify({
                "success": False,
                "error": "Could not load the dataset."
            }), 400


        if df.empty:

            return jsonify({
                "success": False,
                "error": "The uploaded CSV file is empty."
            }), 400


        if len(df.columns) < 2:

            return jsonify({
                "success": False,
                "error":
                "The dataset must contain at least two columns."
            }), 400


        # ----------------------------------------------------
        # Clean column names
        # ----------------------------------------------------

        df.columns = [

            str(column).strip()

            for column in df.columns

        ]


        # ----------------------------------------------------
        # Handle duplicate column names
        # ----------------------------------------------------

        if df.columns.duplicated().any():

            new_columns = []

            counts = {}

            for column in df.columns:

                if column not in counts:

                    counts[column] = 0

                    new_columns.append(column)

                else:

                    counts[column] += 1

                    new_columns.append(
                        f"{column}_{counts[column]}"
                    )

            df.columns = new_columns


        # ----------------------------------------------------
        # TARGET
        #
        # Requirement:
        # Default target = LAST column
        # ----------------------------------------------------

        target = df.columns[-1]


        # ----------------------------------------------------
        # TASK DETECTION
        #
        # detect_task() expects the target Series.
        #
        # NOT:
        # detect_task(df, target)
        #
        # Correct:
        # detect_task(df[target])
        # ----------------------------------------------------

        task = detect_task(
            df[target]
        )


        # ----------------------------------------------------
        # Save current dataframe
        # ----------------------------------------------------

        CURRENT_DATAFRAME = df


        # ----------------------------------------------------
        # Dataset summary
        # ----------------------------------------------------

        summary = dataset_summary(df)


        # ----------------------------------------------------
        # Preview
        # ----------------------------------------------------

        preview = df.head(5).to_dict(
            orient="records"
        )


        # ----------------------------------------------------
        # Header detection
        #
        # This reports what the detector thinks.
        # The actual parsing follows the user's toggle.
        # ----------------------------------------------------

        detected_header = detect_header(
            filepath
        )


        # ----------------------------------------------------
        # Distribution plots
        # ----------------------------------------------------

        try:

            distribution_plots = plot_distribution(df)

        except Exception as plot_error:

            print(
                "Distribution plot warning:",
                plot_error
            )

            distribution_plots = []


        # ----------------------------------------------------
        # MODEL OPTIONS
        # ----------------------------------------------------

        if task == "Classification":

            models = [

                "Logistic Regression",

                "Decision Tree",

                "Random Forest",

                "KNN",

                "Naive Bayes",

                "SVM"

            ]

        else:

            models = [

                "Linear Regression",

                "Decision Tree Regressor",

                "Random Forest Regressor",

                "SVR",

                "Ridge",

                "Lasso"

            ]


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {

            "success": True,

            "message": "Upload successful.",

            "filename": filename,

            "columns": [
                str(column)
                for column in df.columns
            ],

            "rows": int(len(df)),

            "preview": preview,

            "header_detected": bool(
                detected_header
            ),

            "header_used": bool(
                has_header
            ),

            "target": str(target),

            "task": task,

            "models": models,

            "summary": summary,

            "distribution_plots": [

                "/" + str(path).replace("\\", "/")

                for path in distribution_plots

            ]

        }


        print()
        print("=" * 60)
        print("DATASET UPLOADED")
        print("=" * 60)
        print("File   :", filename)
        print("Header :", has_header)
        print("Rows   :", len(df))
        print("Columns:", len(df.columns))
        print("Target :", target)
        print("Task   :", task)
        print("=" * 60)
        print()


        return jsonify(
            make_json_serializable(response)
        )


    except UnicodeDecodeError:

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error":
            "Could not decode this CSV file. "
            "The file may use an unsupported encoding."

        }), 400


    except pd.errors.ParserError as e:

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error":
            f"CSV formatting error: {str(e)}"

        }), 400


    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error":
            f"Upload failed: {str(e)}"

        }), 400


# ============================================================
# TRAIN MODEL
# ============================================================

@app.route("/train", methods=["POST"])
def train():

    global CURRENT_DATAFRAME

    try:

        # ----------------------------------------------------
        # Dataset check
        # ----------------------------------------------------

        if CURRENT_DATAFRAME is None:

            return jsonify({

                "success": False,

                "error":
                "Please upload a dataset first."

            }), 400


        # ----------------------------------------------------
        # Request data
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({

                "success": False,

                "error":
                "No training configuration received."

            }), 400


        # ----------------------------------------------------
        # Read configuration
        # ----------------------------------------------------

        target = data.get("target")

        task = data.get("task")

        model_name = data.get("model")


        # ----------------------------------------------------
        # Validate target
        # ----------------------------------------------------

        if not target:

            return jsonify({

                "success": False,

                "error":
                "Please select a target variable."

            }), 400


        if target not in CURRENT_DATAFRAME.columns:

            return jsonify({

                "success": False,

                "error":
                f"'{target}' is not a valid target column."

            }), 400


        # ----------------------------------------------------
        # Validate task
        # ----------------------------------------------------

        if task not in [
            "Classification",
            "Regression"
        ]:

            return jsonify({

                "success": False,

                "error":
                "Invalid task type."

            }), 400


        # ----------------------------------------------------
        # Validate model
        # ----------------------------------------------------

        if not model_name:

            return jsonify({

                "success": False,

                "error":
                "Please select a model."

            }), 400


        # ----------------------------------------------------
        # Allowed models
        # ----------------------------------------------------

        classification_models = {

            "Logistic Regression",

            "Decision Tree",

            "Random Forest",

            "KNN",

            "Naive Bayes",

            "SVM"

        }


        regression_models = {

            "Linear Regression",

            "Decision Tree Regressor",

            "Random Forest Regressor",

            "SVR",

            "Ridge",

            "Lasso"

        }


        # ----------------------------------------------------
        # Validate model against task
        # ----------------------------------------------------

        if task == "Classification":

            if model_name not in classification_models:

                return jsonify({

                    "success": False,

                    "error":
                    f"'{model_name}' is not a Classification model."

                }), 400


        else:

            if model_name not in regression_models:

                return jsonify({

                    "success": False,

                    "error":
                    f"'{model_name}' is not a Regression model."

                }), 400


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT automatically reject a manually overridden
        # task here.
        #
        # The frontend is allowed to override the detected task.
        #
        # However, the selected model must belong to that task.
        # ----------------------------------------------------


        # ----------------------------------------------------
        # Debug information
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)
        print("Target :", target)
        print("Task   :", task)
        print("Model  :", model_name)
        print("Rows   :", len(CURRENT_DATAFRAME))
        print(
            "Columns:",
            len(CURRENT_DATAFRAME.columns)
        )
        print("=" * 60)


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        result = train_model(

            CURRENT_DATAFRAME,

            target,

            task,

            model_name

        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = make_json_serializable(

            result.get(
                "metrics",
                {}
            )

        )


        # ----------------------------------------------------
        # Plots
        # ----------------------------------------------------

        plots = {}


        # ----------------------------------------------------
        # Classification confusion matrix
        # ----------------------------------------------------

        if task == "Classification":

            if (

                "Confusion Matrix" in metrics

                and

                "Labels" in metrics

            ):

                try:

                    cm_path = plot_confusion_matrix(

                        metrics["Confusion Matrix"],

                        metrics["Labels"]

                    )

                    if cm_path:

                        plots[
                            "confusion_matrix"
                        ] = (

                            "/" +
                            str(cm_path).replace(
                                "\\",
                                "/"
                            )

                        )

                except Exception as e:

                    print(
                        "Confusion matrix warning:",
                        e
                    )


        # ----------------------------------------------------
        # Regression actual vs predicted
        # ----------------------------------------------------

        else:

            try:

                y_test = result.get(
                    "y_test"
                )

                predictions = result.get(
                    "predictions"
                )


                if (

                    y_test is not None

                    and

                    predictions is not None

                ):

                    regression_plot = (

                        plot_actual_vs_predicted(

                            y_test,

                            predictions

                        )

                    )


                    if regression_plot:

                        plots[
                            "actual_vs_predicted"
                        ] = (

                            "/" +
                            str(
                                regression_plot
                            ).replace(
                                "\\",
                                "/"
                            )

                        )

            except Exception as e:

                print(
                    "Regression plot warning:",
                    e
                )


        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        feature_importance = result.get(
            "feature_importance"
        )


        if feature_importance is not None:

            try:

                feature_plot = (
                    plot_feature_importance(
                        feature_importance
                    )
                )


                if feature_plot:

                    plots[
                        "feature_importance"
                    ] = (

                        "/" +
                        str(
                            feature_plot
                        ).replace(
                            "\\",
                            "/"
                        )

                    )

            except Exception as e:

                print(
                    "Feature importance warning:",
                    e
                )


        # ----------------------------------------------------
        # AI COPILOT
        # ----------------------------------------------------

        try:

            copilot_message = generate_copilot(

                task,

                metrics,

                model_name

            )

        except Exception as e:

            print(
                "Copilot warning:",
                e
            )

            copilot_message = (
                f"{task} model completed.\n\n"
                "Suggestions:\n\n"
                "- Compare multiple models\n"
                "- Tune model parameters\n"
                "- Check feature quality\n"
                "- Review the evaluation metrics"
            )


        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        response = {

            "success": True,

            "message":
            f"{model_name} trained successfully.",

            "task": task,

            "model": model_name,

            "target": target,

            "metrics": metrics,

            "copilot": copilot_message,

            "plots": plots

        }


        print()
        print("=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print("Task   :", task)
        print("Model  :", model_name)
        print("Target :", target)
        print("Metrics:", metrics)
        print("=" * 60)
        print()


        return jsonify(
            make_json_serializable(
                response
            )
        )


    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error":
            f"Training failed: {str(e)}"

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "running",

        "project": "AutoML Lite",

        "version": "1.0.0"

    })


# ============================================================
# 404
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success": False,

        "error":
        "Endpoint not found."

    }), 404


# ============================================================
# 500
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({

        "success": False,

        "error":
        "Internal server error."

    }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print("AutoML Lite Server Started")

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 60)


    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )