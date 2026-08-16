import pandas as pd
import numpy as np
import csv


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(filepath, use_header=True):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]

    last_error = None

    for encoding in encodings:

        try:

            return pd.read_csv(
                filepath,
                header=0 if use_header else None,
                encoding=encoding
            )

        except UnicodeDecodeError as e:

            last_error = e

    raise ValueError(
        "Could not read this CSV file. "
        "Try saving it as UTF-8 CSV.\n"
        f"Last encoding error: {last_error}"
    )


# ============================================================
# HEADER DETECTION
# ============================================================

def detect_header(filepath):

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8-sig",
            errors="replace"
        ) as f:

            sample = f.read(8192)

        try:

            return csv.Sniffer().has_header(
                sample
            )

        except csv.Error:

            return True

    except Exception:

        return True


# ============================================================
# TARGET DETECTION
# ============================================================

def detect_target(df):

    if df is None or df.empty:

        return None

    candidates = []

    for column in df.columns:

        series = df[column]

        clean = series.dropna()

        if len(clean) == 0:
            continue

        name = str(column).lower().strip()

        # Ignore obvious IDs
        if (
            name == "id"
            or name.endswith("_id")
            or name.startswith("id_")
            or "identifier" in name
        ):
            continue

        unique_count = clean.nunique()

        if unique_count < 2:
            continue

        unique_ratio = (
            unique_count / len(clean)
        )

        # Reject obvious row identifiers
        if unique_ratio >= 0.98:
            continue

        candidates.append(
            column
        )

    # Prefer the rightmost valid target
    if candidates:

        return candidates[-1]

    return None


# ============================================================
# TASK DETECTION
# ============================================================

def detect_task(target):

    if target is None:

        return None

    clean = target.dropna()

    if len(clean) == 0:

        return None

    numeric = pd.to_numeric(
        clean,
        errors="coerce"
    )

    numeric_ratio = (
        numeric.notna().mean()
    )

    # Text target
    if numeric_ratio < 0.95:

        unique_count = clean.nunique()

        if unique_count <= 50:

            return "Classification"

        return "Classification"

    # Numeric target
    unique_count = numeric.nunique()

    if unique_count < 2:

        return None

    # Small number of numerical classes
    if unique_count <= 10:

        return "Classification"

    # Small discrete target
    if unique_count / len(clean) < 0.05:

        return "Classification"

    return "Regression"


# ============================================================
# DATASET SUMMARY
# ============================================================

def safe_number(value):

    try:

        if pd.isna(value):

            return None

        value = float(value)

        if np.isfinite(value):

            return value

    except Exception:

        pass

    return None


def dataset_summary(df):

    statistics = {}

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_numeric_dtype(
            series
        ):

            statistics[str(column)] = {

                "mean":
                    safe_number(series.mean()),

                "std":
                    safe_number(series.std()),

                "min":
                    safe_number(series.min()),

                "max":
                    safe_number(series.max()),

                "unique":
                    int(series.nunique())
            }

        else:

            statistics[str(column)] = {

                "mean": None,

                "std": None,

                "min": None,

                "max": None,

                "unique":
                    int(series.nunique())
            }

    return {

        "rows":
            int(len(df)),

        "columns":
            int(len(df.columns)),

        "missing":
            int(df.isna().sum().sum()),

        "statistics":
            statistics
    }


# ============================================================
# JSON SERIALIZATION
# ============================================================

def make_json_serializable(obj):

    if isinstance(obj, dict):

        return {
            str(k):
            make_json_serializable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):

        return [
            make_json_serializable(v)
            for v in obj
        ]

    if isinstance(obj, tuple):

        return [
            make_json_serializable(v)
            for v in obj
        ]

    if isinstance(obj, np.integer):

        return int(obj)

    if isinstance(obj, np.floating):

        if np.isfinite(obj):

            return float(obj)

        return None

    if isinstance(obj, np.ndarray):

        return [
            make_json_serializable(v)
            for v in obj.tolist()
        ]

    if isinstance(obj, pd.Series):

        return [
            make_json_serializable(v)
            for v in obj.tolist()
        ]

    if isinstance(obj, pd.DataFrame):

        return make_json_serializable(
            obj.to_dict(
                orient="records"
            )
        )

    try:

        if pd.isna(obj):

            return None

    except Exception:

        pass

    return obj