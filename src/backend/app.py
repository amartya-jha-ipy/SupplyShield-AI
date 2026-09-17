from pathlib import Path

import math

import numpy as np
import pandas as pd

from flask import Flask, jsonify, request
from flask_cors import CORS


# ============================================================
# APPLICATION SETUP
# ============================================================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": "*"
        }
    }
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# DATA FILES
# ============================================================

DATA_FILES = {
    "final_risk": "final_shortage_risk.csv",
    "xai": "shortage_xai_explanations.csv",
    "predictions": "shortage_risk_predictions.csv",
    "spatial": "spatial_shortage_signals.csv",
    "cascade": "cascade_simulation.csv",
    "intervention": "intervention_results.csv",
    "feasibility": "redistribution_feasibility.csv",
}


# ============================================================
# DATA STORAGE
# ============================================================

DATA = {}

LOAD_STATUS = {}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_csv(filename):
    """
    Load a processed CSV file from data/processed/.
    """

    path = PROCESSED_DIR / filename

    if not path.exists():

        print(
            f"[WARNING] Missing file: {path}"
        )

        return pd.DataFrame()

    try:

        dataframe = pd.read_csv(path)

        print(
            f"[LOADED] {filename} "
            f"({len(dataframe):,} rows)"
        )

        return dataframe

    except Exception as error:

        print(
            f"[ERROR] Could not load {filename}: "
            f"{error}"
        )

        return pd.DataFrame()


def clean_value(value):
    """
    Convert pandas / NumPy values into JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):

        if np.isnan(value) or np.isinf(value):
            return None

        return float(value)

    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def dataframe_to_records(dataframe):
    """
    Convert a DataFrame to JSON-safe records.
    """

    if dataframe is None or dataframe.empty:
        return []

    records = dataframe.to_dict(
        orient="records"
    )

    cleaned = []

    for record in records:

        cleaned_record = {}

        for key, value in record.items():

            cleaned_record[str(key)] = clean_value(
                value
            )

        cleaned.append(cleaned_record)

    return cleaned


def find_column(dataframe, candidates):
    """
    Find the first matching column from a list of
    possible column names.
    """

    if dataframe is None or dataframe.empty:
        return None

    normalized = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:

        candidate_lower = (
            str(candidate)
            .strip()
            .lower()
        )

        if candidate_lower in normalized:

            return normalized[
                candidate_lower
            ]

    return None


def numeric_value(value, default=0.0):
    """
    Safely convert a value to float.
    """

    try:

        result = float(value)

        if math.isnan(result) or math.isinf(result):
            return default

        return result

    except (TypeError, ValueError):

        return default


def get_unique_count(dataframe, candidates):
    """
    Count unique values in the first available column.
    """

    column = find_column(
        dataframe,
        candidates
    )

    if column is None:
        return 0

    return int(
        dataframe[column]
        .dropna()
        .nunique()
    )


# ============================================================
# ALERT DATA PREPARATION
# ============================================================

def get_alert_dataframe(limit=100):
    """
    Return the same alert population used by /api/alerts.

    Alerts are sorted by final shortage score in descending
    order and limited to the requested number of records.

    Keeping this logic in one function ensures that the
    dashboard summary and visible alert cards use the same
    population.
    """

    dataframe = DATA["final_risk"]

    if dataframe.empty:

        return pd.DataFrame()

    result_dataframe = dataframe.copy()

    score_column = find_column(
        result_dataframe,
        [
            "final_shortage_score",
            "risk_score",
            "score",
        ]
    )

    if score_column is not None:

        result_dataframe = (
            result_dataframe
            .sort_values(
                score_column,
                ascending=False
            )
        )

    try:

        limit = int(limit)

    except (TypeError, ValueError):

        limit = 100

    limit = max(
        1,
        min(
            limit,
            1000
        )
    )

    return (
        result_dataframe
        .head(limit)
        .copy()
    )


def calculate_risk_distribution(dataframe):
    """
    Calculate risk distribution from a specific DataFrame.
    """

    distribution = {
        "HIGH": 0,
        "MODERATE": 0,
        "LOW": 0,
        "VERY_LOW": 0,
    }

    if dataframe is None or dataframe.empty:

        return distribution

    risk_column = find_column(
        dataframe,
        [
            "risk_level",
            "final_risk_level",
            "shortage_risk_level",
        ]
    )

    if risk_column is None:

        return distribution

    normalized_risk = (
        dataframe[risk_column]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(
            " ",
            "_",
            regex=False
        )
        .str.replace(
            "-",
            "_",
            regex=False
        )
    )

    counts = normalized_risk.value_counts()

    for category in distribution:

        distribution[category] = int(
            counts.get(
                category,
                0
            )
        )

    return distribution


# ============================================================
# VALUE NORMALIZATION
# ============================================================

def normalize_binary_value(value):
    """
    Convert common boolean / binary representations into:

        1 = positive / detected / yes
        0 = negative / not detected / no
        NaN = unknown / invalid

    Handles values such as:

        True
        False
        1
        0
        Yes
        No
        Y
        N
        Detected
        Not Detected
    """

    if pd.isna(value):

        return np.nan

    text = (
        str(value)
        .strip()
        .lower()
    )

    positive_values = {
        "true",
        "1",
        "yes",
        "y",
        "detected",
        "present",
        "positive",
        "signal",
        "shortage",
    }

    negative_values = {
        "false",
        "0",
        "no",
        "n",
        "not detected",
        "absent",
        "negative",
        "no signal",
        "none",
        "normal",
    }

    if text in positive_values:

        return 1

    if text in negative_values:

        return 0

    try:

        numeric = float(value)

        if (
            math.isnan(numeric)
            or math.isinf(numeric)
        ):

            return np.nan

        return (
            1
            if numeric > 0
            else 0
        )

    except (
        TypeError,
        ValueError
    ):

        return np.nan


# ============================================================
# HISTORICAL REGIONAL-SIGNAL VALIDATION
# ============================================================

def calculate_regional_signal_metrics(dataframe):
    """
    Calculate historical spatial-validation metrics.

    These are observation-level historical statistics:

    1. Number of observations with a regional shortage signal.
    2. Observed stockout rate when a signal is present.
    3. Observed stockout rate when no signal is present.
    4. Ratio between those two observed stockout rates.

    IMPORTANT:
    These metrics are historical validation evidence.
    They are NOT current live shortage counts.
    """

    result = {
        "regional_signal_records": 0,
        "regional_signal_stockout_rate": None,
        "no_signal_stockout_rate": None,
        "regional_signal_ratio": None,
    }

    if dataframe is None or dataframe.empty:

        return result

    # --------------------------------------------------------
    # Find regional-signal column
    #
    # IMPORTANT:
    # regional_event is included because the actual
    # spatial dataset may encode the regional signal using
    # this field.
    # --------------------------------------------------------

    signal_column = find_column(
        dataframe,
        [
            "regional_signal",
            "regional_shortage_signal",
            "regional_event",
            "spatial_signal",
        ]
    )

    # --------------------------------------------------------
    # Find stockout column
    # --------------------------------------------------------

    stockout_column = find_column(
        dataframe,
        [
            "current_stockout",
            "stockout",
        ]
    )

    if signal_column is None:

        print(
            "[WARNING] Regional signal column "
            "not found in spatial dataset."
        )

        print(
            "[INFO] Available spatial columns:",
            list(dataframe.columns)
        )

        return result

    if stockout_column is None:

        print(
            "[WARNING] Stockout column "
            "not found in spatial dataset."
        )

        print(
            "[INFO] Available spatial columns:",
            list(dataframe.columns)
        )

        return result

    # --------------------------------------------------------
    # Create working dataframe
    # --------------------------------------------------------

    working = dataframe[
        [
            signal_column,
            stockout_column,
        ]
    ].copy()

    # --------------------------------------------------------
    # Normalize regional signal
    # --------------------------------------------------------

    working["_regional_signal"] = (
        working[signal_column]
        .apply(normalize_binary_value)
    )

    # --------------------------------------------------------
    # Normalize stockout
    # --------------------------------------------------------

    working["_stockout"] = (
        working[stockout_column]
        .apply(normalize_binary_value)
    )

    # --------------------------------------------------------
    # Remove invalid observations
    # --------------------------------------------------------

    working = working.dropna(
        subset=[
            "_regional_signal",
            "_stockout",
        ]
    )

    if working.empty:

        print(
            "[WARNING] No valid observations "
            "available for regional validation."
        )

        return result

    # --------------------------------------------------------
    # Split observations
    # --------------------------------------------------------

    signal_records = working[
        working["_regional_signal"] == 1
    ]

    no_signal_records = working[
        working["_regional_signal"] == 0
    ]

    # --------------------------------------------------------
    # Number of observations with signal
    # --------------------------------------------------------

    result[
        "regional_signal_records"
    ] = int(
        len(signal_records)
    )

    # --------------------------------------------------------
    # Stockout rate when signal exists
    # --------------------------------------------------------

    if not signal_records.empty:

        result[
            "regional_signal_stockout_rate"
        ] = float(
            signal_records[
                "_stockout"
            ].mean()
        )

    # --------------------------------------------------------
    # Stockout rate when no signal exists
    # --------------------------------------------------------

    if not no_signal_records.empty:

        result[
            "no_signal_stockout_rate"
        ] = float(
            no_signal_records[
                "_stockout"
            ].mean()
        )

    # --------------------------------------------------------
    # Signal ratio
    # --------------------------------------------------------

    signal_rate = result[
        "regional_signal_stockout_rate"
    ]

    no_signal_rate = result[
        "no_signal_stockout_rate"
    ]

    if (
        signal_rate is not None
        and no_signal_rate is not None
        and no_signal_rate > 0
    ):

        result[
            "regional_signal_ratio"
        ] = float(
            signal_rate
            / no_signal_rate
        )

    return result


# ============================================================
# LOAD ALL DATA
# ============================================================

for key, filename in DATA_FILES.items():

    dataframe = load_csv(filename)

    DATA[key] = dataframe

    LOAD_STATUS[key] = not dataframe.empty


print()
print("=" * 60)
print("DATA LOADING COMPLETE")
print("=" * 60)
print()

# ------------------------------------------------------------
# Diagnostic information for spatial dataset
# ------------------------------------------------------------

if not DATA["spatial"].empty:

    print(
        "[SPATIAL] Columns:"
    )

    print(
        list(
            DATA["spatial"].columns
        )
    )

    regional_column = find_column(
        DATA["spatial"],
        [
            "regional_signal",
            "regional_shortage_signal",
            "regional_event",
            "spatial_signal",
        ]
    )

    if regional_column is not None:

        print(
            f"[SPATIAL] Regional signal column: "
            f"{regional_column}"
        )

        print(
            "[SPATIAL] Regional signal values:"
        )

        print(
            DATA["spatial"][
                regional_column
            ]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

    else:

        print(
            "[WARNING] No regional signal column "
            "was detected."
        )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    record_counts = {}

    for key, dataframe in DATA.items():

        record_counts[key] = int(
            len(dataframe)
        )

    return jsonify(
        {
            "service": "SupplyShield AI Backend",
            "status": "ok",
            "datasets": LOAD_STATUS,
            "record_counts": record_counts,
        }
    )


# ============================================================
# SUMMARY ENDPOINT
# ============================================================

@app.route(
    "/api/summary",
    methods=["GET"]
)
def summary():

    final_risk = DATA["final_risk"]
    spatial = DATA["spatial"]
    cascade = DATA["cascade"]
    feasibility = DATA["feasibility"]

    # --------------------------------------------------------
    # Facilities
    # --------------------------------------------------------

    facilities = get_unique_count(
        spatial,
        [
            "facility_id",
            "facility",
            "hf_pk",
        ]
    )

    if facilities == 0:

        facilities = get_unique_count(
            final_risk,
            [
                "facility_id",
                "facility",
                "hf_pk",
            ]
        )

    # --------------------------------------------------------
    # Visible alert population
    #
    # This is intentionally the SAME population returned
    # by /api/alerts.
    # --------------------------------------------------------

    alert_dataframe = get_alert_dataframe(
        limit=100
    )

    alerts = len(
        alert_dataframe
    )

    # --------------------------------------------------------
    # Risk distribution
    #
    # Risk Overview represents the same 100 highest-risk
    # records displayed in Facility Alerts.
    # --------------------------------------------------------

    risk_distribution = (
        calculate_risk_distribution(
            alert_dataframe
        )
    )

    high_risk = risk_distribution[
        "HIGH"
    ]

    # --------------------------------------------------------
    # Latest risk date
    #
    # Informational only.
    # --------------------------------------------------------

    date_column = find_column(
        final_risk,
        [
            "date",
            "risk_date",
            "snapshot_date",
        ]
    )

    latest_date = None

    if (
        date_column is not None
        and not final_risk.empty
    ):

        parsed_dates = pd.to_datetime(
            final_risk[date_column],
            errors="coerce"
        )

        valid_dates = (
            parsed_dates
            .dropna()
        )

        if not valid_dates.empty:

            latest_date = (
                valid_dates.max()
            )

    # --------------------------------------------------------
    # Regional signals
    #
    # Includes regional_event.
    # --------------------------------------------------------

    regional_signals = 0

    regional_column = find_column(
        spatial,
        [
            "regional_signal",
            "regional_shortage_signal",
            "regional_event",
            "spatial_signal",
        ]
    )

    if regional_column is not None:

        normalized_values = (
            spatial[regional_column]
            .apply(
                normalize_binary_value
            )
        )

        regional_signals = int(
            (
                normalized_values == 1
            ).sum()
        )

    # --------------------------------------------------------
    # Historical spatial validation
    # --------------------------------------------------------

    regional_validation = (
        calculate_regional_signal_metrics(
            spatial
        )
    )

    # --------------------------------------------------------
    # Medicine
    # --------------------------------------------------------

    medicine = (
        "Paracetamol 500 mg"
    )

    # --------------------------------------------------------
    # Build response
    # --------------------------------------------------------

    result = {

        "medicine":
            medicine,

        "facilities":
            facilities,

        "alerts":
            alerts,

        "high_risk":
            high_risk,

        "risk_distribution":
            risk_distribution,

        "latest_risk_date":
            (
                latest_date.isoformat()
                if latest_date is not None
                else None
            ),

        "regional_signals":
            regional_signals,

        "regional_validation":
            regional_validation,

        "risk_records":
            len(final_risk),

        "spatial_records":
            len(spatial),

        "cascade_records":
            len(cascade),

        "feasibility_records":
            len(feasibility),

    }

    return jsonify(
        {
            "status": "ok",
            "summary": result,
        }
    )


# ============================================================
# ALERTS ENDPOINT
# ============================================================

@app.route(
    "/api/alerts",
    methods=["GET"]
)
def alerts():

    dataframe = DATA["final_risk"]

    if dataframe.empty:

        return jsonify(
            {
                "status": "ok",
                "alerts": [],
                "count": 0,
            }
        )

    # --------------------------------------------------------
    # Optional limit
    # --------------------------------------------------------

    try:

        limit = int(
            request.args.get(
                "limit",
                100
            )
        )

    except (TypeError, ValueError):

        limit = 100

    limit = max(
        1,
        min(
            limit,
            1000
        )
    )

    # --------------------------------------------------------
    # Use shared alert population
    # --------------------------------------------------------

    result_dataframe = (
        get_alert_dataframe(
            limit=limit
        )
    )

    records = (
        dataframe_to_records(
            result_dataframe
        )
    )

    return jsonify(
        {
            "status": "ok",
            "alerts": records,
            "count": len(records),
        }
    )


# ============================================================
# XAI EXPLANATIONS ENDPOINT
# ============================================================

@app.route(
    "/api/alerts/explanations",
    methods=["GET"]
)
def alert_explanations():

    dataframe = DATA["xai"]

    if dataframe.empty:

        return jsonify(
            {
                "status": "ok",
                "explanations": [],
                "count": 0,
            }
        )

    try:

        limit = int(
            request.args.get(
                "limit",
                100
            )
        )

    except (TypeError, ValueError):

        limit = 100

    limit = max(
        1,
        min(
            limit,
            1000
        )
    )

    result_dataframe = dataframe.copy()

    score_column = find_column(
        result_dataframe,
        [
            "final_shortage_score",
            "risk_score",
            "score",
        ]
    )

    if score_column is not None:

        result_dataframe = (
            result_dataframe
            .sort_values(
                score_column,
                ascending=False
            )
        )

    result_dataframe = (
        result_dataframe
        .head(limit)
    )

    records = (
        dataframe_to_records(
            result_dataframe
        )
    )

    return jsonify(
        {
            "status": "ok",
            "explanations": records,
            "count": len(records),
        }
    )


# ============================================================
# FACILITY ENDPOINT
# ============================================================

@app.route(
    "/api/facility/<facility_id>",
    methods=["GET"]
)
def facility_details(facility_id):

    result = {}

    # --------------------------------------------------------
    # Final risk
    # --------------------------------------------------------

    final_risk = DATA["final_risk"]

    facility_column = find_column(
        final_risk,
        [
            "facility_id",
            "facility",
            "hf_pk",
        ]
    )

    if facility_column is not None:

        matches = final_risk[
            final_risk[facility_column]
            .astype(str)
            == str(facility_id)
        ]

        result["risk"] = (
            dataframe_to_records(
                matches
            )
        )

    else:

        result["risk"] = []

    # --------------------------------------------------------
    # XAI
    # --------------------------------------------------------

    xai = DATA["xai"]

    facility_column = find_column(
        xai,
        [
            "facility_id",
            "facility",
            "hf_pk",
        ]
    )

    if facility_column is not None:

        matches = xai[
            xai[facility_column]
            .astype(str)
            == str(facility_id)
        ]

        result["explanations"] = (
            dataframe_to_records(
                matches
            )
        )

    else:

        result["explanations"] = []

    # --------------------------------------------------------
    # Spatial
    # --------------------------------------------------------

    spatial = DATA["spatial"]

    facility_column = find_column(
        spatial,
        [
            "facility_id",
            "facility",
            "hf_pk",
        ]
    )

    if facility_column is not None:

        matches = spatial[
            spatial[facility_column]
            .astype(str)
            == str(facility_id)
        ]

        result["spatial"] = (
            dataframe_to_records(
                matches
            )
        )

    else:

        result["spatial"] = []

    # --------------------------------------------------------
    # Redistribution options
    # --------------------------------------------------------

    feasibility = DATA["feasibility"]

    recipient_column = find_column(
        feasibility,
        [
            "recipient_facility_id",
            "recipient_id",
        ]
    )

    if recipient_column is not None:

        matches = feasibility[
            feasibility[recipient_column]
            .astype(str)
            == str(facility_id)
        ]

        result["redistribution"] = (
            dataframe_to_records(
                matches
            )
        )

    else:

        result["redistribution"] = []

    return jsonify(
        {
            "status": "ok",
            "facility_id": facility_id,
            "data": result,
        }
    )


# ============================================================
# CASCADE ENDPOINT
# ============================================================

@app.route(
    "/api/cascade",
    methods=["GET", "POST"]
)
def cascade():

    dataframe = DATA["cascade"]

    if dataframe.empty:

        return jsonify(
            {
                "status": "ok",
                "cascade": [],
                "count": 0,
            }
        )

    result_dataframe = (
        dataframe.copy()
    )

    # --------------------------------------------------------
    # Optional facility filter
    # --------------------------------------------------------

    facility_id = None

    if request.method == "POST":

        body = request.get_json(
            silent=True
        )

        if isinstance(body, dict):

            facility_id = body.get(
                "facility_id"
            )

    else:

        facility_id = request.args.get(
            "facility_id"
        )

    if facility_id is not None:

        facility_column = find_column(
            result_dataframe,
            [
                "origin_facility_id",
                "facility_id",
                "origin_id",
            ]
        )

        if facility_column is not None:

            result_dataframe = (
                result_dataframe[
                    result_dataframe[
                        facility_column
                    ]
                    .astype(str)
                    == str(facility_id)
                ]
            )

    # --------------------------------------------------------
    # Optional limit
    # --------------------------------------------------------

    try:

        limit = int(
            request.args.get(
                "limit",
                1000
            )
        )

    except (TypeError, ValueError):

        limit = 1000

    limit = max(
        1,
        min(
            limit,
            5000
        )
    )

    result_dataframe = (
        result_dataframe
        .head(limit)
    )

    records = (
        dataframe_to_records(
            result_dataframe
        )
    )

    return jsonify(
        {
            "status": "ok",
            "cascade": records,
            "count": len(records),
        }
    )


# ============================================================
# INTERVENTION ENDPOINT
# ============================================================

@app.route(
    "/api/intervention",
    methods=["GET", "POST"]
)
def intervention():

    dataframe = DATA["intervention"]

    if dataframe.empty:

        return jsonify(
            {
                "status": "ok",
                "interventions": [],
                "count": 0,
            }
        )

    result_dataframe = (
        dataframe.copy()
    )

    # --------------------------------------------------------
    # Optional recipient filter
    # --------------------------------------------------------

    recipient_id = None

    if request.method == "POST":

        body = request.get_json(
            silent=True
        )

        if isinstance(body, dict):

            recipient_id = body.get(
                "recipient_id"
            )

    else:

        recipient_id = request.args.get(
            "recipient_id"
        )

    if recipient_id is not None:

        recipient_column = find_column(
            result_dataframe,
            [
                "recipient_facility_id",
                "recipient_id",
            ]
        )

        if recipient_column is not None:

            result_dataframe = (
                result_dataframe[
                    result_dataframe[
                        recipient_column
                    ]
                    .astype(str)
                    == str(recipient_id)
                ]
            )

    records = (
        dataframe_to_records(
            result_dataframe
        )
    )

    return jsonify(
        {
            "status": "ok",
            "interventions": records,
            "count": len(records),
        }
    )


# ============================================================
# REDISTRIBUTION ENDPOINT
# ============================================================

@app.route(
    "/api/redistribution/<recipient_id>",
    methods=["GET"]
)
def redistribution(recipient_id):

    dataframe = DATA["feasibility"]

    if dataframe.empty:

        return jsonify(
            {
                "status": "ok",
                "recipient_id": recipient_id,
                "options": [],
                "count": 0,
            }
        )

    recipient_column = find_column(
        dataframe,
        [
            "recipient_facility_id",
            "recipient_id",
        ]
    )

    if recipient_column is None:

        return jsonify(
            {
                "status": "ok",
                "recipient_id": recipient_id,
                "options": [],
                "count": 0,
            }
        )

    result_dataframe = dataframe[
        dataframe[recipient_column]
        .astype(str)
        == str(recipient_id)
    ].copy()

    # --------------------------------------------------------
    # Sort by feasibility score
    # --------------------------------------------------------

    score_column = find_column(
        result_dataframe,
        [
            "feasibility_score",
            "score",
        ]
    )

    if score_column is not None:

        result_dataframe = (
            result_dataframe
            .sort_values(
                score_column,
                ascending=False
            )
        )

    # --------------------------------------------------------
    # Limit
    # --------------------------------------------------------

    try:

        limit = int(
            request.args.get(
                "limit",
                20
            )
        )

    except (TypeError, ValueError):

        limit = 20

    limit = max(
        1,
        min(
            limit,
            100
        )
    )

    result_dataframe = (
        result_dataframe
        .head(limit)
    )

    records = (
        dataframe_to_records(
            result_dataframe
        )
    )

    return jsonify(
        {
            "status": "ok",
            "recipient_id": recipient_id,
            "options": records,
            "count": len(records),
        }
    )


# ============================================================
# GENERAL API INFORMATION
# ============================================================

@app.route(
    "/api",
    methods=["GET"]
)
def api_information():

    return jsonify(
        {
            "service":
                "SupplyShield AI Backend",

            "status":
                "ok",

            "endpoints": [

                "/api/health",

                "/api/summary",

                "/api/alerts",

                "/api/alerts/explanations",

                "/api/facility/<facility_id>",

                "/api/cascade",

                "/api/intervention",

                "/api/redistribution/<recipient_id>",

            ],
        }
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify(
        {
            "status": "error",
            "error": "Endpoint not found",
        }
    ), 404


@app.errorhandler(500)
def internal_error(error):

    return jsonify(
        {
            "status": "error",
            "error": "Internal server error",
            "details": str(error),
        }
    ), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("SUPPLYSHIELD AI BACKEND")
    print("=" * 60)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Processed data: {PROCESSED_DIR}"
    )

    print()

    print(
        "CORS enabled for /api/*"
    )

    print(
        "Running on http://127.0.0.1:5000"
    )

    print(
        "Press CTRL+C to quit"
    )

    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )