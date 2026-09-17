import numpy as np
import pandas as pd

from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "supply_chain_data.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "demand_forecasts.csv"
)


# ============================================================
# 2. FEATURE CONFIGURATION
# ============================================================

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "day_of_week",
    "month",
    "trend"
]

TARGET_COLUMN = "demand"


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():
    """
    Load the synthetic supply-chain dataset.
    """

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "day",
        "facility_id",
        "demand"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Dataset is missing required columns: "
            + str(missing_columns)
        )

    return df


# ============================================================
# 4. CREATE TEMPORAL FEATURES
# ============================================================

def create_features(df):
    """
    Create temporal demand-forecasting features
    independently for every facility.
    """

    data = df.copy()

    data = data.sort_values(
        [
            "facility_id",
            "day"
        ]
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # Previous-day demand
    # --------------------------------------------------------

    data["lag_1"] = (
        data
        .groupby("facility_id")["demand"]
        .shift(1)
    )


    # --------------------------------------------------------
    # Same-day-of-week demand from previous week
    # --------------------------------------------------------

    data["lag_7"] = (
        data
        .groupby("facility_id")["demand"]
        .shift(7)
    )


    # --------------------------------------------------------
    # Day of week
    #
    # Day 1 -> Monday-like index 0
    # Day 7 -> index 6
    # --------------------------------------------------------

    data["day_of_week"] = (
        (data["day"] - 1) % 7
    )


    # --------------------------------------------------------
    # Month
    #
    # The synthetic dataset does not contain a real calendar
    # date, so we use a constant synthetic month.
    #
    # This keeps the feature available for future real-world
    # data where an actual date will be available.
    # --------------------------------------------------------

    data["month"] = 1


    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    data["trend"] = (
        data["day"]
        / max(data["day"].max(), 1)
    )


    return data


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

def split_data(data):
    """
    Perform a chronological split.

    The first 70% of observations for each facility are used
    for training and the remaining observations are used for
    testing.
    """

    train_parts = []
    test_parts = []

    for facility_id, facility_data in data.groupby(
        "facility_id"
    ):

        facility_data = facility_data.sort_values(
            "day"
        ).copy()

        split_index = int(
            len(facility_data) * 0.70
        )

        train_parts.append(
            facility_data.iloc[:split_index]
        )

        test_parts.append(
            facility_data.iloc[split_index:]
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True
    )

    return train_df, test_df


# ============================================================
# 6. TRAIN MODEL
# ============================================================

def train_model(train_df):
    """
    Train a LightGBM demand regression model.
    """

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    model = LGBMRegressor(
        objective="regression",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# 7. CALCULATE RESIDUAL UNCERTAINTY
# ============================================================

def calculate_uncertainty(
    model,
    train_df
):
    """
    Estimate forecast uncertainty from training residuals.

    This is a practical uncertainty estimate for the
    hackathon prototype rather than a formal probabilistic
    LightGBM prediction interval.
    """

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    predictions = model.predict(
        X_train
    )

    residuals = (
        y_train.to_numpy()
        - predictions
    )

    residual_std = float(
        np.std(residuals)
    )

    # Prevent zero-width intervals.
    residual_std = max(
        residual_std,
        0.5
    )

    return residual_std


# ============================================================
# 8. GENERATE FORECASTS
# ============================================================

def generate_forecasts(
    model,
    test_df,
    residual_std
):
    """
    Generate demand predictions and approximate prediction
    intervals.
    """

    result = test_df.copy()

    X_test = result[
        FEATURE_COLUMNS
    ]

    predictions = model.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0
    )

    result["predicted_demand"] = (
        predictions
    )

    # --------------------------------------------------------
    # Approximate 95% prediction interval
    # --------------------------------------------------------

    interval_margin = (
        1.96
        * residual_std
    )

    result["prediction_lower"] = np.maximum(
        result["predicted_demand"]
        - interval_margin,
        0
    )

    result["prediction_upper"] = (
        result["predicted_demand"]
        + interval_margin
    )


    # --------------------------------------------------------
    # Confidence from interval width
    #
    # Narrower interval -> higher confidence.
    # --------------------------------------------------------

    interval_width = (
        result["prediction_upper"]
        - result["prediction_lower"]
    )

    typical_demand = max(
        result["demand"].mean(),
        1
    )

    normalized_width = (
        interval_width
        / typical_demand
    )

    result["forecast_confidence"] = (
        1
        / (1 + normalized_width)
    )

    result["forecast_confidence"] = (
        result["forecast_confidence"]
        .clip(0, 1)
    )

    return result


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def run_forecasting_pipeline():
    """
    Execute the complete demand forecasting pipeline.
    """

    print("\n" + "=" * 60)

    print(
        "SUPPLYSHIELD AI - DEMAND FORECASTER"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "\n[1/6] Loading dataset..."
    )

    df = load_data()

    print(
        f"Loaded {len(df)} records."
    )


    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print(
        "\n[2/6] Creating temporal features..."
    )

    data = create_features(
        df
    )


    # --------------------------------------------------------
    # Remove rows where lag features are unavailable
    # --------------------------------------------------------

    model_data = data.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    print(
        f"Usable records after lag creation: "
        f"{len(model_data)}"
    )


    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    print(
        "\n[3/6] Creating chronological train/test split..."
    )

    train_df, test_df = split_data(
        model_data
    )

    print(
        f"Training records: {len(train_df)}"
    )

    print(
        f"Testing records : {len(test_df)}"
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\n[4/6] Training LightGBM model..."
    )

    model = train_model(
        train_df
    )

    print(
        "Model training complete."
    )


    # --------------------------------------------------------
    # Training uncertainty
    # --------------------------------------------------------

    residual_std = calculate_uncertainty(
        model,
        train_df
    )

    print(
        f"Residual standard deviation: "
        f"{residual_std:.4f}"
    )


    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    print(
        "\n[5/6] Generating demand forecasts..."
    )

    forecasts = generate_forecasts(
        model,
        test_df,
        residual_std
    )


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            forecasts["demand"],
            forecasts["predicted_demand"]
        )
    )

    print(
        f"Test RMSE: {rmse:.4f}"
    )


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    print(
        "\n[6/6] Saving forecast results..."
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    forecasts.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Forecasts saved to:\n{OUTPUT_PATH}"
    )


    # --------------------------------------------------------
    # Display sample
    # --------------------------------------------------------

    print(
        "\nForecast sample:"
    )

    print(
        forecasts[
            [
                "day",
                "facility_id",
                "demand",
                "predicted_demand",
                "prediction_lower",
                "prediction_upper",
                "forecast_confidence"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


    print(
        "\nDemand forecasting complete."
    )


    return (
        model,
        forecasts,
        rmse
    )


# ============================================================
# 10. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_forecasting_pipeline()