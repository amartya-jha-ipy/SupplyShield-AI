import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/processed/prototype_paracetamol_500mg.csv"
)

OUTPUT_DIR = Path("data/processed")

MODEL_PATH = (
    OUTPUT_DIR / "shortage_risk_model.pkl"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR / "shortage_risk_predictions.csv"
)

TRAIN_DATA_PATH = (
    OUTPUT_DIR / "shortage_risk_train.csv"
)

TEST_DATA_PATH = (
    OUTPUT_DIR / "shortage_risk_test.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("SUPPLYSHIELD AI - NEXT-MONTH SHORTAGE RISK MODEL")
print("=" * 75)

if not DATA_PATH.exists():
    print("\nERROR: Prepared dataset not found:")
    print(DATA_PATH.resolve())
    raise SystemExit(1)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.sort_values(
    ["facility_id", "date"]
).reset_index(drop=True)

print(
    f"\nLoaded records: {len(df):,}"
)

print(
    f"Facilities: {df['facility_id'].nunique():,}"
)

print(
    f"Date range: "
    f"{df['date'].min().date()} "
    f"to "
    f"{df['date'].max().date()}"
)


# ============================================================
# CREATE NEXT-MONTH TARGET
# ============================================================

print("\n" + "=" * 75)
print("1. CREATING NEXT-MONTH STOCKOUT TARGET")
print("=" * 75)

# The target is the stockout status at the next
# observed month for the SAME facility.
#
# We first shift the data within each facility.

df["next_date"] = (
    df.groupby("facility_id")["date"]
    .shift(-1)
)

df["next_stockout"] = (
    df.groupby("facility_id")["stockout"]
    .shift(-1)
)


# Calculate the number of months between
# current observation and next observation.

df["month_gap"] = (
    (
        df["next_date"].dt.year
        - df["date"].dt.year
    ) * 12
    +
    (
        df["next_date"].dt.month
        - df["date"].dt.month
    )
)


# Only retain observations where the next
# observation is exactly one month later.

valid_target = (
    df["month_gap"] == 1
)

print(
    f"Rows with exactly one-month-ahead observation: "
    f"{valid_target.sum():,}"
)

print(
    f"Rows excluded because of missing/longer gap: "
    f"{(~valid_target).sum():,}"
)

model_df = df[valid_target].copy()

model_df["target_next_stockout"] = (
    model_df["next_stockout"]
    .astype(int)
)

print(
    "\nNext-month stockout rate: "
    f"{model_df['target_next_stockout'].mean() * 100:.2f}%"
)


# ============================================================
# CREATE MODEL FEATURES
# ============================================================

print("\n" + "=" * 75)
print("2. CREATING PREDICTION FEATURES")
print("=" * 75)


# ------------------------------------------------------------
# Current inventory
# ------------------------------------------------------------

feature_columns = [
    "closing_inventory",
    "opening_inventory",
    "received",
    "consumption",
    "rolling_consumption_3",
    "inventory_coverage",
    "inventory_change",
    "consumption_change",
    "stockout",
    "latitude",
    "longitude",
    "month",
    "year",
]


# ------------------------------------------------------------
# Add historical features
# ------------------------------------------------------------

# Previous month's consumption
model_df["previous_consumption"] = (
    model_df
    .groupby("facility_id")["consumption"]
    .shift(1)
)

# Previous month's inventory
model_df["previous_inventory"] = (
    model_df
    .groupby("facility_id")["closing_inventory"]
    .shift(1)
)

# Previous month's stockout
model_df["previous_stockout"] = (
    model_df
    .groupby("facility_id")["stockout"]
    .shift(1)
)


# ------------------------------------------------------------
# Consumption momentum
# ------------------------------------------------------------

model_df["consumption_ratio"] = np.where(
    model_df["previous_consumption"] > 0,
    model_df["consumption"]
    / model_df["previous_consumption"],
    np.nan,
)


# ------------------------------------------------------------
# Inventory pressure
# ------------------------------------------------------------

model_df["inventory_to_consumption"] = np.where(
    model_df["consumption"] > 0,
    model_df["closing_inventory"]
    / model_df["consumption"],
    np.nan,
)


# ------------------------------------------------------------
# Recent stockout history
# ------------------------------------------------------------

model_df["stockout_last_2_months"] = (
    model_df
    .groupby("facility_id")["stockout"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=2,
            min_periods=1
        )
        .sum()
    )
)


# ------------------------------------------------------------
# Update feature list
# ------------------------------------------------------------

feature_columns.extend(
    [
        "previous_consumption",
        "previous_inventory",
        "previous_stockout",
        "consumption_ratio",
        "inventory_to_consumption",
        "stockout_last_2_months",
    ]
)


print("\nFeatures used by the model:")

for feature in feature_columns:
    print(f"  - {feature}")


# ============================================================
# CLEAN MODEL DATA
# ============================================================

print("\n" + "=" * 75)
print("3. CLEANING MODEL INPUT")
print("=" * 75)

before_cleaning = len(model_df)

# Replace infinite values generated by ratios.

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)

# LightGBM can handle missing values, but we remove
# rows where essential prediction information is missing.

essential_columns = [
    "closing_inventory",
    "consumption",
    "target_next_stockout",
]

model_df = model_df.dropna(
    subset=essential_columns
)

print(
    f"Rows before cleaning: "
    f"{before_cleaning:,}"
)

print(
    f"Rows after cleaning: "
    f"{len(model_df):,}"
)


# ============================================================
# TEMPORAL TRAIN/TEST SPLIT
# ============================================================

print("\n" + "=" * 75)
print("4. TEMPORAL TRAIN/TEST SPLIT")
print("=" * 75)

# IMPORTANT:
# We do NOT randomly split the data.
#
# Earlier months -> training
# Later months  -> testing
#
# This better represents how the model would actually
# operate in the future.

unique_dates = np.sort(
    model_df["date"].unique()
)

split_index = int(
    len(unique_dates) * 0.80
)

train_dates = unique_dates[:split_index]

test_dates = unique_dates[split_index:]

train_df = model_df[
    model_df["date"].isin(train_dates)
].copy()

test_df = model_df[
    model_df["date"].isin(test_dates)
].copy()

print(
    f"Total unique dates: "
    f"{len(unique_dates)}"
)

print(
    f"Training dates: "
    f"{len(train_dates)}"
)

print(
    f"Testing dates: "
    f"{len(test_dates)}"
)

print(
    f"\nTraining period: "
    f"{train_df['date'].min().date()} "
    f"to "
    f"{train_df['date'].max().date()}"
)

print(
    f"Testing period: "
    f"{test_df['date'].min().date()} "
    f"to "
    f"{test_df['date'].max().date()}"
)

print(
    f"\nTraining rows: "
    f"{len(train_df):,}"
)

print(
    f"Testing rows: "
    f"{len(test_df):,}"
)


# ============================================================
# PREPARE X AND Y
# ============================================================

X_train = train_df[feature_columns].copy()

y_train = train_df[
    "target_next_stockout"
].astype(int)

X_test = test_df[feature_columns].copy()

y_test = test_df[
    "target_next_stockout"
].astype(int)


# ============================================================
# CLASS BALANCE
# ============================================================

print("\n" + "=" * 75)
print("5. TARGET DISTRIBUTION")
print("=" * 75)

train_positive = int(y_train.sum())
train_negative = int(
    len(y_train) - train_positive
)

test_positive = int(y_test.sum())
test_negative = int(
    len(y_test) - test_positive
)

print("\nTraining:")
print(
    f"  No stockout: {train_negative:,}"
)
print(
    f"  Stockout:    {train_positive:,}"
)
print(
    f"  Stockout rate: "
    f"{y_train.mean() * 100:.2f}%"
)

print("\nTesting:")
print(
    f"  No stockout: {test_negative:,}"
)
print(
    f"  Stockout:    {test_positive:,}"
)
print(
    f"  Stockout rate: "
    f"{y_test.mean() * 100:.2f}%"
)


# ============================================================
# CLASS WEIGHT
# ============================================================

if train_positive > 0:

    scale_pos_weight = (
        train_negative
        / train_positive
    )

else:

    print(
        "\nERROR: Training set contains "
        "no positive stockout examples."
    )

    raise SystemExit(1)


print(
    "\nClass weight:"
    f" {scale_pos_weight:.3f}"
)


# ============================================================
# TRAIN LIGHTGBM
# ============================================================

print("\n" + "=" * 75)
print("6. TRAINING LIGHTGBM MODEL")
print("=" * 75)

model = LGBMClassifier(
    objective="binary",
    n_estimators=250,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbosity=-1,
)

model.fit(
    X_train,
    y_train
)

print(
    "\nModel training complete."
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("7. GENERATING STOCKOUT RISK SCORES")
print("=" * 75)

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

# A simple operational threshold.
#
# This is NOT claimed to be an optimal threshold.
# It simply converts probability into a binary alert
# for demonstration.

ALERT_THRESHOLD = 0.50

test_predictions = (
    test_probabilities >= ALERT_THRESHOLD
).astype(int)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("8. MODEL EVALUATION")
print("=" * 75)

if len(np.unique(y_test)) >= 2:

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        test_probabilities
    )

    print(
        f"\nROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:  {pr_auc:.4f}"
    )

else:

    print(
        "\nWARNING: Test set contains only "
        "one target class."
    )


print(
    "\nClassification report:"
)

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

print(
    "Confusion matrix:"
)

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 75)
print("9. FEATURE IMPORTANCE")
print("=" * 75)

importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }
)

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("10. SAVING RESULTS")
print("=" * 75)

prediction_output = test_df[
    [
        "facility_id",
        "medicine",
        "date",
        "district",
        "latitude",
        "longitude",
        "closing_inventory",
        "consumption",
        "received",
        "stockout",
        "next_date",
        "target_next_stockout",
    ]
].copy()

prediction_output[
    "stockout_risk"
] = test_probabilities

prediction_output[
    "risk_alert"
] = test_predictions

prediction_output[
    "risk_level"
] = pd.cut(
    prediction_output["stockout_risk"],
    bins=[
        -np.inf,
        0.30,
        0.60,
        np.inf,
    ],
    labels=[
        "LOW",
        "MEDIUM",
        "HIGH",
    ]
)

prediction_output = prediction_output.sort_values(
    "stockout_risk",
    ascending=False
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

prediction_output.to_csv(
    PREDICTIONS_PATH,
    index=False
)

train_df.to_csv(
    TRAIN_DATA_PATH,
    index=False
)

test_df.to_csv(
    TEST_DATA_PATH,
    index=False
)

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# HIGH-RISK FACILITY PREVIEW
# ============================================================

print("\n" + "=" * 75)
print("11. HIGH-RISK FACILITY PREVIEW")
print("=" * 75)

display_columns = [
    "facility_id",
    "district",
    "date",
    "closing_inventory",
    "consumption",
    "stockout_risk",
    "risk_level",
    "target_next_stockout",
]

print(
    prediction_output[
        display_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STAGE 5 COMPLETE")
print("=" * 75)

print(
    "\nModel:"
    "\n  LightGBM binary classifier"
)

print(
    "\nPrediction:"
    "\n  Next observed month's stockout"
)

print(
    "\nTraining data:"
    f"\n  {len(train_df):,} rows"
)

print(
    "\nTesting data:"
    f"\n  {len(test_df):,} rows"
)

print(
    "\nSaved files:"
)

print(
    f"  {MODEL_PATH}"
)

print(
    f"  {PREDICTIONS_PATH}"
)

print(
    f"  {TRAIN_DATA_PATH}"
)

print(
    f"  {TEST_DATA_PATH}"
)

print(
    "\nThe stockout_risk column is the model's "
    "predicted probability of a next-month stockout."
)