import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ANOMALY_FILE = "data/processed/anomaly_signals.csv"

OUTPUT_DIR = "data/processed"

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "anomaly_signal_validation.csv"
)

BIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "anomaly_score_validation.csv"
)

TYPE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "anomaly_type_validation.csv"
)


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def find_column(df, candidates):

    normalized = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = str(candidate).lower()

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SUPPLYSHIELD AI - ANOMALY SIGNAL VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. LOAD ANOMALY SIGNALS
    # --------------------------------------------------------

    print_section("1. LOADING ANOMALY SIGNALS")

    if not os.path.exists(ANOMALY_FILE):

        raise FileNotFoundError(
            f"Anomaly signal file not found: {ANOMALY_FILE}"
        )

    df = pd.read_csv(
        ANOMALY_FILE
    )

    print(
        f"Records: {len(df):,}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    # --------------------------------------------------------
    # 2. IDENTIFY COLUMNS
    # --------------------------------------------------------

    print_section("2. IDENTIFYING ANOMALY COLUMNS")

    candidates = {

        "facility_id": [
            "facility_id",
            "hf_pk",
        ],

        "date": [
            "date",
        ],

        "stockout": [
            "stockout",
            "current_stockout",
        ],

        "combined_score": [
            "combined_anomaly_score",
            "anomaly_score",
            "combined_score",
        ],

        # IMPORTANT:
        # The actual anomaly_signals.csv contains
        # "shortage_anomaly".
        #
        # We therefore check it first.
        "shortage_relevant": [
            "shortage_anomaly",
            "shortage_relevant_anomaly",
            "shortage_relevant",
        ],

        "consumption_anomaly": [
            "consumption_anomaly",
        ],

        "inventory_anomaly": [
            "inventory_anomaly",
        ],

        "low_inventory": [
            "low_inventory_anomaly",
            "low_inventory",
        ],

        "high_consumption": [
            "high_consumption_anomaly",
            "high_consumption",
        ],
    }

    resolved = {}

    for name, options in candidates.items():

        column = find_column(
            df,
            options
        )

        if column is None:

            raise ValueError(
                f"Could not identify column for '{name}'. "
                f"Available columns: {list(df.columns)}"
            )

        resolved[name] = column

        print(
            f"{name}: {column}"
        )

    facility_col = resolved["facility_id"]
    date_col = resolved["date"]
    stockout_col = resolved["stockout"]
    score_col = resolved["combined_score"]
    shortage_col = resolved["shortage_relevant"]
    consumption_anomaly_col = resolved["consumption_anomaly"]
    inventory_anomaly_col = resolved["inventory_anomaly"]
    low_inventory_col = resolved["low_inventory"]
    high_consumption_col = resolved["high_consumption"]

    # --------------------------------------------------------
    # 3. CLEAN DATA
    # --------------------------------------------------------

    print_section("3. PREPARING VALIDATION DATA")

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    numeric_columns = [
        stockout_col,
        score_col,
        shortage_col,
        consumption_anomaly_col,
        inventory_anomaly_col,
        low_inventory_col,
        high_consumption_col,
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            date_col,
            stockout_col,
            score_col,
            shortage_col,
            consumption_anomaly_col,
            inventory_anomaly_col,
            low_inventory_col,
            high_consumption_col,
        ]
    ).copy()

    # Convert binary indicators to 0/1.

    df[stockout_col] = (
        df[stockout_col] > 0
    ).astype(int)

    df[shortage_col] = (
        df[shortage_col] > 0
    ).astype(int)

    df[consumption_anomaly_col] = (
        df[consumption_anomaly_col] > 0
    ).astype(int)

    df[inventory_anomaly_col] = (
        df[inventory_anomaly_col] > 0
    ).astype(int)

    df[low_inventory_col] = (
        df[low_inventory_col] > 0
    ).astype(int)

    df[high_consumption_col] = (
        df[high_consumption_col] > 0
    ).astype(int)

    print(
        f"Valid records: {len(df):,}"
    )

    print(
        f"Observed stockout rate: "
        f"{df[stockout_col].mean():.2%}"
    )

    print(
        f"Shortage-relevant anomaly rate: "
        f"{df[shortage_col].mean():.2%}"
    )

    # --------------------------------------------------------
    # 4. STOCKOUT RATE: ANOMALY VS NO ANOMALY
    # --------------------------------------------------------

    print_section(
        "4. STOCKOUT RATE: ANOMALY VS NO ANOMALY"
    )

    anomaly_group = (
        df.groupby(
            shortage_col
        )[stockout_col]
        .agg(
            records="count",
            stockouts="sum",
            stockout_rate="mean",
        )
        .reset_index()
    )

    anomaly_group["anomaly_status"] = (
        anomaly_group[
            shortage_col
        ]
        .map(
            {
                0: "No shortage anomaly",
                1: "Shortage-relevant anomaly",
            }
        )
    )

    anomaly_group = anomaly_group[
        [
            "anomaly_status",
            "records",
            "stockouts",
            "stockout_rate",
        ]
    ]

    print(
        anomaly_group.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 5. ANOMALY SIGNAL LIFT
    # --------------------------------------------------------

    print_section(
        "5. ANOMALY SIGNAL LIFT"
    )

    anomaly_rate = (
        df.loc[
            df[shortage_col] == 1,
            stockout_col
        ].mean()
    )

    no_anomaly_rate = (
        df.loc[
            df[shortage_col] == 0,
            stockout_col
        ].mean()
    )

    if (
        not np.isnan(no_anomaly_rate)
        and no_anomaly_rate > 0
    ):

        anomaly_ratio = (
            anomaly_rate
            / no_anomaly_rate
        )

    else:

        anomaly_ratio = np.nan

    print(
        f"No-anomaly stockout rate: "
        f"{no_anomaly_rate:.2%}"
    )

    print(
        f"Anomaly stockout rate: "
        f"{anomaly_rate:.2%}"
    )

    print(
        f"Anomaly rate ratio: "
        f"{anomaly_ratio:.2f}x"
    )

    # --------------------------------------------------------
    # 6. INDIVIDUAL ANOMALY TYPES
    # --------------------------------------------------------

    print_section(
        "6. STOCKOUT RATE BY ANOMALY TYPE"
    )

    anomaly_types = [
        (
            "Consumption anomaly",
            consumption_anomaly_col
        ),
        (
            "Inventory anomaly",
            inventory_anomaly_col
        ),
        (
            "Low-inventory anomaly",
            low_inventory_col
        ),
        (
            "High-consumption anomaly",
            high_consumption_col
        ),
    ]

    type_results = []

    for name, column in anomaly_types:

        anomaly_mask = (
            df[column] == 1
        )

        normal_mask = (
            df[column] == 0
        )

        anomaly_stockout_rate = (
            df.loc[
                anomaly_mask,
                stockout_col
            ].mean()
        )

        normal_stockout_rate = (
            df.loc[
                normal_mask,
                stockout_col
            ].mean()
        )

        if (
            not np.isnan(normal_stockout_rate)
            and normal_stockout_rate > 0
        ):

            ratio = (
                anomaly_stockout_rate
                / normal_stockout_rate
            )

        else:

            ratio = np.nan

        type_results.append(
            {
                "anomaly_type": name,
                "anomaly_records": int(
                    anomaly_mask.sum()
                ),
                "anomaly_stockout_rate": (
                    anomaly_stockout_rate
                ),
                "normal_stockout_rate": (
                    normal_stockout_rate
                ),
                "rate_ratio": ratio,
            }
        )

        print(
            f"{name}: "
            f"anomaly_rate="
            f"{anomaly_stockout_rate:.2%} | "
            f"normal_rate="
            f"{normal_stockout_rate:.2%} | "
            f"ratio="
            f"{ratio:.2f}x"
        )

    type_df = pd.DataFrame(
        type_results
    )

    # --------------------------------------------------------
    # 7. COMBINED ANOMALY SCORE
    # --------------------------------------------------------

    print_section(
        "7. STOCKOUT RATE BY ANOMALY SCORE"
    )

    score_min = df[score_col].min()
    score_max = df[score_col].max()

    print(
        f"Anomaly score range: "
        f"{score_min:.4f} - {score_max:.4f}"
    )

    # The combined anomaly score is not a probability.
    # Therefore we evaluate relative score quantiles.

    try:

        score_bins = pd.qcut(
            df[score_col],
            q=5,
            duplicates="drop"
        )

        score_summary = (
            df.groupby(
                score_bins,
                observed=False
            )
            .agg(
                records=(
                    stockout_col,
                    "count"
                ),
                stockouts=(
                    stockout_col,
                    "sum"
                ),
                stockout_rate=(
                    stockout_col,
                    "mean"
                ),
                average_score=(
                    score_col,
                    "mean"
                ),
            )
            .reset_index()
        )

        print(
            score_summary.to_string(
                index=False
            )
        )

    except ValueError:

        print(
            "Could not create five score quantiles "
            "because there is insufficient score variation."
        )

        score_summary = pd.DataFrame()

    # --------------------------------------------------------
    # 8. MONTHLY ANOMALY VALIDATION
    # --------------------------------------------------------

    print_section(
        "8. MONTHLY ANOMALY VALIDATION"
    )

    monthly = (
        df.groupby(
            date_col
        )
        .agg(
            records=(
                facility_col,
                "count"
            ),

            observed_stockouts=(
                stockout_col,
                "sum"
            ),

            stockout_rate=(
                stockout_col,
                "mean"
            ),

            shortage_anomalies=(
                shortage_col,
                "sum"
            ),

            anomaly_fraction=(
                shortage_col,
                "mean"
            ),

            average_anomaly_score=(
                score_col,
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        monthly.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 9. MONTHS WITH HIGHEST ANOMALY ACTIVITY
    # --------------------------------------------------------

    print_section(
        "9. MONTHS WITH HIGHEST ANOMALY ACTIVITY"
    )

    top_months = (
        monthly
        .sort_values(
            "anomaly_fraction",
            ascending=False
        )
        .head(10)
    )

    print(
        top_months[
            [
                date_col,
                "records",
                "stockout_rate",
                "shortage_anomalies",
                "anomaly_fraction",
                "average_anomaly_score",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 10. SAVE SUMMARY
    # --------------------------------------------------------

    print_section(
        "10. SAVING VALIDATION OUTPUTS"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    summary_rows = [
        {
            "metric": "valid_records",
            "value": len(df),
        },
        {
            "metric": "observed_stockout_rate",
            "value": df[stockout_col].mean(),
        },
        {
            "metric": "shortage_anomaly_rate",
            "value": df[shortage_col].mean(),
        },
        {
            "metric": "anomaly_stockout_rate",
            "value": anomaly_rate,
        },
        {
            "metric": "no_anomaly_stockout_rate",
            "value": no_anomaly_rate,
        },
        {
            "metric": "anomaly_rate_ratio",
            "value": anomaly_ratio,
        },
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )

    print(
        f"Saved: {SUMMARY_OUTPUT}"
    )

    # --------------------------------------------------------
    # 11. SAVE SCORE VALIDATION
    # --------------------------------------------------------

    if not score_summary.empty:

        score_summary.to_csv(
            BIN_OUTPUT,
            index=False
        )

        print(
            f"Saved: {BIN_OUTPUT}"
        )

    # --------------------------------------------------------
    # 12. SAVE ANOMALY TYPE RESULTS
    # --------------------------------------------------------

    type_df.to_csv(
        TYPE_OUTPUT,
        index=False
    )

    print(
        f"Saved: {TYPE_OUTPUT}"
    )

    # --------------------------------------------------------
    # 13. FINAL STATUS
    # --------------------------------------------------------

    print_section(
        "STAGE 11C COMPLETE"
    )

    print(
        "Anomaly signals have been compared "
        "against observed stockouts."
    )

    print(
        f"Anomaly stockout rate: "
        f"{anomaly_rate:.2%}"
    )

    print(
        f"No-anomaly stockout rate: "
        f"{no_anomaly_rate:.2%}"
    )

    if not np.isnan(anomaly_ratio):

        print(
            f"Anomaly rate ratio: "
            f"{anomaly_ratio:.2f}x"
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The anomaly score is an unusualness/evidence "
        "measure, not a probability."
    )

    print(
        "This validation measures association with "
        "observed stockouts; it does not establish causation."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()