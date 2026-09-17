import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

SIGNAL_FILE = "data/processed/spatial_shortage_signals.csv"

OUTPUT_DIR = "data/processed"

FACILITY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "spatial_signal_validation.csv"
)

MONTH_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "spatial_month_validation.csv"
)


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def find_column(df, candidates):
    """
    Return the first matching column from a list of candidates.
    Matching is case-insensitive.
    """
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
    print("SUPPLYSHIELD AI - SPATIAL SIGNAL VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. LOAD SPATIAL SIGNALS
    # --------------------------------------------------------

    print_section("1. LOADING SPATIAL SHORTAGE SIGNALS")

    if not os.path.exists(SIGNAL_FILE):
        raise FileNotFoundError(
            f"Spatial signal file not found: {SIGNAL_FILE}"
        )

    df = pd.read_csv(SIGNAL_FILE)

    print(f"Records: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # --------------------------------------------------------
    # 2. IDENTIFY COLUMNS
    # --------------------------------------------------------

    print_section("2. IDENTIFYING SIGNAL COLUMNS")

    required_candidates = {

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

        "regional_signal": [
            "regional_signal",
        ],

        "local_pressure": [
            "local_shortage_pressure",
        ],

        "nearby_rate": [
            "nearby_stockout_rate",
        ],

        "nearby_count": [
            "nearby_facilities",
        ],
    }

    resolved = {}

    for name, candidates in required_candidates.items():

        found = find_column(
            df,
            candidates
        )

        if found is None:

            raise ValueError(
                f"Could not find required column for '{name}'. "
                f"Available columns: {list(df.columns)}"
            )

        resolved[name] = found

        print(
            f"{name}: {found}"
        )

    facility_col = resolved["facility_id"]
    date_col = resolved["date"]
    stockout_col = resolved["stockout"]
    signal_col = resolved["regional_signal"]
    pressure_col = resolved["local_pressure"]
    nearby_rate_col = resolved["nearby_rate"]
    nearby_count_col = resolved["nearby_count"]

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
        signal_col,
        pressure_col,
        nearby_rate_col,
        nearby_count_col,
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
            signal_col,
            pressure_col,
            nearby_rate_col,
            nearby_count_col,
        ]
    ).copy()

    df[stockout_col] = (
        df[stockout_col] > 0
    ).astype(int)

    df[signal_col] = (
        df[signal_col] > 0
    ).astype(int)

    print(
        f"Valid records: {len(df):,}"
    )

    print(
        f"Observed stockout rate: "
        f"{df[stockout_col].mean():.2%}"
    )

    print(
        f"Regional signal rate: "
        f"{df[signal_col].mean():.2%}"
    )

    # --------------------------------------------------------
    # 4. COMPARE SIGNAL VS NO SIGNAL
    # --------------------------------------------------------

    print_section(
        "4. STOCKOUT RATE: SIGNAL VS NO SIGNAL"
    )

    signal_group = (
        df.groupby(signal_col)[stockout_col]
        .agg(
            records="count",
            stockouts="sum",
            stockout_rate="mean",
        )
        .reset_index()
    )

    signal_group["signal_status"] = (
        signal_group[signal_col]
        .map(
            {
                0: "No regional signal",
                1: "Regional signal",
            }
        )
    )

    signal_group = signal_group[
        [
            "signal_status",
            "records",
            "stockouts",
            "stockout_rate",
        ]
    ]

    print(
        signal_group.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 5. CALCULATE SPATIAL SIGNAL RATE RATIO
    # --------------------------------------------------------

    print_section(
        "5. SPATIAL SIGNAL LIFT"
    )

    no_signal_mask = (
        df[signal_col] == 0
    )

    signal_mask = (
        df[signal_col] == 1
    )

    no_signal_rate = (
        df.loc[
            no_signal_mask,
            stockout_col
        ].mean()
    )

    signal_rate = (
        df.loc[
            signal_mask,
            stockout_col
        ].mean()
    )

    if (
        not np.isnan(no_signal_rate)
        and no_signal_rate > 0
    ):

        rate_ratio = (
            signal_rate
            / no_signal_rate
        )

    else:

        rate_ratio = np.nan

    print(
        f"No-signal stockout rate: "
        f"{no_signal_rate:.2%}"
    )

    print(
        f"Signal stockout rate: "
        f"{signal_rate:.2%}"
    )

    print(
        f"Spatial signal rate ratio: "
        f"{rate_ratio:.2f}x"
    )

    # --------------------------------------------------------
    # 6. LOCAL PRESSURE BINS
    # --------------------------------------------------------

    print_section(
        "6. STOCKOUT RATE BY LOCAL SHORTAGE PRESSURE"
    )

    pressure_bins = [
        -0.001,
        0.20,
        0.40,
        0.60,
        0.80,
        1.01,
    ]

    pressure_labels = [
        "0.00-0.20",
        "0.20-0.40",
        "0.40-0.60",
        "0.60-0.80",
        "0.80-1.00",
    ]

    df["pressure_bin"] = pd.cut(
        df[pressure_col],
        bins=pressure_bins,
        labels=pressure_labels,
        include_lowest=True,
    )

    pressure_summary = (
        df.groupby(
            "pressure_bin",
            observed=False
        )[stockout_col]
        .agg(
            records="count",
            stockouts="sum",
            stockout_rate="mean",
        )
        .reset_index()
    )

    print(
        pressure_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 7. NEARBY STOCKOUT RATE BINS
    # --------------------------------------------------------

    print_section(
        "7. STOCKOUT RATE BY NEARBY STOCKOUT RATE"
    )

    nearby_bins = [
        -0.001,
        0.20,
        0.40,
        0.60,
        0.80,
        1.01,
    ]

    nearby_labels = [
        "0.00-0.20",
        "0.20-0.40",
        "0.40-0.60",
        "0.60-0.80",
        "0.80-1.00",
    ]

    df["nearby_rate_bin"] = pd.cut(
        df[nearby_rate_col],
        bins=nearby_bins,
        labels=nearby_labels,
        include_lowest=True,
    )

    nearby_summary = (
        df.groupby(
            "nearby_rate_bin",
            observed=False
        )[stockout_col]
        .agg(
            records="count",
            stockouts="sum",
            stockout_rate="mean",
        )
        .reset_index()
    )

    print(
        nearby_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 8. MONTHLY VALIDATION
    # --------------------------------------------------------

    print_section(
        "8. MONTHLY SPATIAL VALIDATION"
    )

    monthly = (
        df.groupby(date_col)
        .agg(
            facilities=(
                facility_col,
                "nunique"
            ),

            observed_stockouts=(
                stockout_col,
                "sum"
            ),

            overall_stockout_rate=(
                stockout_col,
                "mean"
            ),

            regional_signal_facilities=(
                signal_col,
                "sum"
            ),

            average_local_pressure=(
                pressure_col,
                "mean"
            ),

            average_nearby_stockout_rate=(
                nearby_rate_col,
                "mean"
            ),
        )
        .reset_index()
    )

    monthly[
        "regional_signal_fraction"
    ] = (
        monthly[
            "regional_signal_facilities"
        ]
        / monthly["facilities"]
    )

    print(
        monthly[
            [
                date_col,
                "facilities",
                "overall_stockout_rate",
                "regional_signal_facilities",
                "regional_signal_fraction",
                "average_local_pressure",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 9. HIGHEST-PRESSURE MONTHS
    # --------------------------------------------------------

    print_section(
        "9. MONTHS WITH HIGHEST SPATIAL PRESSURE"
    )

    top_months = (
        monthly
        .sort_values(
            "average_local_pressure",
            ascending=False
        )
        .head(10)
    )

    print(
        top_months[
            [
                date_col,
                "facilities",
                "overall_stockout_rate",
                "regional_signal_facilities",
                "regional_signal_fraction",
                "average_local_pressure",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 10. SAVE FACILITY-LEVEL VALIDATION
    # --------------------------------------------------------

    print_section(
        "10. SAVING VALIDATION OUTPUTS"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    facility_output = signal_group.copy()

    facility_output[
        "no_signal_stockout_rate"
    ] = no_signal_rate

    facility_output[
        "signal_stockout_rate"
    ] = signal_rate

    facility_output[
        "signal_rate_ratio"
    ] = rate_ratio

    facility_output.to_csv(
        FACILITY_OUTPUT,
        index=False
    )

    print(
        f"Saved: {FACILITY_OUTPUT}"
    )

    # --------------------------------------------------------
    # 11. SAVE MONTHLY VALIDATION
    # --------------------------------------------------------

    monthly.to_csv(
        MONTH_OUTPUT,
        index=False
    )

    print(
        f"Saved: {MONTH_OUTPUT}"
    )

    # --------------------------------------------------------
    # 12. FINAL STATUS
    # --------------------------------------------------------

    print_section(
        "STAGE 11B COMPLETE"
    )

    print(
        "Spatial shortage signals have been compared "
        "against observed stockouts."
    )

    print(
        f"Signal stockout rate: "
        f"{signal_rate:.2%}"
    )

    print(
        f"No-signal stockout rate: "
        f"{no_signal_rate:.2%}"
    )

    if not np.isnan(rate_ratio):

        print(
            f"Signal rate ratio: "
            f"{rate_ratio:.2f}x"
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This validation measures association between "
        "the spatial signal and observed stockouts."
    )

    print(
        "It does not prove that spatial proximity "
        "caused the stockout."
    )


if __name__ == "__main__":
    main()