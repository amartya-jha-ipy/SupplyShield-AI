import numpy as np
import pandas as pd

from pathlib import Path
from itertools import combinations
from scipy.stats import pearsonr


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
    / "synchronization_results.csv"
)


# ============================================================
# 2. DETECTOR CONFIGURATION
# ============================================================

CORRELATION_THRESHOLD = 0.72

P_VALUE_THRESHOLD = 0.05

MAX_LAG_DAYS = 2


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():
    """
    Load the supply-chain dataset.
    """

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    required_columns = [
        "day",
        "facility_id",
        "supplier_id",
        "inventory",
        "supplier_disrupted"
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
# 4. CALCULATE DEPLETION VELOCITY
# ============================================================

def calculate_depletion_velocity(df):
    """
    Calculate daily inventory change for each facility.

    ΔS(i,t) = S(i,t) - S(i,t-1)

    Negative values indicate inventory depletion.
    Positive values indicate inventory increase.
    """

    data = df.copy()

    data = data.sort_values(
        [
            "facility_id",
            "day"
        ]
    ).reset_index(drop=True)

    data["depletion_velocity"] = (
        data
        .groupby("facility_id")["inventory"]
        .diff()
    )

    return data


# ============================================================
# 5. NORMALIZE A SIGNAL
# ============================================================

def normalize_signal(signal):
    """
    Standardize a depletion-velocity signal.

    This removes differences in scale between facilities.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    mean = np.mean(signal)

    std = np.std(signal)

    if std == 0:

        return np.zeros_like(
            signal
        )

    return (
        (signal - mean)
        / std
    )


# ============================================================
# 6. LAGGED CORRELATION
# ============================================================

def calculate_lagged_correlation(
    signal_a,
    signal_b,
    lag
):
    """
    Calculate Pearson correlation between two signals
    at a specified time lag.

    Positive lag means signal B is shifted forward relative
    to signal A.
    """

    signal_a = np.asarray(
        signal_a,
        dtype=float
    )

    signal_b = np.asarray(
        signal_b,
        dtype=float
    )


    if lag > 0:

        a = signal_a[:-lag]

        b = signal_b[lag:]


    elif lag < 0:

        a = signal_a[-lag:]

        b = signal_b[:lag]


    else:

        a = signal_a

        b = signal_b


    if len(a) < 3:

        return np.nan, np.nan


    # --------------------------------------------------------
    # Normalize the two signals.
    # --------------------------------------------------------

    a = normalize_signal(a)

    b = normalize_signal(b)


    # --------------------------------------------------------
    # Constant signals cannot produce meaningful correlation.
    # --------------------------------------------------------

    if (
        np.std(a) == 0
        or np.std(b) == 0
    ):

        return np.nan, np.nan


    correlation, p_value = pearsonr(
        a,
        b
    )

    return (
        float(correlation),
        float(p_value)
    )


# ============================================================
# 7. FIND MAXIMUM CROSS-CORRELATION
# ============================================================

def find_best_synchronization(
    signal_a,
    signal_b,
    max_lag=MAX_LAG_DAYS
):
    """
    Search over time lags from -max_lag to +max_lag and
    return the strongest positive correlation.
    """

    results = []


    for lag in range(
        -max_lag,
        max_lag + 1
    ):

        correlation, p_value = (
            calculate_lagged_correlation(
                signal_a,
                signal_b,
                lag
            )
        )

        if not np.isnan(correlation):

            results.append(
                {
                    "lag_days": lag,

                    "correlation": correlation,

                    "p_value": p_value
                }
            )


    if not results:

        return {
            "lag_days": np.nan,
            "correlation": np.nan,
            "p_value": np.nan
        }


    # --------------------------------------------------------
    # Select the strongest positive correlation.
    # --------------------------------------------------------

    best_result = max(
        results,
        key=lambda x: x["correlation"]
    )

    return best_result


# ============================================================
# 8. DETECT SYNCHRONIZED FACILITY PAIRS
# ============================================================

def detect_synchronized_pairs(
    data
):
    """
    Compare every pair of facilities.

    A pair is considered synchronized when:

        correlation >= 0.72
        AND
        p-value < 0.05
    """

    facilities = sorted(
        data["facility_id"]
        .unique()
        .tolist()
    )

    results = []


    # --------------------------------------------------------
    # Create inventory depletion signals.
    # --------------------------------------------------------

    signals = {}


    for facility_id in facilities:

        facility_data = (
            data[
                data["facility_id"]
                == facility_id
            ]
            .sort_values("day")
        )

        signals[facility_id] = (
            facility_data[
                "depletion_velocity"
            ]
            .fillna(0)
            .to_numpy()
        )


    # --------------------------------------------------------
    # Compare every unique facility pair.
    # --------------------------------------------------------

    for facility_a, facility_b in combinations(
        facilities,
        2
    ):

        result = find_best_synchronization(
            signals[facility_a],
            signals[facility_b]
        )

        correlation = result[
            "correlation"
        ]

        p_value = result[
            "p_value"
        ]

        synchronized = (
            not np.isnan(correlation)
            and not np.isnan(p_value)
            and correlation
            >= CORRELATION_THRESHOLD
            and p_value
            < P_VALUE_THRESHOLD
        )


        # ----------------------------------------------------
        # Supplier relationship
        # ----------------------------------------------------

        supplier_a = (
            data[
                data["facility_id"]
                == facility_a
            ]["supplier_id"]
            .iloc[0]
        )

        supplier_b = (
            data[
                data["facility_id"]
                == facility_b
            ]["supplier_id"]
            .iloc[0]
        )

        shared_supplier = (
            supplier_a
            == supplier_b
        )


        results.append(
            {
                "facility_a": facility_a,

                "facility_b": facility_b,

                "supplier_a": supplier_a,

                "supplier_b": supplier_b,

                "shared_supplier": (
                    shared_supplier
                ),

                "best_lag_days": result[
                    "lag_days"
                ],

                "correlation": correlation,

                "p_value": p_value,

                "synchronized": synchronized
            }
        )


    return pd.DataFrame(
        results
    )


# ============================================================
# 9. DETECT SUPPLIER DISRUPTIONS
# ============================================================

def detect_supplier_disruptions(
    data
):
    """
    Summarize disruption activity by supplier.
    """

    supplier_summary = (
        data
        .groupby("supplier_id")
        .agg(
            total_records=(
                "supplier_disrupted",
                "count"
            ),

            disrupted_records=(
                "supplier_disrupted",
                "sum"
            ),

            affected_facilities=(
                "facility_id",
                "nunique"
            )
        )
        .reset_index()
    )


    supplier_summary[
        "disruption_rate"
    ] = (
        supplier_summary[
            "disrupted_records"
        ]
        / supplier_summary[
            "total_records"
        ]
    )


    supplier_summary[
        "supplier_disrupted"
    ] = (
        supplier_summary[
            "disrupted_records"
        ]
        > 0
    )


    return supplier_summary


# ============================================================
# 10. SAVE RESULTS
# ============================================================

def save_results(
    synchronization_results,
    supplier_results
):
    """
    Save synchronization results.

    Supplier-level results are stored in a separate CSV.
    """

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    synchronization_results.to_csv(
        OUTPUT_PATH,
        index=False
    )


    supplier_output_path = (
        PROJECT_ROOT
        / "data"
        / "synthetic"
        / "supplier_disruptions.csv"
    )

    supplier_results.to_csv(
        supplier_output_path,
        index=False
    )

    return supplier_output_path


# ============================================================
# 11. MAIN PIPELINE
# ============================================================

def run_synchronization_pipeline():
    """
    Execute the complete synchronization-detection pipeline.
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SUPPLYSHIELD AI - SYNCHRONIZATION DETECTOR"
    )

    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "\n[1/5] Loading dataset..."
    )

    df = load_data()

    print(
        f"Loaded {len(df)} records."
    )


    # --------------------------------------------------------
    # Depletion velocity
    # --------------------------------------------------------

    print(
        "\n[2/5] Calculating depletion velocity..."
    )

    data = calculate_depletion_velocity(
        df
    )

    print(
        "Depletion velocity calculated."
    )


    # --------------------------------------------------------
    # Synchronization
    # --------------------------------------------------------

    print(
        "\n[3/5] Detecting synchronized facility pairs..."
    )

    synchronization_results = (
        detect_synchronized_pairs(
            data
        )
    )


    synchronized_count = int(
        synchronization_results[
            "synchronized"
        ].sum()
    )


    print(
        f"Facility pairs analyzed: "
        f"{len(synchronization_results)}"
    )

    print(
        f"Synchronized pairs found: "
        f"{synchronized_count}"
    )


    # --------------------------------------------------------
    # Supplier disruption
    # --------------------------------------------------------

    print(
        "\n[4/5] Analyzing supplier disruptions..."
    )

    supplier_results = (
        detect_supplier_disruptions(
            data
        )
    )


    print(
        "\nSupplier disruption summary:"
    )

    print(
        supplier_results.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\n[5/5] Saving synchronization results..."
    )

    supplier_output_path = save_results(
        synchronization_results,
        supplier_results
    )


    print(
        f"Synchronization results saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"Supplier results saved to:\n"
        f"{supplier_output_path}"
    )


    # --------------------------------------------------------
    # Display synchronized pairs
    # --------------------------------------------------------

    synchronized_pairs = (
        synchronization_results[
            synchronization_results[
                "synchronized"
            ]
        ]
    )


    print(
        "\nSynchronized facility pairs:"
    )


    if synchronized_pairs.empty:

        print(
            "No statistically significant synchronized "
            "pairs detected."
        )

    else:

        print(
            synchronized_pairs[
                [
                    "facility_a",
                    "facility_b",
                    "supplier_a",
                    "supplier_b",
                    "shared_supplier",
                    "best_lag_days",
                    "correlation",
                    "p_value"
                ]
            ]
            .to_string(index=False)
        )


    print(
        "\nSynchronization detection complete."
    )


    return (
        data,
        synchronization_results,
        supplier_results
    )


# ============================================================
# 12. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_synchronization_pipeline()