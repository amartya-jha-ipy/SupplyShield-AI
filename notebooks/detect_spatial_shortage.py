import pandas as pd
import numpy as np

from pathlib import Path
from scipy.spatial.distance import cdist


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/processed/prototype_paracetamol_500mg.csv"
)

OUTPUT_DIR = Path("data/processed")

OUTPUT_PATH = (
    OUTPUT_DIR / "spatial_shortage_signals.csv"
)

EVENTS_PATH = (
    OUTPUT_DIR / "regional_shortage_events.csv"
)

# Maximum geographic distance between facilities.
# We use 25 km as the regional-neighborhood radius.

DISTANCE_THRESHOLD_KM = 25.0

# Minimum number of nearby facilities required
# to consider a regional signal.

MIN_NEIGHBOURS = 2

# A facility-month is considered locally elevated
# when its stockout status is 1.
#
# We then look for clusters where several nearby
# facilities have elevated stockout rates.

MIN_CLUSTER_STOCKOUT_RATE = 0.50


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance_matrix(
    latitudes,
    longitudes
):
    """
    Calculate pairwise great-circle distances
    between facilities.

    Returns:
        distance matrix in kilometres.
    """

    lat = np.radians(
        np.asarray(latitudes)
    )

    lon = np.radians(
        np.asarray(longitudes)
    )

    lat1 = lat[:, None]
    lat2 = lat[None, :]

    lon1 = lon[:, None]
    lon2 = lon[None, :]

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    a = np.clip(
        a,
        0,
        1
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    earth_radius_km = 6371.0

    return earth_radius_km * c


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("SUPPLYSHIELD AI - SPATIAL SHORTAGE DETECTOR")
print("=" * 75)

if not DATA_PATH.exists():

    print(
        "\nERROR: Prepared dataset not found:"
    )

    print(
        DATA_PATH.resolve()
    )

    raise SystemExit(1)


df = pd.read_csv(
    DATA_PATH
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

print(
    f"\nLoaded records: {len(df):,}"
)

print(
    f"Facilities: "
    f"{df['facility_id'].nunique():,}"
)

print(
    f"Dates: "
    f"{df['date'].nunique():,}"
)


# ============================================================
# FACILITY MASTER TABLE
# ============================================================

print("\n" + "=" * 75)
print("1. BUILDING FACILITY LOCATION TABLE")
print("=" * 75)

facility_columns = [
    "facility_id",
    "facility_type",
    "latitude",
    "longitude",
    "district",
]

facilities = (
    df[facility_columns]
    .drop_duplicates(
        subset=["facility_id"]
    )
    .reset_index(drop=True)
)

print(
    f"Unique facilities: "
    f"{len(facilities):,}"
)


# ============================================================
# CHECK COORDINATES
# ============================================================

invalid_coordinates = (
    facilities["latitude"].isna()
    | facilities["longitude"].isna()
)

if invalid_coordinates.any():

    print(
        "\nRemoving facilities with missing "
        "coordinates:"
        f" {invalid_coordinates.sum():,}"
    )

    facilities = facilities[
        ~invalid_coordinates
    ].copy()

    facilities = facilities.reset_index(
        drop=True
    )


# ============================================================
# BUILD SPATIAL DISTANCE MATRIX
# ============================================================

print("\n" + "=" * 75)
print("2. CALCULATING FACILITY DISTANCES")
print("=" * 75)

print(
    "\nCalculating pairwise geographic distances..."
)

distance_matrix = haversine_distance_matrix(
    facilities["latitude"].values,
    facilities["longitude"].values,
)

print(
    "Distance matrix created:"
)

print(
    f"  Shape: {distance_matrix.shape}"
)


# ============================================================
# BUILD NEIGHBOUR NETWORK
# ============================================================

print("\n" + "=" * 75)
print("3. BUILDING 25 KM FACILITY NETWORK")
print("=" * 75)

# adjacency[i, j] = True when facility j is
# within 25 km of facility i.

adjacency = (
    distance_matrix <= DISTANCE_THRESHOLD_KM
)

# Remove self-connections.

np.fill_diagonal(
    adjacency,
    False
)

neighbour_counts = adjacency.sum(
    axis=1
)

facilities["nearby_facilities"] = (
    neighbour_counts
)

print(
    f"Distance threshold: "
    f"{DISTANCE_THRESHOLD_KM:.1f} km"
)

print(
    f"Facilities with at least "
    f"{MIN_NEIGHBOURS} neighbours: "
    f"{(neighbour_counts >= MIN_NEIGHBOURS).sum():,}"
)

print(
    "\nNeighbour statistics:"
)

print(
    pd.Series(
        neighbour_counts
    ).describe().to_string()
)


# ============================================================
# MONTHLY FACILITY STOCKOUT DATA
# ============================================================

print("\n" + "=" * 75)
print("4. CALCULATING MONTHLY STOCKOUT SIGNAL")
print("=" * 75)

# The dataset already has a binary stockout field.
#
# Because there can be at most one row per
# facility-date-medicine combination, the monthly
# facility stockout signal is simply the stockout value.

monthly = df[
    [
        "facility_id",
        "date",
        "stockout",
        "closing_inventory",
        "consumption",
    ]
].copy()

monthly["stockout"] = (
    monthly["stockout"]
    .astype(int)
)

print(
    f"Facility-month observations: "
    f"{len(monthly):,}"
)


# ============================================================
# MAP FACILITIES TO INDICES
# ============================================================

facility_to_index = {
    facility_id: index
    for index, facility_id
    in enumerate(
        facilities["facility_id"]
    )
}

monthly["facility_index"] = (
    monthly["facility_id"]
    .map(facility_to_index)
)

monthly = monthly.dropna(
    subset=["facility_index"]
).copy()

monthly["facility_index"] = (
    monthly["facility_index"]
    .astype(int)
)


# ============================================================
# CALCULATE LOCAL SPATIAL SIGNAL
# ============================================================

print("\n" + "=" * 75)
print("5. CALCULATING LOCAL SHORTAGE PRESSURE")
print("=" * 75)

results = []

dates = sorted(
    monthly["date"].unique()
)

print(
    f"Months to analyse: "
    f"{len(dates)}"
)

for date in dates:

    month_data = monthly[
        monthly["date"] == date
    ].copy()

    stockout_array = np.zeros(
        len(facilities),
        dtype=float
    )

    inventory_array = np.full(
        len(facilities),
        np.nan
    )

    consumption_array = np.full(
        len(facilities),
        np.nan
    )

    observed_array = np.zeros(
        len(facilities),
        dtype=bool
    )

    for row in month_data.itertuples():

        index = row.facility_index

        stockout_array[index] = (
            row.stockout
        )

        inventory_array[index] = (
            row.closing_inventory
        )

        consumption_array[index] = (
            row.consumption
        )

        observed_array[index] = True


    # --------------------------------------------------------
    # For every facility calculate:
    #
    # - number of nearby observed facilities
    # - number of nearby stockout facilities
    # - nearby stockout rate
    # --------------------------------------------------------

    for i in range(
        len(facilities)
    ):

        if not observed_array[i]:
            continue

        neighbours = np.where(
            adjacency[i]
            & observed_array
        )[0]

        neighbour_count = len(
            neighbours
        )

        if neighbour_count == 0:

            nearby_stockouts = 0
            nearby_stockout_rate = np.nan

        else:

            nearby_stockouts = int(
                stockout_array[
                    neighbours
                ].sum()
            )

            nearby_stockout_rate = (
                nearby_stockouts
                / neighbour_count
            )


        current_stockout = int(
            stockout_array[i]
        )


        # ----------------------------------------------------
        # Local shortage pressure
        # ----------------------------------------------------

        if np.isnan(
            nearby_stockout_rate
        ):

            local_pressure = np.nan

        else:

            local_pressure = (
                nearby_stockout_rate
                * (
                    neighbour_count
                    /
                    (
                        neighbour_count
                        + MIN_NEIGHBOURS
                    )
                )
            )


        # ----------------------------------------------------
        # Regional signal
        #
        # We require at least MIN_NEIGHBOURS
        # nearby facilities and a high local
        # stockout rate.
        # ----------------------------------------------------

        regional_signal = int(
            (
                neighbour_count
                >= MIN_NEIGHBOURS
            )
            and
            (
                nearby_stockout_rate
                >= MIN_CLUSTER_STOCKOUT_RATE
            )
        )


        results.append(
            {
                "date": date,
                "facility_id":
                    facilities.iloc[i][
                        "facility_id"
                    ],
                "district":
                    facilities.iloc[i][
                        "district"
                    ],
                "latitude":
                    facilities.iloc[i][
                        "latitude"
                    ],
                "longitude":
                    facilities.iloc[i][
                        "longitude"
                    ],
                "current_stockout":
                    current_stockout,
                "nearby_facilities":
                    neighbour_count,
                "nearby_stockouts":
                    nearby_stockouts,
                "nearby_stockout_rate":
                    nearby_stockout_rate,
                "local_shortage_pressure":
                    local_pressure,
                "regional_signal":
                    regional_signal,
                "closing_inventory":
                    inventory_array[i],
                "consumption":
                    consumption_array[i],
            }
        )


signals = pd.DataFrame(
    results
)


# ============================================================
# ADD MONTHLY REGIONAL STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("6. CALCULATING REGIONAL SUMMARY SIGNALS")
print("=" * 75)

monthly_summary = (
    signals
    .groupby("date")
    .agg(
        observed_facilities=(
            "facility_id",
            "nunique",
        ),
        stockout_facilities=(
            "current_stockout",
            "sum",
        ),
        average_local_pressure=(
            "local_shortage_pressure",
            "mean",
        ),
        regional_signal_facilities=(
            "regional_signal",
            "sum",
        ),
    )
    .reset_index()
)

monthly_summary[
    "overall_stockout_rate"
] = (
    monthly_summary[
        "stockout_facilities"
    ]
    /
    monthly_summary[
        "observed_facilities"
    ]
)

monthly_summary[
    "regional_signal_fraction"
] = (
    monthly_summary[
        "regional_signal_facilities"
    ]
    /
    monthly_summary[
        "observed_facilities"
    ]
)


# ============================================================
# REGIONAL EVENT DETECTION
# ============================================================

print("\n" + "=" * 75)
print("7. DETECTING REGIONAL SHORTAGE EVENTS")
print("=" * 75)

# A month becomes a regional event when multiple
# facility-level regional signals are simultaneously
# present.

monthly_summary[
    "regional_event"
] = (
    monthly_summary[
        "regional_signal_facilities"
    ]
    >= MIN_NEIGHBOURS
).astype(int)


# ============================================================
# ADD SUMMARY BACK TO FACILITY SIGNALS
# ============================================================

signals = signals.merge(
    monthly_summary[
        [
            "date",
            "overall_stockout_rate",
            "regional_signal_facilities",
            "regional_signal_fraction",
            "regional_event",
        ]
    ],
    on="date",
    how="left",
)


# ============================================================
# SAVE FACILITY-LEVEL SIGNALS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

signals.to_csv(
    OUTPUT_PATH,
    index=False
)

monthly_summary.to_csv(
    EVENTS_PATH,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 75)
print("8. RESULTS")
print("=" * 75)

print(
    f"\nFacility-month spatial signals: "
    f"{len(signals):,}"
)

print(
    "\nFacility-months with a regional signal:"
)

print(
    int(
        signals[
            "regional_signal"
        ].sum()
    )
)

print(
    "\nMonths with a regional shortage event:"
)

print(
    int(
        monthly_summary[
            "regional_event"
        ].sum()
    )
)

print(
    "\nMonths analysed:"
)

print(
    monthly_summary[
        [
            "date",
            "observed_facilities",
            "overall_stockout_rate",
            "regional_signal_facilities",
            "regional_event",
        ]
    ].to_string(
        index=False,
        formatters={
            "overall_stockout_rate":
                "{:.2%}".format,
        }
    )
)


# ============================================================
# TOP REGIONAL SIGNALS
# ============================================================

print("\n" + "=" * 75)
print("9. STRONGEST REGIONAL SIGNALS")
print("=" * 75)

top_signals = (
    signals[
        signals["regional_signal"] == 1
    ]
    .sort_values(
        [
            "nearby_stockout_rate",
            "nearby_facilities",
        ],
        ascending=False
    )
    .head(20)
)

if top_signals.empty:

    print(
        "\nNo facility-level regional signals "
        "were detected using the current thresholds."
    )

else:

    print(
        top_signals[
            [
                "date",
                "facility_id",
                "district",
                "nearby_facilities",
                "nearby_stockouts",
                "nearby_stockout_rate",
                "local_shortage_pressure",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "nearby_stockout_rate":
                    "{:.2%}".format,
                "local_shortage_pressure":
                    "{:.3f}".format,
            }
        )
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STAGE 6 COMPLETE")
print("=" * 75)

print(
    "\nSpatial method:"
)

print(
    "  Haversine geographic distance"
)

print(
    f"  {DISTANCE_THRESHOLD_KM:.0f} km neighbourhood radius"
)

print(
    "\nRegional signal:"
)

print(
    f"  At least {MIN_NEIGHBOURS} nearby "
    "facilities"
)

print(
    f"  Nearby stockout rate >= "
    f"{MIN_CLUSTER_STOCKOUT_RATE:.0%}"
)

print(
    "\nSaved files:"
)

print(
    f"  {OUTPUT_PATH}"
)

print(
    f"  {EVENTS_PATH}"
)

print(
    "\nThese outputs provide the spatial evidence "
    "that will later be combined with the ML "
    "shortage-risk predictions."
)