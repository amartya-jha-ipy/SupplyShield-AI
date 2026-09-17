import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/S2_Dhis2Data.csv")
OUTPUT_DIR = Path("data/processed")

TARGET_PRODUCT_ID = 50


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("SUPPLYSHIELD AI - PREPARE PROTOTYPE DATA")
print("=" * 75)

if not DATA_PATH.exists():
    print("\nERROR: Dataset not found:")
    print(DATA_PATH.resolve())
    raise SystemExit(1)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(
    df["date"],
    format="%m/%d/%y",
    errors="coerce"
)

print(f"\nOriginal dataset: {len(df):,} rows")


# ============================================================
# SELECT PROTOTYPE MEDICINE
# ============================================================

print("\n" + "=" * 75)
print("1. SELECTING PROTOTYPE MEDICINE")
print("=" * 75)

prototype = df[
    df["productID"] == TARGET_PRODUCT_ID
].copy()

if prototype.empty:
    print(
        f"\nERROR: productID {TARGET_PRODUCT_ID} "
        "was not found."
    )
    raise SystemExit(1)

medicine_name = prototype["name1"].iloc[0]

print(f"Product ID: {TARGET_PRODUCT_ID}")
print(f"Medicine: {medicine_name}")
print(f"Records: {len(prototype):,}")


# ============================================================
# SELECT REQUIRED COLUMNS
# ============================================================

print("\n" + "=" * 75)
print("2. SELECTING MODELING COLUMNS")
print("=" * 75)

prototype = prototype[
    [
        "hf_pk",
        "name1",
        "date",
        "stockout",
        "received",
        "consumption",
        "closeBalance",
        "openBalance",
        "facility_type",
        "lat",
        "long",
        "district",
        "productID",
    ]
].copy()

prototype = prototype.rename(
    columns={
        "hf_pk": "facility_id",
        "name1": "medicine",
        "closeBalance": "closing_inventory",
        "openBalance": "opening_inventory",
        "lat": "latitude",
        "long": "longitude",
    }
)

print("\nColumns:")
for column in prototype.columns:
    print(f"  - {column}")


# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

print("\n" + "=" * 75)
print("3. DATA QUALITY CHECK")
print("=" * 75)

before = len(prototype)

invalid_date = prototype["date"].isna()

invalid_coordinates = (
    prototype["latitude"].isna()
    | prototype["longitude"].isna()
)

invalid_numeric = (
    prototype["consumption"].isna()
    | prototype["received"].isna()
    | prototype["closing_inventory"].isna()
    | prototype["opening_inventory"].isna()
    | prototype["stockout"].isna()
)

invalid = (
    invalid_date
    | invalid_coordinates
    | invalid_numeric
)

print(f"Records before cleaning: {before:,}")
print(f"Invalid dates: {invalid_date.sum():,}")
print(f"Invalid coordinates: {invalid_coordinates.sum():,}")
print(f"Invalid numeric values: {invalid_numeric.sum():,}")
print(f"Total invalid records: {invalid.sum():,}")

prototype = prototype[~invalid].copy()

print(f"Records after cleaning: {len(prototype):,}")


# ============================================================
# CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 75)
print("4. DUPLICATE CHECK")
print("=" * 75)

duplicate_keys = [
    "facility_id",
    "date",
    "productID",
]

duplicate_mask = prototype.duplicated(
    subset=duplicate_keys,
    keep=False
)

duplicate_rows = int(duplicate_mask.sum())

print(
    "Duplicate facility-date-medicine rows: "
    f"{duplicate_rows:,}"
)

if duplicate_rows > 0:

    print(
        "\nDuplicate examples:"
    )

    print(
        prototype[
            duplicate_mask
        ]
        .sort_values(duplicate_keys)
        .head(20)
        .to_string(index=False)
    )

    print(
        "\nDuplicates will be aggregated by:"
        "\n  facility + date + medicine"
    )

    prototype = (
        prototype
        .groupby(
            [
                "facility_id",
                "medicine",
                "date",
                "productID",
                "facility_type",
                "latitude",
                "longitude",
                "district",
            ],
            as_index=False
        )
        .agg(
            stockout=("stockout", "max"),
            received=("received", "sum"),
            consumption=("consumption", "sum"),
            closing_inventory=("closing_inventory", "sum"),
            opening_inventory=("opening_inventory", "sum"),
        )
    )

print(
    f"Records after duplicate handling: "
    f"{len(prototype):,}"
)


# ============================================================
# SORT DATA
# ============================================================

prototype = prototype.sort_values(
    ["facility_id", "date"]
).reset_index(drop=True)


# ============================================================
# ADD TIME FEATURES
# ============================================================

print("\n" + "=" * 75)
print("5. CREATING TIME FEATURES")
print("=" * 75)

prototype["year"] = prototype["date"].dt.year
prototype["month"] = prototype["date"].dt.month

prototype["month_index"] = (
    prototype["date"].dt.year * 12
    + prototype["date"].dt.month
)

print("Added:")
print("  year")
print("  month")
print("  month_index")


# ============================================================
# CREATE INVENTORY-TO-DEMAND FEATURES
# ============================================================

print("\n" + "=" * 75)
print("6. CREATING INVENTORY FEATURES")
print("=" * 75)

# A simple measure of how many months of demand
# the current inventory could cover.
#
# We use a rolling 3-observation consumption average
# within each facility.
#
# This is NOT a final shortage-risk model.
# It is simply a useful explanatory feature.

prototype["rolling_consumption_3"] = (
    prototype
    .groupby("facility_id")["consumption"]
    .transform(
        lambda x: x.rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)

prototype["inventory_coverage"] = np.where(
    prototype["rolling_consumption_3"] > 0,
    prototype["closing_inventory"]
    / prototype["rolling_consumption_3"],
    np.nan
)

print(
    "Added rolling_consumption_3"
)

print(
    "Added inventory_coverage"
)


# ============================================================
# INVENTORY CHANGE
# ============================================================

prototype["inventory_change"] = (
    prototype
    .groupby("facility_id")["closing_inventory"]
    .diff()
)

prototype["consumption_change"] = (
    prototype
    .groupby("facility_id")["consumption"]
    .diff()
)

print("Added:")
print("  inventory_change")
print("  consumption_change")


# ============================================================
# VERIFY DATA RANGE
# ============================================================

print("\n" + "=" * 75)
print("7. FINAL DATA RANGE")
print("=" * 75)

print(
    f"Date range: "
    f"{prototype['date'].min().date()} "
    f"to "
    f"{prototype['date'].max().date()}"
)

print(
    f"Unique facilities: "
    f"{prototype['facility_id'].nunique():,}"
)

print(
    f"Unique dates: "
    f"{prototype['date'].nunique():,}"
)

print(
    f"Districts: "
    f"{prototype['district'].nunique():,}"
)


# ============================================================
# FINAL STOCKOUT STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("8. FINAL STOCKOUT STATISTICS")
print("=" * 75)

print(
    f"Stockout records: "
    f"{prototype['stockout'].sum():,}"
)

print(
    f"Stockout rate: "
    f"{prototype['stockout'].mean() * 100:.2f}%"
)

print(
    f"Zero-inventory records: "
    f"{(prototype['closing_inventory'] == 0).sum():,}"
)

print(
    f"Zero-inventory rate: "
    f"{(prototype['closing_inventory'] == 0).mean() * 100:.2f}%"
)


# ============================================================
# FACILITY COVERAGE
# ============================================================

print("\n" + "=" * 75)
print("9. FACILITY COVERAGE")
print("=" * 75)

facility_counts = (
    prototype
    .groupby("facility_id")
    .size()
)

print(
    "Observations per facility:"
)

print(
    facility_counts.describe().to_string()
)

print(
    f"\nFacilities with >= 10 observations: "
    f"{(facility_counts >= 10).sum():,}"
)

print(
    f"Facilities with >= 20 observations: "
    f"{(facility_counts >= 20).sum():,}"
)

print(
    f"Facilities with >= 30 observations: "
    f"{(facility_counts >= 30).sum():,}"
)


# ============================================================
# MONTHLY DATA COVERAGE
# ============================================================

print("\n" + "=" * 75)
print("10. MONTHLY DATA COVERAGE")
print("=" * 75)

monthly = (
    prototype
    .groupby("date")
    .agg(
        facilities=("facility_id", "nunique"),
        records=("facility_id", "size"),
        stockout_rate=("stockout", "mean"),
        avg_inventory=("closing_inventory", "mean"),
        avg_consumption=("consumption", "mean"),
    )
    .reset_index()
)

monthly["stockout_rate"] *= 100

print(
    monthly.to_string(
        index=False,
        formatters={
            "stockout_rate": "{:.2f}%".format,
            "avg_inventory": "{:.2f}".format,
            "avg_consumption": "{:.2f}".format,
        }
    )
)


# ============================================================
# GEOGRAPHIC RANGE
# ============================================================

print("\n" + "=" * 75)
print("11. GEOGRAPHIC RANGE")
print("=" * 75)

print(
    f"Latitude range: "
    f"{prototype['latitude'].min():.6f} "
    f"to "
    f"{prototype['latitude'].max():.6f}"
)

print(
    f"Longitude range: "
    f"{prototype['longitude'].min():.6f} "
    f"to "
    f"{prototype['longitude'].max():.6f}"
)


# ============================================================
# PREVIEW
# ============================================================

print("\n" + "=" * 75)
print("12. FINAL DATA PREVIEW")
print("=" * 75)

preview_columns = [
    "facility_id",
    "medicine",
    "date",
    "stockout",
    "received",
    "consumption",
    "opening_inventory",
    "closing_inventory",
    "latitude",
    "longitude",
    "district",
    "rolling_consumption_3",
    "inventory_coverage",
    "inventory_change",
]

print(
    prototype[
        preview_columns
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# SAVE CLEAN DATA
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    OUTPUT_DIR
    / "prototype_paracetamol_500mg.csv"
)

prototype.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 75)
print("STAGE 4 COMPLETE")
print("=" * 75)

print(
    "\nClean prototype dataset saved to:"
)

print(
    f"  {output_path}"
)

print(
    f"\nFinal rows: {len(prototype):,}"
)

print(
    "\nThis file will be the primary input "
    "for the next modeling stage."
)