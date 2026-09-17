import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/S2_Dhis2Data.csv")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SUPPLYSHIELD AI - REAL DATA PROFILING")
print("=" * 70)

if not DATA_PATH.exists():
    print(f"\nERROR: Dataset not found at:")
    print(DATA_PATH.resolve())
    print("\nMake sure S2_Dhis2Data.csv is inside:")
    print("data/raw/")
    raise SystemExit(1)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. COLUMNS")
print("=" * 70)

for i, col in enumerate(df.columns, start=1):
    print(f"{i:2}. {col}")


# ============================================================
# DATE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("2. DATE COVERAGE")
print("=" * 70)

df["date"] = pd.to_datetime(
    df["date"],
    format="%m/%d/%y",
    errors="coerce"
)

print(f"Invalid dates: {df['date'].isna().sum():,}")
print(f"Minimum date: {df['date'].min()}")
print(f"Maximum date: {df['date'].max()}")
print(f"Unique dates: {df['date'].nunique():,}")

print("\nRecords by date:")

date_counts = (
    df.groupby("date")
    .size()
    .reset_index(name="records")
    .sort_values("date")
)

print(date_counts.to_string(index=False))


# ============================================================
# FACILITY INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("3. FACILITY PROFILE")
print("=" * 70)

print(f"Unique facilities: {df['hf_pk'].nunique():,}")

print("\nFacility types:")

facility_types = (
    df.groupby("facility_type")
    .size()
    .reset_index(name="records")
    .sort_values("records", ascending=False)
)

print(facility_types.to_string(index=False))

print("\nFacilities per district:")

district_facilities = (
    df.groupby("district")["hf_pk"]
    .nunique()
    .reset_index(name="facilities")
    .sort_values("facilities", ascending=False)
)

print(district_facilities.to_string(index=False))


# ============================================================
# MEDICINE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("4. MEDICINE PROFILE")
print("=" * 70)

product_info = (
    df.groupby("productID")
    .agg(
        records=("productID", "size"),
        facilities=("hf_pk", "nunique"),
        dates=("date", "nunique"),
        stockout_rate=("stockout", "mean"),
        avg_consumption=("consumption", "mean"),
        avg_close_balance=("closeBalance", "mean"),
    )
    .reset_index()
)

product_names = (
    df.groupby("productID")["name1"]
    .first()
    .reset_index()
)

product_info = product_info.merge(
    product_names,
    on="productID",
    how="left"
)

product_info["stockout_rate"] *= 100

product_info = product_info[
    [
        "productID",
        "name1",
        "records",
        "facilities",
        "dates",
        "stockout_rate",
        "avg_consumption",
        "avg_close_balance",
    ]
]

product_info = product_info.sort_values(
    "records",
    ascending=False
)

print(f"Unique medicines: {df['productID'].nunique():,}")

print("\nMedicine coverage:")

print(
    product_info.to_string(
        index=False,
        formatters={
            "stockout_rate": "{:.2f}%".format,
            "avg_consumption": "{:.2f}".format,
            "avg_close_balance": "{:.2f}".format,
        }
    )
)


# ============================================================
# STOCKOUT PROFILE
# ============================================================

print("\n" + "=" * 70)
print("5. STOCKOUT PROFILE")
print("=" * 70)

total_rows = len(df)

stockout_rows = int(df["stockout"].sum())

print(f"Total records: {total_rows:,}")
print(f"Stockout records: {stockout_rows:,}")
print(
    f"Overall stockout rate: "
    f"{stockout_rows / total_rows * 100:.2f}%"
)

print("\nStockout records by medicine:")

stockout_by_product = (
    df.groupby(["productID", "name1"])["stockout"]
    .agg(
        stockout_records="sum",
        total_records="size"
    )
    .reset_index()
)

stockout_by_product["stockout_rate"] = (
    stockout_by_product["stockout_records"]
    / stockout_by_product["total_records"]
    * 100
)

stockout_by_product = stockout_by_product.sort_values(
    "stockout_rate",
    ascending=False
)

print(
    stockout_by_product.head(15).to_string(
        index=False,
        formatters={
            "stockout_rate": "{:.2f}%".format
        }
    )
)


# ============================================================
# CONSUMPTION PROFILE
# ============================================================

print("\n" + "=" * 70)
print("6. CONSUMPTION PROFILE")
print("=" * 70)

print(
    f"Mean consumption: "
    f"{df['consumption'].mean():.2f}"
)

print(
    f"Median consumption: "
    f"{df['consumption'].median():.2f}"
)

print(
    f"Maximum consumption: "
    f"{df['consumption'].max():.2f}"
)

print(
    f"Zero-consumption records: "
    f"{(df['consumption'] == 0).sum():,}"
)

print(
    f"Zero-consumption percentage: "
    f"{(df['consumption'] == 0).mean() * 100:.2f}%"
)


# ============================================================
# INVENTORY PROFILE
# ============================================================

print("\n" + "=" * 70)
print("7. INVENTORY PROFILE")
print("=" * 70)

print(
    f"Mean closing balance: "
    f"{df['closeBalance'].mean():.2f}"
)

print(
    f"Median closing balance: "
    f"{df['closeBalance'].median():.2f}"
)

print(
    f"Maximum closing balance: "
    f"{df['closeBalance'].max():.2f}"
)

print(
    f"Zero-inventory records: "
    f"{(df['closeBalance'] == 0).sum():,}"
)

print(
    f"Zero-inventory percentage: "
    f"{(df['closeBalance'] == 0).mean() * 100:.2f}%"
)


# ============================================================
# RECEIPT PROFILE
# ============================================================

print("\n" + "=" * 70)
print("8. REPLENISHMENT / RECEIPT PROFILE")
print("=" * 70)

print(
    f"Mean received quantity: "
    f"{df['received'].mean():.2f}"
)

print(
    f"Records with zero receipts: "
    f"{(df['received'] == 0).sum():,}"
)

print(
    f"Zero-receipt percentage: "
    f"{(df['received'] == 0).mean() * 100:.2f}%"
)


# ============================================================
# FACILITY-MEDICINE COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("9. FACILITY-MEDICINE COVERAGE")
print("=" * 70)

pair_counts = (
    df.groupby(["hf_pk", "productID"])
    .size()
)

print(f"Facility-medicine pairs: {len(pair_counts):,}")

print("\nObservations per facility-medicine pair:")

print(pair_counts.describe().to_string())

print("\nCoverage buckets:")

coverage_buckets = {
    "1-4 observations": (1, 4),
    "5-9 observations": (5, 9),
    "10-19 observations": (10, 19),
    "20-29 observations": (20, 29),
    "30-39 observations": (30, 39),
    "40+ observations": (40, float("inf")),
}

for label, (low, high) in coverage_buckets.items():

    if high == float("inf"):
        count = (pair_counts >= low).sum()
    else:
        count = ((pair_counts >= low) & (pair_counts <= high)).sum()

    percentage = count / len(pair_counts) * 100

    print(
        f"{label:<25}: "
        f"{count:>7,} pairs "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# MEDICINES WITH STRONG DATA COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("10. MEDICINES WITH STRONG DATA COVERAGE")
print("=" * 70)

strong_products = product_info[
    (product_info["records"] >= 1000)
    & (product_info["facilities"] >= 20)
    & (product_info["dates"] >= 20)
].copy()

print(
    f"Medicines satisfying:"
    f"\n  records >= 1000"
    f"\n  facilities >= 20"
    f"\n  dates >= 20"
)

print(
    f"\nNumber of candidate medicines: "
    f"{len(strong_products)}"
)

print(
    strong_products.to_string(
        index=False,
        formatters={
            "stockout_rate": "{:.2f}%".format,
            "avg_consumption": "{:.2f}".format,
            "avg_close_balance": "{:.2f}".format,
        }
    )
)


# ============================================================
# GEOGRAPHIC COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("11. GEOGRAPHIC COVERAGE")
print("=" * 70)

print(
    f"Facilities with latitude: "
    f"{df['lat'].notna().sum():,}"
)

print(
    f"Facilities with longitude: "
    f"{df['long'].notna().sum():,}"
)

facility_locations = (
    df[
        ["hf_pk", "name1", "lat", "long", "district"]
    ]
    .drop_duplicates("hf_pk")
)

print(
    f"Unique facility locations: "
    f"{len(facility_locations):,}"
)

print("\nFacilities by district:")

print(
    facility_locations
    .groupby("district")
    .size()
    .sort_values(ascending=False)
    .to_string()
)


# ============================================================
# STOCKOUT BY DISTRICT
# ============================================================

print("\n" + "=" * 70)
print("12. STOCKOUT BY DISTRICT")
print("=" * 70)

district_stockout = (
    df.groupby("district")
    .agg(
        records=("stockout", "size"),
        stockouts=("stockout", "sum"),
        facilities=("hf_pk", "nunique"),
    )
    .reset_index()
)

district_stockout["stockout_rate"] = (
    district_stockout["stockouts"]
    / district_stockout["records"]
    * 100
)

district_stockout = district_stockout.sort_values(
    "stockout_rate",
    ascending=False
)

print(
    district_stockout.to_string(
        index=False,
        formatters={
            "stockout_rate": "{:.2f}%".format
        }
    )
)


# ============================================================
# DATA CONTINUITY
# ============================================================

print("\n" + "=" * 70)
print("13. DATA CONTINUITY")
print("=" * 70)

# Sort chronologically
df_sorted = df.sort_values(
    ["hf_pk", "productID", "date"]
).copy()

df_sorted["date_diff"] = (
    df_sorted
    .groupby(["hf_pk", "productID"])["date"]
    .diff()
    .dt.days
)

continuity = (
    df_sorted
    .groupby(["hf_pk", "productID"])
    .agg(
        observations=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        median_gap_days=("date_diff", "median"),
        max_gap_days=("date_diff", "max"),
    )
    .reset_index()
)

print(
    "\nContinuity statistics:"
)

print(
    continuity[
        [
            "observations",
            "median_gap_days",
            "max_gap_days"
        ]
    ].describe().to_string()
)

continuous_pairs = continuity[
    (continuity["observations"] >= 20)
    & (continuity["median_gap_days"] <= 2)
].copy()

print(
    f"\nFacility-medicine pairs with:"
    f"\n  >= 20 observations"
    f"\n  median gap <= 2 days"
)

print(
    f"Candidate continuous pairs: "
    f"{len(continuous_pairs):,}"
)


# ============================================================
# ACCOUNTING CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("14. INVENTORY ACCOUNTING CHECK")
print("=" * 70)

expected_close = (
    df["openBalance"]
    + df["received"]
    - df["consumption"]
)

mismatch = (
    expected_close
    != df["closeBalance"]
)

mismatch_count = int(mismatch.sum())

print(
    f"Rows where "
    f"openBalance + received - consumption "
    f"!= closeBalance:"
)

print(f"{mismatch_count:,} / {len(df):,}")

print(
    f"Mismatch percentage: "
    f"{mismatch.mean() * 100:.2f}%"
)


# ============================================================
# FINAL CANDIDATE MEDICINES
# ============================================================

print("\n" + "=" * 70)
print("15. INITIAL CANDIDATE MEDICINES")
print("=" * 70)

candidate_products = product_info[
    (product_info["records"] >= 5000)
    & (product_info["facilities"] >= 50)
    & (product_info["dates"] >= 20)
    & (product_info["stockout_rate"] > 1)
].copy()

candidate_products = candidate_products.sort_values(
    ["stockout_rate", "records"],
    ascending=[False, False]
)

print(
    "These are NOT final selections."
)
print(
    "They are simply candidates for the next ML stage."
)

print(
    candidate_products.to_string(
        index=False,
        formatters={
            "stockout_rate": "{:.2f}%".format,
            "avg_consumption": "{:.2f}".format,
            "avg_close_balance": "{:.2f}".format,
        }
    )
)


# ============================================================
# SAVE PROFILE OUTPUTS
# ============================================================

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

product_info.to_csv(
    OUTPUT_DIR / "product_profile.csv",
    index=False
)

district_stockout.to_csv(
    OUTPUT_DIR / "district_stockout_profile.csv",
    index=False
)

continuity.to_csv(
    OUTPUT_DIR / "facility_product_continuity.csv",
    index=False
)

print("\n" + "=" * 70)
print("PROFILING COMPLETE")
print("=" * 70)

print("\nSaved:")
print("  data/processed/product_profile.csv")
print("  data/processed/district_stockout_profile.csv")
print("  data/processed/facility_product_continuity.csv")

print("\nDo NOT build the next ML model yet.")
print("First inspect the results above.")