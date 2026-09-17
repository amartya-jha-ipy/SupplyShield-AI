import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/S2_Dhis2Data.csv")
OUTPUT_DIR = Path("data/processed")

# Minimum data requirements for a prototype medicine
MIN_RECORDS = 5000
MIN_FACILITIES = 50
MIN_DATES = 20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("SUPPLYSHIELD AI - PROTOTYPE MEDICINE ANALYSIS")
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

print(f"\nDataset loaded: {len(df):,} rows")


# ============================================================
# CREATE MONTH INDEX
# ============================================================

df = df.sort_values(
    ["productID", "hf_pk", "date"]
).copy()

df["month_index"] = (
    df["date"].dt.year * 12
    + df["date"].dt.month
)

print(
    f"Date range: "
    f"{df['date'].min().date()} "
    f"to "
    f"{df['date'].max().date()}"
)


# ============================================================
# BASIC PRODUCT PROFILE
# ============================================================

print("\n" + "=" * 75)
print("1. CANDIDATE MEDICINES")
print("=" * 75)

product_profile = (
    df.groupby("productID")
    .agg(
        records=("productID", "size"),
        facilities=("hf_pk", "nunique"),
        dates=("date", "nunique"),
        stockout_records=("stockout", "sum"),
        stockout_rate=("stockout", "mean"),
        avg_consumption=("consumption", "mean"),
        avg_inventory=("closeBalance", "mean"),
        median_inventory=("closeBalance", "median"),
        zero_inventory_rate=(
            "closeBalance",
            lambda x: (x == 0).mean()
        ),
    )
    .reset_index()
)

names = (
    df.groupby("productID")["name1"]
    .first()
    .reset_index()
)

product_profile = product_profile.merge(
    names,
    on="productID",
    how="left"
)

product_profile["stockout_rate"] *= 100
product_profile["zero_inventory_rate"] *= 100

candidate_products = product_profile[
    (product_profile["records"] >= MIN_RECORDS)
    & (product_profile["facilities"] >= MIN_FACILITIES)
    & (product_profile["dates"] >= MIN_DATES)
].copy()

candidate_products = candidate_products.sort_values(
    "stockout_rate",
    ascending=False
)

print(
    candidate_products[
        [
            "productID",
            "name1",
            "records",
            "facilities",
            "dates",
            "stockout_rate",
            "zero_inventory_rate",
            "avg_consumption",
            "median_inventory",
        ]
    ].to_string(
        index=False,
        formatters={
            "stockout_rate": "{:.2f}%".format,
            "zero_inventory_rate": "{:.2f}%".format,
            "avg_consumption": "{:.2f}".format,
            "median_inventory": "{:.2f}".format,
        }
    )
)


# ============================================================
# MONTHLY BEHAVIOR
# ============================================================

print("\n" + "=" * 75)
print("2. MONTHLY STOCKOUT BEHAVIOR")
print("=" * 75)

monthly_product = (
    df.groupby(["productID", "date"])
    .agg(
        facilities=("hf_pk", "nunique"),
        stockout_rate=("stockout", "mean"),
        avg_inventory=("closeBalance", "mean"),
        avg_consumption=("consumption", "mean"),
        total_received=("received", "sum"),
    )
    .reset_index()
)

monthly_product["stockout_rate"] *= 100


# ============================================================
# STOCKOUT VARIABILITY
# ============================================================

print("\n" + "=" * 75)
print("3. STOCKOUT VARIABILITY")
print("=" * 75)

stockout_variability = (
    monthly_product[
        monthly_product["productID"].isin(
            candidate_products["productID"]
        )
    ]
    .groupby("productID")
    .agg(
        mean_stockout_rate=("stockout_rate", "mean"),
        max_stockout_rate=("stockout_rate", "max"),
        min_stockout_rate=("stockout_rate", "min"),
        stockout_rate_std=("stockout_rate", "std"),
        months_observed=("date", "nunique"),
    )
    .reset_index()
)

stockout_variability = stockout_variability.merge(
    names,
    on="productID",
    how="left"
)

stockout_variability = stockout_variability.sort_values(
    "stockout_rate_std",
    ascending=False
)

print(
    stockout_variability.to_string(
        index=False,
        formatters={
            "mean_stockout_rate": "{:.2f}%".format,
            "max_stockout_rate": "{:.2f}%".format,
            "min_stockout_rate": "{:.2f}%".format,
            "stockout_rate_std": "{:.2f}".format,
        }
    )
)


# ============================================================
# TEMPORAL TREND
# ============================================================

print("\n" + "=" * 75)
print("4. STOCKOUT TREND")
print("=" * 75)

trend_results = []

for product_id in candidate_products["productID"]:

    temp = monthly_product[
        monthly_product["productID"] == product_id
    ].sort_values("date")

    if len(temp) < 10:
        continue

    x = np.arange(len(temp))
    y = temp["stockout_rate"].values

    slope = np.polyfit(x, y, 1)[0]

    trend_results.append(
        {
            "productID": product_id,
            "name1": names.loc[
                names["productID"] == product_id,
                "name1"
            ].iloc[0],
            "trend_slope_per_month": slope,
            "first_month_stockout_rate": y[0],
            "last_month_stockout_rate": y[-1],
            "max_stockout_rate": y.max(),
        }
    )

trend_results = pd.DataFrame(trend_results)

trend_results = trend_results.sort_values(
    "trend_slope_per_month",
    ascending=False
)

print(
    trend_results.to_string(
        index=False,
        formatters={
            "trend_slope_per_month": "{:.4f}".format,
            "first_month_stockout_rate": "{:.2f}%".format,
            "last_month_stockout_rate": "{:.2f}%".format,
            "max_stockout_rate": "{:.2f}%".format,
        }
    )
)


# ============================================================
# FACILITY-LEVEL STOCKOUT CONCENTRATION
# ============================================================

print("\n" + "=" * 75)
print("5. FACILITY-LEVEL STOCKOUT CONCENTRATION")
print("=" * 75)

facility_product = (
    df.groupby(["productID", "hf_pk"])
    .agg(
        observations=("productID", "size"),
        stockout_rate=("stockout", "mean"),
        avg_inventory=("closeBalance", "mean"),
        avg_consumption=("consumption", "mean"),
    )
    .reset_index()
)

facility_product["stockout_rate"] *= 100

facility_concentration = []

for product_id in candidate_products["productID"]:

    temp = facility_product[
        facility_product["productID"] == product_id
    ]

    # Only facilities with at least 3 observations
    temp = temp[temp["observations"] >= 3]

    if len(temp) == 0:
        continue

    facility_concentration.append(
        {
            "productID": product_id,
            "name1": names.loc[
                names["productID"] == product_id,
                "name1"
            ].iloc[0],
            "facilities": len(temp),
            "mean_facility_stockout_rate":
                temp["stockout_rate"].mean(),
            "median_facility_stockout_rate":
                temp["stockout_rate"].median(),
            "facilities_with_stockout":
                (temp["stockout_rate"] > 0).sum(),
            "facilities_high_stockout":
                (temp["stockout_rate"] >= 20).sum(),
        }
    )

facility_concentration = pd.DataFrame(
    facility_concentration
)

facility_concentration = facility_concentration.sort_values(
    "facilities_high_stockout",
    ascending=False
)

print(
    facility_concentration.to_string(
        index=False,
        formatters={
            "mean_facility_stockout_rate": "{:.2f}%".format,
            "median_facility_stockout_rate": "{:.2f}%".format,
        }
    )
)


# ============================================================
# REGIONAL / DISTRICT BEHAVIOR
# ============================================================

print("\n" + "=" * 75)
print("6. DISTRICT-LEVEL BEHAVIOR")
print("=" * 75)

district_product = (
    df.groupby(["productID", "district"])
    .agg(
        records=("productID", "size"),
        facilities=("hf_pk", "nunique"),
        stockout_rate=("stockout", "mean"),
        avg_inventory=("closeBalance", "mean"),
    )
    .reset_index()
)

district_product["stockout_rate"] *= 100

district_summary = []

for product_id in candidate_products["productID"]:

    temp = district_product[
        district_product["productID"] == product_id
    ]

    if len(temp) == 0:
        continue

    district_summary.append(
        {
            "productID": product_id,
            "name1": names.loc[
                names["productID"] == product_id,
                "name1"
            ].iloc[0],
            "districts": len(temp),
            "highest_district_stockout":
                temp["stockout_rate"].max(),
            "lowest_district_stockout":
                temp["stockout_rate"].min(),
            "district_stockout_std":
                temp["stockout_rate"].std(),
            "districts_above_20pct":
                (temp["stockout_rate"] >= 20).sum(),
        }
    )

district_summary = pd.DataFrame(
    district_summary
)

district_summary = district_summary.sort_values(
    "district_stockout_std",
    ascending=False
)

print(
    district_summary.to_string(
        index=False,
        formatters={
            "highest_district_stockout": "{:.2f}%".format,
            "lowest_district_stockout": "{:.2f}%".format,
            "district_stockout_std": "{:.2f}".format,
        }
    )
)


# ============================================================
# PROTOTYPE CANDIDATE TABLE
# ============================================================

print("\n" + "=" * 75)
print("7. PROTOTYPE CANDIDATE SUMMARY")
print("=" * 75)

summary = (
    candidate_products[
        [
            "productID",
            "name1",
            "records",
            "facilities",
            "dates",
            "stockout_rate",
            "zero_inventory_rate",
            "avg_consumption",
            "median_inventory",
        ]
    ]
    .merge(
        stockout_variability[
            [
                "productID",
                "stockout_rate_std",
                "max_stockout_rate",
            ]
        ],
        on="productID",
        how="left"
    )
    .merge(
        facility_concentration[
            [
                "productID",
                "facilities_with_stockout",
                "facilities_high_stockout",
            ]
        ],
        on="productID",
        how="left"
    )
    .merge(
        district_summary[
            [
                "productID",
                "district_stockout_std",
                "districts_above_20pct",
            ]
        ],
        on="productID",
        how="left"
    )
)

summary = summary.sort_values(
    "stockout_rate",
    ascending=False
)

print(
    summary.to_string(
        index=False,
        formatters={
            "stockout_rate": "{:.2f}%".format,
            "zero_inventory_rate": "{:.2f}%".format,
            "avg_consumption": "{:.2f}".format,
            "median_inventory": "{:.2f}".format,
            "stockout_rate_std": "{:.2f}".format,
            "max_stockout_rate": "{:.2f}%".format,
            "district_stockout_std": "{:.2f}".format,
        }
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

candidate_products.to_csv(
    OUTPUT_DIR / "candidate_medicines.csv",
    index=False
)

monthly_product.to_csv(
    OUTPUT_DIR / "monthly_product_behavior.csv",
    index=False
)

stockout_variability.to_csv(
    OUTPUT_DIR / "product_stockout_variability.csv",
    index=False
)

facility_concentration.to_csv(
    OUTPUT_DIR / "product_facility_concentration.csv",
    index=False
)

district_summary.to_csv(
    OUTPUT_DIR / "product_district_summary.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "prototype_medicine_summary.csv",
    index=False
)


# ============================================================
# END
# ============================================================

print("\n" + "=" * 75)
print("STAGE 3 COMPLETE")
print("=" * 75)

print("\nSaved files:")
print("  data/processed/candidate_medicines.csv")
print("  data/processed/monthly_product_behavior.csv")
print("  data/processed/product_stockout_variability.csv")
print("  data/processed/product_facility_concentration.csv")
print("  data/processed/product_district_summary.csv")
print("  data/processed/prototype_medicine_summary.csv")

print("\nIMPORTANT:")
print("No medicine has been automatically selected.")
print("We will inspect these results before making the next step.")