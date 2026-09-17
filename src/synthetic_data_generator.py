from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = BASE_DIR / "data" / "synthetic" / "config.yaml"
OUTPUT_PATH = BASE_DIR / "data" / "synthetic" / "supply_chain_data.csv"


# ============================================================
# LOAD CONFIGURATION
# ============================================================

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# ============================================================
# RANDOM SEED
# ============================================================

SEED = config["simulation"]["random_seed"]
rng = np.random.default_rng(SEED)


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

NUM_FACILITIES = config["simulation"]["num_facilities"]
NUM_SUPPLIERS = config["simulation"]["num_suppliers"]
NUM_DAYS = config["simulation"]["num_days"]

MEDICINE_NAME = config["medicine"]["name"]

INITIAL_STOCK_MIN = config["medicine"]["initial_stock_min"]
INITIAL_STOCK_MAX = config["medicine"]["initial_stock_max"]

BASE_DEMAND_MIN = config["demand"]["base_daily_min"]
BASE_DEMAND_MAX = config["demand"]["base_daily_max"]

LEAD_TIME_MIN = config["suppliers"]["lead_time_min_days"]
LEAD_TIME_MAX = config["suppliers"]["lead_time_max_days"]

DISRUPTION_ENABLED = config["shortage"]["enable_disruption"]
DISRUPTION_START = config["shortage"]["disruption_start_day"]
DISRUPTION_DURATION = config["shortage"]["disruption_duration_days"]
DISRUPTION_REDUCTION = config["shortage"]["disruption_reduction"]

DISRUPTION_END = DISRUPTION_START + DISRUPTION_DURATION - 1


# ============================================================
# FACILITY DEFINITIONS
# ============================================================
#
# Facilities are deliberately placed into three geographic
# clusters so the spatial anomaly detector has nearby facilities
# to analyze.
#
# Cluster A -> Supplier S01
# Cluster B -> Supplier S02
# Cluster C -> Supplier S03
#
# Each cluster contains facilities within roughly 25 km.
# ============================================================

FACILITIES = {
    "F01": {
        "supplier_id": "S01",
        "latitude": 20.0000,
        "longitude": 77.0000,
        "initial_stock": 90,
        "cluster": "A",
    },
    "F04": {
        "supplier_id": "S01",
        "latitude": 20.0600,
        "longitude": 77.0500,
        "initial_stock": 85,
        "cluster": "A",
    },
    "F07": {
        "supplier_id": "S01",
        "latitude": 20.1100,
        "longitude": 77.0000,
        "initial_stock": 110,
        "cluster": "A",
    },
    "F10": {
        "supplier_id": "S01",
        "latitude": 20.0500,
        "longitude": 77.1200,
        "initial_stock": 115,
        "cluster": "A",
    },
    "F02": {
        "supplier_id": "S02",
        "latitude": 20.5000,
        "longitude": 77.5000,
        "initial_stock": 100,
        "cluster": "B",
    },
    "F05": {
        "supplier_id": "S02",
        "latitude": 20.5500,
        "longitude": 77.5500,
        "initial_stock": 75,
        "cluster": "B",
    },
    "F08": {
        "supplier_id": "S02",
        "latitude": 20.6000,
        "longitude": 77.5000,
        "initial_stock": 105,
        "cluster": "B",
    },
    "F03": {
        "supplier_id": "S03",
        "latitude": 20.9000,
        "longitude": 77.9000,
        "initial_stock": 105,
        "cluster": "C",
    },
    "F06": {
        "supplier_id": "S03",
        "latitude": 20.9500,
        "longitude": 77.9500,
        "initial_stock": 105,
        "cluster": "C",
    },
    "F09": {
        "supplier_id": "S03",
        "latitude": 21.0000,
        "longitude": 77.9000,
        "initial_stock": 105,
        "cluster": "C",
    },
}


# ============================================================
# SUPPLIER DEFINITIONS
# ============================================================

SUPPLIERS = {
    "S01": ["F01", "F04", "F07", "F10"],
    "S02": ["F02", "F05", "F08"],
    "S03": ["F03", "F06", "F09"],
}


# ============================================================
# WEEKLY DEMAND SEASONALITY
# ============================================================

WEEKLY_MULTIPLIER = {
    0: 1.00,  # Monday
    1: 1.05,  # Tuesday
    2: 1.10,  # Wednesday
    3: 1.00,  # Thursday
    4: 0.95,  # Friday
    5: 0.85,  # Saturday
    6: 0.80,  # Sunday
}


# ============================================================
# BASE DEMAND GENERATION
# ============================================================

def generate_base_demands():
    """
    Assign each facility a stable underlying daily demand.
    """

    base_demands = {}

    for facility_id in FACILITIES:
        base_demands[facility_id] = int(
            rng.integers(
                BASE_DEMAND_MIN,
                BASE_DEMAND_MAX + 1
            )
        )

    return base_demands


# ============================================================
# DEMAND GENERATION
# ============================================================

def generate_demand(facility_id, day, base_demand):
    """
    Generate daily medicine consumption.

    Normal demand:
        base demand
        × weekly seasonality
        + small random noise

    During disruption:
        Cluster A experiences a strong demand surge.
        Cluster B experiences a moderate demand surge.

    This allows the system to detect both:
        1. supply-side shortage signals
        2. regional demand anomalies
    """

    weekday = (day - 1) % 7

    seasonal_multiplier = WEEKLY_MULTIPLIER[weekday]

    demand = base_demand * seasonal_multiplier

    cluster = FACILITIES[facility_id]["cluster"]

    # Regional demand surge in Cluster A.
    if cluster == "A" and 16 <= day <= 21:
        demand *= 1.55

    # Smaller regional demand surge in Cluster B.
    elif cluster == "B" and 18 <= day <= 20:
        demand *= 1.25

    # Small random demand variation.
    noise = rng.normal(0, 0.8)

    demand = demand + noise

    return max(1, int(round(demand)))


# ============================================================
# REPLENISHMENT GENERATION
# ============================================================

def generate_replenishment(
    facility_id,
    day,
    demand,
    base_demand,
    lead_time
):
    """
    Generate incoming medicine.

    IMPORTANT:
    Replenishment is now tied to consumption and supplier lead time.

    Normal conditions:
        shipment roughly covers expected consumption during
        the supplier's lead-time cycle.

    Disruption:
        supplier capacity is reduced by 50%.

    Therefore:
        normal replenishment ≈ consumption
        disrupted replenishment < consumption

    This causes inventory to decline during disruption.
    """

    supplier_id = FACILITIES[facility_id]["supplier_id"]

    # Normal expected shipment.
    #
    # We do NOT multiply the shipment by 2-3x daily demand.
    # Instead, shipment capacity is based on expected demand
    # accumulated over the lead-time period.
    normal_quantity = (
        base_demand
        * lead_time
        * 0.95
    )

    # During disruption, supplier capacity is reduced.
    if (
        DISRUPTION_ENABLED
        and DISRUPTION_START <= day <= DISRUPTION_END
    ):
        normal_quantity *= (1 - DISRUPTION_REDUCTION)

    # Shipment noise.
    shipment_noise = rng.uniform(0.90, 1.05)

    replenishment = normal_quantity * shipment_noise

    # Occasionally a shipment arrives late.
    #
    # This introduces additional realistic variation without
    # creating excessive randomness.
    if day > 1 and rng.random() < 0.08:
        replenishment *= 0.65

    replenishment = max(0, int(round(replenishment)))

    return replenishment


# ============================================================
# MAIN SIMULATION
# ============================================================

def generate_dataset():

    base_demands = generate_base_demands()

    # Inventory state for each facility.
    inventory = {
        facility_id: FACILITIES[facility_id]["initial_stock"]
        for facility_id in FACILITIES
    }

    # Each facility receives a fixed supplier lead time.
    lead_times = {
        facility_id: int(
            rng.integers(
                LEAD_TIME_MIN,
                LEAD_TIME_MAX + 1
            )
        )
        for facility_id in FACILITIES
    }

    records = []

    for day in range(1, NUM_DAYS + 1):

        for facility_id, facility in FACILITIES.items():

            supplier_id = facility["supplier_id"]

            base_demand = base_demands[facility_id]

            lead_time = lead_times[facility_id]

            # ------------------------------------------------
            # Demand
            # ------------------------------------------------

            demand = generate_demand(
                facility_id=facility_id,
                day=day,
                base_demand=base_demand
            )

            # Expected demand without random noise.
            weekday = (day - 1) % 7

            expected_demand = (
                base_demand
                * WEEKLY_MULTIPLIER[weekday]
            )

            cluster = facility["cluster"]

            if cluster == "A" and 16 <= day <= 21:
                expected_demand *= 1.55

            elif cluster == "B" and 18 <= day <= 20:
                expected_demand *= 1.25

            expected_demand = round(expected_demand, 2)

            # ------------------------------------------------
            # Replenishment
            # ------------------------------------------------

            replenishment = generate_replenishment(
                facility_id=facility_id,
                day=day,
                demand=demand,
                base_demand=base_demand,
                lead_time=lead_time
            )

            # ------------------------------------------------
            # Supplier disruption status
            # ------------------------------------------------

            supplier_disrupted = (
                DISRUPTION_ENABLED
                and DISRUPTION_START <= day <= DISRUPTION_END
            )

            # ------------------------------------------------
            # Inventory update
            # ------------------------------------------------

            stock_before_demand = (
                inventory[facility_id] + replenishment
            )

            inventory_after_demand = (
                stock_before_demand - demand
            )

            # Inventory cannot be negative.
            inventory_after_demand = max(
                0,
                inventory_after_demand
            )

            stockout = inventory_after_demand <= 0

            inventory[facility_id] = inventory_after_demand

            # ------------------------------------------------
            # Estimated days of stock
            # ------------------------------------------------

            if expected_demand > 0:
                estimated_days_of_stock = round(
                    inventory_after_demand / expected_demand,
                    2
                )
            else:
                estimated_days_of_stock = 0

            # ------------------------------------------------
            # Record
            # ------------------------------------------------

            records.append(
                {
                    "day": day,
                    "facility_id": facility_id,
                    "supplier_id": supplier_id,
                    "latitude": facility["latitude"],
                    "longitude": facility["longitude"],
                    "demand": demand,
                    "expected_demand": expected_demand,
                    "replenishment": replenishment,
                    "stock_before_demand": stock_before_demand,
                    "inventory": inventory_after_demand,
                    "estimated_days_of_stock": estimated_days_of_stock,
                    "supplier_disrupted": supplier_disrupted,
                    "stockout": stockout,
                }
            )

    return pd.DataFrame(records)


# ============================================================
# SAVE + REPORT
# ============================================================

def main():

    print("=" * 60)
    print("SUPPLYSHIELD AI - SYNTHETIC DATA GENERATOR")
    print("=" * 60)

    print("\nGenerating synthetic supply-chain data...\n")

    df = generate_dataset()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Generated {len(df)} records.")
    print(f"Facilities: {df['facility_id'].nunique()}")
    print(f"Suppliers: {df['supplier_id'].nunique()}")
    print(f"Days: {df['day'].nunique()}")
    print(
        f"Disrupted records: "
        f"{df['supplier_disrupted'].sum()}"
    )
    print(
        f"Stockout records: "
        f"{df['stockout'].sum()}"
    )

    print("\nAverage inventory by facility:")

    print(
        df.groupby("facility_id")["inventory"]
        .mean()
        .round(2)
    )

    print("\nMaximum demand by facility:")

    print(
        df.groupby("facility_id")["demand"]
        .max()
    )

    print("\nInventory on disruption boundary:")

    boundary = df[
        df["day"].isin(
            [
                DISRUPTION_START - 2,
                DISRUPTION_START,
                DISRUPTION_END,
                DISRUPTION_END + 2,
            ]
        )
    ]

    boundary_summary = (
        boundary
        .pivot(
            index="facility_id",
            columns="day",
            values="inventory"
        )
        .round(2)
    )

    print(boundary_summary)

    print("\nDataset saved to:")
    print(OUTPUT_PATH)

    print("\nSynthetic data generation complete.")


if __name__ == "__main__":
    main()