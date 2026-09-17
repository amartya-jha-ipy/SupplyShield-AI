"""
SupplyShield AI
Stage 10 - Medicine Shortage Cascade Simulator

Purpose
-------
Simulate how a shortage at one facility can create pressure
on nearby facilities and how a redistribution intervention
could reduce that pressure.

IMPORTANT
---------
This is a prototype simulation.

The underlying medicine inventory, consumption, facility
locations and observed stockouts come from the real dataset.

Supplier relationships, supplier disruption and transportation
assumptions are simulated because they are not observed in the
real dataset.

This simulator is intended for:
    - hackathon demonstration
    - intervention comparison
    - shortage propagation visualization
    - decision-support experimentation

It does NOT claim to reproduce the actual historical supply
network.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# INPUT FILES
# ============================================================

PROTOTYPE_FILE = Path(
    "data/processed/prototype_paracetamol_500mg.csv"
)

RISK_FILE = Path(
    "data/processed/final_shortage_risk.csv"
)

FEASIBILITY_FILE = Path(
    "data/processed/redistribution_feasibility.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

CASCADE_FILE = Path(
    "data/processed/cascade_simulation.csv"
)

INTERVENTION_FILE = Path(
    "data/processed/intervention_results.csv"
)


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

NEIGHBOR_RADIUS_KM = 25.0

MAX_CASCADE_STEPS = 3

SPATIAL_PROPAGATION_WEIGHT = 0.60

SUPPLIER_PROPAGATION_WEIGHT = 0.40

INTERVENTION_EFFECT = 0.75

DONOR_SAFETY_FACTOR = 1.20


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def haversine_distance_km(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate great-circle distance between coordinates.

    Returns distance in kilometres.
    """

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arcsin(
        np.sqrt(a)
    )

    earth_radius_km = 6371.0

    return earth_radius_km * c


def normalize(
    value,
    minimum,
    maximum
):
    """
    Normalize value to [0, 1].
    """

    if maximum <= minimum:
        return 0.0

    return float(
        np.clip(
            (value - minimum)
            /
            (maximum - minimum),
            0.0,
            1.0
        )
    )


def find_column(
    df,
    candidates,
    required=True
):
    """
    Find the first available column from candidates.
    """

    for column in candidates:

        if column in df.columns:
            return column

    if required:

        raise ValueError(
            "Could not find any of these columns: "
            + ", ".join(candidates)
        )

    return None


def estimate_supplier_assignment(
    facilities
):
    """
    Create a deterministic simulated supplier assignment.

    The real dataset does NOT contain supplier IDs.

    We therefore create three simulated supplier regions
    using facility longitude.

    This is only used to demonstrate the shared-supplier
    propagation mechanism.

    Supplier assignment is explicitly labelled as simulated.
    """

    facilities = facilities.copy()

    longitude = facilities[
        "longitude"
    ].astype(float)

    minimum = longitude.min()

    maximum = longitude.max()

    if maximum == minimum:

        normalized_longitude = (
            pd.Series(
                0.5,
                index=facilities.index
            )
        )

    else:

        normalized_longitude = (
            longitude - minimum
        ) / (
            maximum - minimum
        )

    facilities[
        "simulated_supplier_id"
    ] = np.select(
        [
            normalized_longitude < 1 / 3,
            normalized_longitude < 2 / 3,
        ],
        [
            "SIM_SUPPLIER_01",
            "SIM_SUPPLIER_02",
        ],
        default="SIM_SUPPLIER_03"
    )

    return facilities


def choose_default_origin(
    risk_data
):
    """
    Select a default initiating facility.

    Priority:
        1. highest final shortage score
        2. current stockout
        3. highest ML shortage risk
    """

    if risk_data.empty:

        raise ValueError(
            "No shortage-risk records available."
        )

    # Stage 9 calls the observed-stockout column
    # "current_stockout".

    stockout_column = find_column(
        risk_data,
        [
            "current_stockout",
            "risk_current_stockout",
        ]
    )

    ranked = risk_data.sort_values(
        [
            "final_shortage_score",
            stockout_column,
            "ml_shortage_risk",
        ],
        ascending=[
            False,
            False,
            False,
        ]
    )

    return int(
        ranked.iloc[0][
            "facility_id"
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SUPPLYSHIELD AI - SHORTAGE CASCADE SIMULATOR")
    print("=" * 75)

    # ========================================================
    # 1. LOAD REAL FACILITY DATA
    # ========================================================

    print("\n" + "=" * 75)
    print("1. LOADING REAL FACILITY / MEDICINE DATA")
    print("=" * 75)

    if not PROTOTYPE_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: {PROTOTYPE_FILE}"
        )

    prototype = pd.read_csv(
        PROTOTYPE_FILE
    )

    prototype["date"] = pd.to_datetime(
        prototype["date"],
        errors="coerce"
    )

    print(
        f"Prototype records: "
        f"{len(prototype):,}"
    )

    print(
        f"Facilities: "
        f"{prototype['facility_id'].nunique():,}"
    )

    print(
        f"Dates: "
        f"{prototype['date'].nunique():,}"
    )

    # ========================================================
    # 2. CREATE FACILITY SNAPSHOT
    # ========================================================

    print("\n" + "=" * 75)
    print("2. BUILDING FACILITY SNAPSHOT")
    print("=" * 75)

    latest_date = prototype[
        "date"
    ].max()

    latest = prototype[
        prototype[
            "date"
        ] == latest_date
    ].copy()

    latest = (
        latest
        .sort_values(
            "date"
        )
        .drop_duplicates(
            "facility_id",
            keep="last"
        )
    )

    facility_columns = [
        "facility_id",
        "facility_type",
        "latitude",
        "longitude",
        "district",
        "closing_inventory",
        "consumption",
        "stockout",
    ]

    facility_columns = [
        column
        for column in facility_columns
        if column in latest.columns
    ]

    facilities = latest[
        facility_columns
    ].copy()

    facilities = facilities.rename(
        columns={
            "stockout":
                "current_stockout"
        }
    )

    facilities[
        "closing_inventory"
    ] = pd.to_numeric(
        facilities[
            "closing_inventory"
        ],
        errors="coerce"
    ).fillna(0)

    facilities[
        "consumption"
    ] = pd.to_numeric(
        facilities[
            "consumption"
        ],
        errors="coerce"
    ).fillna(0)

    facilities[
        "current_stockout"
    ] = pd.to_numeric(
        facilities[
            "current_stockout"
        ],
        errors="coerce"
    ).fillna(0).astype(int)

    facilities = facilities.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    )

    print(
        f"Latest snapshot date: "
        f"{latest_date.date()}"
    )

    print(
        f"Facilities in snapshot: "
        f"{len(facilities):,}"
    )

    # ========================================================
    # 3. ADD SIMULATED SUPPLIERS
    # ========================================================

    print("\n" + "=" * 75)
    print("3. ADDING SIMULATED SUPPLIER NETWORK")
    print("=" * 75)

    facilities = estimate_supplier_assignment(
        facilities
    )

    print(
        "Supplier network status: SIMULATED"
    )

    print(
        "\nSimulated supplier distribution:"
    )

    print(
        facilities[
            "simulated_supplier_id"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================
    # 4. LOAD FINAL RISK
    # ========================================================

    print("\n" + "=" * 75)
    print("4. LOADING SHORTAGE-RISK OUTPUT")
    print("=" * 75)

    if not RISK_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: {RISK_FILE}"
        )

    risk = pd.read_csv(
        RISK_FILE
    )

    risk["date"] = pd.to_datetime(
        risk["date"],
        errors="coerce"
    )

    print(
        f"Risk records: "
        f"{len(risk):,}"
    )

    latest_risk_date = risk[
        "date"
    ].max()

    latest_risk = risk[
        risk[
            "date"
        ] == latest_risk_date
    ].copy()

    latest_risk = (
        latest_risk
        .sort_values(
            "final_shortage_score",
            ascending=False
        )
        .drop_duplicates(
            "facility_id",
            keep="first"
        )
    )

    # Keep the original Stage 9 column names here.
    # This allows choose_default_origin() to work correctly.

    risk_columns = [
        "facility_id",
        "ml_shortage_risk",
        "final_shortage_score",
        "risk_level",
        "current_stockout",
    ]

    risk_columns = [
        column
        for column in risk_columns
        if column in latest_risk.columns
    ]

    latest_risk = latest_risk[
        risk_columns
    ].copy()

    # Avoid duplicate current_stockout after merging with
    # the real facility snapshot.

    if "current_stockout" in latest_risk.columns:

        latest_risk = latest_risk.rename(
            columns={
                "current_stockout":
                    "risk_current_stockout"
            }
        )

    facilities = facilities.merge(
        latest_risk,
        on="facility_id",
        how="left"
    )

    facilities[
        "ml_shortage_risk"
    ] = facilities[
        "ml_shortage_risk"
    ].fillna(0.0)

    facilities[
        "final_shortage_score"
    ] = facilities[
        "final_shortage_score"
    ].fillna(0.0)

    facilities[
        "risk_level"
    ] = facilities[
        "risk_level"
    ].fillna("UNKNOWN")

    # ========================================================
    # 5. LOAD FEASIBILITY
    # ========================================================

    print("\n" + "=" * 75)
    print("5. LOADING REDISTRIBUTION OPTIONS")
    print("=" * 75)

    if not FEASIBILITY_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: {FEASIBILITY_FILE}"
        )

    feasibility = pd.read_csv(
        FEASIBILITY_FILE
    )

    print(
        f"Feasibility pairs: "
        f"{len(feasibility):,}"
    )

    feasibility[
        "feasibility_score"
    ] = pd.to_numeric(
        feasibility[
            "feasibility_score"
        ],
        errors="coerce"
    ).fillna(0).clip(
        0.0,
        1.0
    )

    best_options = (
        feasibility
        .sort_values(
            "feasibility_score",
            ascending=False
        )
        .drop_duplicates(
            "recipient_facility_id"
        )
    )

    # ========================================================
    # 6. SELECT CASCADE ORIGIN
    # ========================================================

    print("\n" + "=" * 75)
    print("6. SELECTING CASCADE ORIGIN")
    print("=" * 75)

    origin_facility = choose_default_origin(
        latest_risk
    )

    origin_row = facilities[
        facilities[
            "facility_id"
        ] == origin_facility
    ]

    if origin_row.empty:

        raise ValueError(
            "Selected origin facility is not "
            "present in the facility snapshot."
        )

    origin_row = origin_row.iloc[0]

    print(
        f"Origin facility: "
        f"{origin_facility}"
    )

    print(
        f"District: "
        f"{origin_row['district']}"
    )

    print(
        f"Origin inventory: "
        f"{origin_row['closing_inventory']:.2f}"
    )

    print(
        f"Origin consumption: "
        f"{origin_row['consumption']:.2f}"
    )

    print(
        f"Origin ML risk: "
        f"{origin_row['ml_shortage_risk']:.4f}"
    )

    print(
        f"Origin final risk score: "
        f"{origin_row['final_shortage_score']:.4f}"
    )

    print(
        f"Origin observed stockout: "
        f"{int(origin_row['current_stockout'])}"
    )

    # ========================================================
    # 7. FIND NEIGHBOURS
    # ========================================================

    print("\n" + "=" * 75)
    print("7. IDENTIFYING NEARBY FACILITIES")
    print("=" * 75)

    facilities[
        "distance_from_origin_km"
    ] = haversine_distance_km(
        origin_row[
            "latitude"
        ],
        origin_row[
            "longitude"
        ],
        facilities[
            "latitude"
        ].values,
        facilities[
            "longitude"
        ].values,
    )

    neighbours = facilities[
        (
            facilities[
                "distance_from_origin_km"
            ]
            <= NEIGHBOR_RADIUS_KM
        )
        &
        (
            facilities[
                "facility_id"
            ]
            != origin_facility
        )
    ].copy()

    print(
        f"Nearby facilities within "
        f"{NEIGHBOR_RADIUS_KM:.0f} km: "
        f"{len(neighbours):,}"
    )

    if not neighbours.empty:

        print(
            "\nNearby facility sample:"
        )

        print(
            neighbours[
                [
                    "facility_id",
                    "district",
                    "distance_from_origin_km",
                    "closing_inventory",
                    "consumption",
                    "ml_shortage_risk",
                    "final_shortage_score",
                ]
            ]
            .sort_values(
                "distance_from_origin_km"
            )
            .head(10)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # 8. INITIALIZE CASCADE
    # ========================================================

    print("\n" + "=" * 75)
    print("8. RUNNING SHORTAGE PROPAGATION")
    print("=" * 75)

    pressure = (
        facilities
        .set_index(
            "facility_id"
        )[
            "final_shortage_score"
        ]
        .astype(float)
        .to_dict()
    )

    initial_pressure = pressure.copy()

    affected = {
        origin_facility
    }

    cascade_records = []

    origin_supplier = origin_row[
        "simulated_supplier_id"
    ]

    for step in range(
        1,
        MAX_CASCADE_STEPS + 1
    ):

        print(
            f"\nCascade step {step}"
        )

        new_affected = set()

        for _, facility in neighbours.iterrows():

            facility_id = int(
                facility[
                    "facility_id"
                ]
            )

            distance = float(
                facility[
                    "distance_from_origin_km"
                ]
            )

            distance_factor = max(
                0.0,
                1.0
                -
                (
                    distance
                    /
                    NEIGHBOR_RADIUS_KM
                )
            )

            spatial_pressure = (
                initial_pressure[
                    origin_facility
                ]
                *
                distance_factor
                *
                SPATIAL_PROPAGATION_WEIGHT
            )

            shared_supplier = (
                facility[
                    "simulated_supplier_id"
                ]
                ==
                origin_supplier
            )

            supplier_pressure = (
                initial_pressure[
                    origin_facility
                ]
                *
                SUPPLIER_PROPAGATION_WEIGHT
                if shared_supplier
                else 0.0
            )

            propagation_pressure = (
                spatial_pressure
                +
                supplier_pressure
            )

            old_pressure = pressure.get(
                facility_id,
                0.0
            )

            new_pressure = min(
                1.0,
                old_pressure
                +
                propagation_pressure
                /
                max(
                    1,
                    step
                )
            )

            pressure[
                facility_id
            ] = new_pressure

            if (
                new_pressure
                >= 0.50
            ):

                new_affected.add(
                    facility_id
                )

            cascade_records.append(
                {
                    "origin_facility_id":
                        origin_facility,

                    "facility_id":
                        facility_id,

                    "cascade_step":
                        step,

                    "distance_from_origin_km":
                        distance,

                    "shared_simulated_supplier":
                        int(shared_supplier),

                    "spatial_propagation":
                        spatial_pressure,

                    "supplier_propagation":
                        supplier_pressure,

                    "propagation_pressure":
                        propagation_pressure,

                    "initial_pressure":
                        old_pressure,

                    "new_pressure":
                        new_pressure,
                }
            )

        affected.update(
            new_affected
        )

        print(
            f"Facilities at or above "
            f"0.50 pressure: "
            f"{len(affected):,}"
        )

    cascade = pd.DataFrame(
        cascade_records
    )

    # ========================================================
    # 9. CASCADE SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("9. CASCADE SUMMARY")
    print("=" * 75)

    initial_high_risk = sum(
        value >= 0.50
        for value in initial_pressure.values()
    )

    final_high_risk = sum(
        value >= 0.50
        for value in pressure.values()
    )

    print(
        f"Initial facilities with "
        f"risk/pressure >= 0.50: "
        f"{initial_high_risk:,}"
    )

    print(
        f"After simulated propagation: "
        f"{final_high_risk:,}"
    )

    print(
        f"Additional facilities crossing "
        f"0.50: "
        f"{max(0, final_high_risk - initial_high_risk):,}"
    )

    # ========================================================
    # 10. SELECT INTERVENTION TARGET
    # ========================================================

    print("\n" + "=" * 75)
    print("10. SELECTING INTERVENTION")
    print("=" * 75)

    intervention_candidates = best_options[
        best_options[
            "recipient_facility_id"
        ].isin(
            neighbours[
                "facility_id"
            ]
        )
    ].copy()

    if intervention_candidates.empty:

        intervention_candidates = best_options.copy()

    intervention_candidates = (
        intervention_candidates
        .sort_values(
            [
                "recipient_stockout_probability",
                "feasibility_score",
            ],
            ascending=[
                False,
                False,
            ]
        )
    )

    if intervention_candidates.empty:

        print(
            "No redistribution option available."
        )

        intervention = None

    else:

        intervention = (
            intervention_candidates
            .iloc[0]
        )

        print(
            f"Recipient facility: "
            f"{int(intervention['recipient_facility_id'])}"
        )

        print(
            f"Donor facility: "
            f"{int(intervention['donor_facility_id'])}"
        )

        print(
            f"Distance: "
            f"{intervention['distance_km']:.2f} km"
        )

        print(
            f"Feasibility score: "
            f"{intervention['feasibility_score']:.4f}"
        )

        print(
            f"Transit feasibility probability: "
            f"{intervention['transit_feasibility_probability']:.4f}"
        )

    # ========================================================
    # 11. INTERVENTION COUNTERFACTUAL
    # ========================================================

    print("\n" + "=" * 75)
    print("11. RUNNING INTERVENTION COUNTERFACTUAL")
    print("=" * 75)

    intervention_records = []

    if intervention is not None:

        recipient_id = int(
            intervention[
                "recipient_facility_id"
            ]
        )

        donor_id = int(
            intervention[
                "donor_facility_id"
            ]
        )

        feasibility_score = float(
            intervention[
                "feasibility_score"
            ]
        )

        transit_probability = float(
            intervention[
                "transit_feasibility_probability"
            ]
        )

        recipient_row = facilities[
            facilities[
                "facility_id"
            ]
            ==
            recipient_id
        ]

        donor_row = facilities[
            facilities[
                "facility_id"
            ]
            ==
            donor_id
        ]

        if (
            not recipient_row.empty
            and
            not donor_row.empty
        ):

            recipient_row = recipient_row.iloc[0]

            donor_row = donor_row.iloc[0]

            before_pressure = pressure.get(
                recipient_id,
                float(
                    recipient_row[
                        "final_shortage_score"
                    ]
                )
            )

            effective_intervention = (
                INTERVENTION_EFFECT
                *
                feasibility_score
                *
                transit_probability
            )

            after_pressure = max(
                0.0,
                before_pressure
                *
                (
                    1.0
                    -
                    effective_intervention
                )
            )

            pressure_reduction = (
                before_pressure
                -
                after_pressure
            )

            donor_available = (
                float(
                    donor_row[
                        "closing_inventory"
                    ]
                )
                >
                float(
                    donor_row[
                        "consumption"
                    ]
                )
                *
                DONOR_SAFETY_FACTOR
            )

            intervention_records.append(
                {
                    "recipient_facility_id":
                        recipient_id,

                    "donor_facility_id":
                        donor_id,

                    "distance_km":
                        float(
                            intervention[
                                "distance_km"
                            ]
                        ),

                    "feasibility_score":
                        feasibility_score,

                    "transit_feasibility_probability":
                        transit_probability,

                    "effective_intervention_strength":
                        effective_intervention,

                    "recipient_pressure_before":
                        before_pressure,

                    "recipient_pressure_after":
                        after_pressure,

                    "pressure_reduction":
                        pressure_reduction,

                    "donor_inventory":
                        float(
                            donor_row[
                                "closing_inventory"
                            ]
                        ),

                    "donor_consumption":
                        float(
                            donor_row[
                                "consumption"
                            ]
                        ),

                    "donor_has_safety_surplus":
                        int(
                            donor_available
                        ),

                    "intervention_status":
                        (
                            "SIMULATED_SUCCESS"
                            if (
                                donor_available
                                and
                                effective_intervention
                                > 0
                            )
                            else
                            "SIMULATED_CONSTRAINED"
                        ),
                }
            )

            print(
                f"Recipient pressure before: "
                f"{before_pressure:.4f}"
            )

            print(
                f"Recipient pressure after: "
                f"{after_pressure:.4f}"
            )

            print(
                f"Pressure reduction: "
                f"{pressure_reduction:.4f}"
            )

            print(
                f"Donor safety surplus: "
                f"{'YES' if donor_available else 'NO'}"
            )

    # ========================================================
    # 12. SAVE OUTPUTS
    # ========================================================

    print("\n" + "=" * 75)
    print("12. SAVING OUTPUTS")
    print("=" * 75)

    CASCADE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cascade.to_csv(
        CASCADE_FILE,
        index=False
    )

    intervention_results = pd.DataFrame(
        intervention_records
    )

    intervention_results.to_csv(
        INTERVENTION_FILE,
        index=False
    )

    print(
        f"Saved: {CASCADE_FILE}"
    )

    print(
        f"Saved: {INTERVENTION_FILE}"
    )

    # ========================================================
    # 13. COMPLETE
    # ========================================================

    print("\n" + "=" * 75)
    print("STAGE 10 COMPLETE")
    print("=" * 75)

    print(
        "\nThe cascade simulator now supports:"
    )

    print(
        "1. Spatial shortage propagation"
    )

    print(
        "2. Simulated shared-supplier propagation"
    )

    print(
        "3. Multi-step cascade simulation"
    )

    print(
        "4. Redistribution intervention"
    )

    print(
        "5. Before-vs-after intervention comparison"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Supplier assignments and transport effects are "
        "simulated prototype assumptions."
    )

    print(
        "They are not observed supplier-network data."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()