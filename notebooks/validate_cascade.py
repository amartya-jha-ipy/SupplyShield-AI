import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

CASCADE_FILE = "data/processed/cascade_simulation.csv"

INTERVENTION_FILE = "data/processed/intervention_results.csv"

OUTPUT_DIR = "data/processed"

CASCADE_VALIDATION_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "cascade_validation.csv"
)

INTERVENTION_VALIDATION_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "intervention_validation.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SUPPLYSHIELD AI - CASCADE VALIDATION")
    print("=" * 75)

    # ========================================================
    # 1. LOAD CASCADE
    # ========================================================

    print("\n" + "=" * 75)
    print("1. LOADING CASCADE SIMULATION")
    print("=" * 75)

    if not os.path.exists(CASCADE_FILE):
        raise FileNotFoundError(
            f"Cascade file not found: {CASCADE_FILE}"
        )

    cascade = pd.read_csv(CASCADE_FILE)

    print(f"Cascade records: {len(cascade):,}")
    print(f"Columns: {list(cascade.columns)}")

    # ========================================================
    # 2. LOAD INTERVENTION
    # ========================================================

    print("\n" + "=" * 75)
    print("2. LOADING INTERVENTION RESULTS")
    print("=" * 75)

    if not os.path.exists(INTERVENTION_FILE):
        raise FileNotFoundError(
            f"Intervention file not found: {INTERVENTION_FILE}"
        )

    intervention = pd.read_csv(INTERVENTION_FILE)

    print(
        f"Intervention records: {len(intervention):,}"
    )

    print(
        f"Columns: {list(intervention.columns)}"
    )

    # ========================================================
    # 3. IDENTIFY CASCADE COLUMNS
    # ========================================================

    print("\n" + "=" * 75)
    print("3. IDENTIFYING CASCADE COLUMNS")
    print("=" * 75)

    required_cascade = [
        "origin_facility_id",
        "facility_id",
        "cascade_step",
        "distance_from_origin_km",
        "shared_simulated_supplier",
        "spatial_propagation",
        "supplier_propagation",
        "propagation_pressure",
        "initial_pressure",
        "new_pressure",
    ]

    missing_cascade = [
        column
        for column in required_cascade
        if column not in cascade.columns
    ]

    if missing_cascade:

        raise ValueError(
            "Missing cascade columns: "
            + ", ".join(missing_cascade)
        )

    print(
        "Origin facility column: origin_facility_id"
    )

    print(
        "Facility column: facility_id"
    )

    print(
        "Cascade step column: cascade_step"
    )

    print(
        "Distance column: distance_from_origin_km"
    )

    print(
        "Propagation pressure column: propagation_pressure"
    )

    print(
        "Initial pressure column: initial_pressure"
    )

    print(
        "New pressure column: new_pressure"
    )

    # ========================================================
    # 4. IDENTIFY INTERVENTION COLUMNS
    # ========================================================

    print("\n" + "=" * 75)
    print("4. IDENTIFYING INTERVENTION COLUMNS")
    print("=" * 75)

    required_intervention = [
        "recipient_facility_id",
        "donor_facility_id",
        "distance_km",
        "feasibility_score",
        "transit_feasibility_probability",
        "effective_intervention_strength",
        "recipient_pressure_before",
        "recipient_pressure_after",
        "pressure_reduction",
        "donor_inventory",
        "donor_consumption",
        "donor_has_safety_surplus",
        "intervention_status",
    ]

    missing_intervention = [
        column
        for column in required_intervention
        if column not in intervention.columns
    ]

    if missing_intervention:

        raise ValueError(
            "Missing intervention columns: "
            + ", ".join(missing_intervention)
        )

    print(
        "Recipient facility column: recipient_facility_id"
    )

    print(
        "Donor facility column: donor_facility_id"
    )

    print(
        "Feasibility column: feasibility_score"
    )

    print(
        "Transit probability column: "
        "transit_feasibility_probability"
    )

    print(
        "Pressure before column: "
        "recipient_pressure_before"
    )

    print(
        "Pressure after column: "
        "recipient_pressure_after"
    )

    print(
        "Pressure reduction column: pressure_reduction"
    )

    # ========================================================
    # 5. NUMERIC CONVERSION
    # ========================================================

    print("\n" + "=" * 75)
    print("5. VALIDATING NUMERIC VALUES")
    print("=" * 75)

    cascade_numeric = [
        "cascade_step",
        "distance_from_origin_km",
        "spatial_propagation",
        "supplier_propagation",
        "propagation_pressure",
        "initial_pressure",
        "new_pressure",
    ]

    intervention_numeric = [
        "distance_km",
        "feasibility_score",
        "transit_feasibility_probability",
        "effective_intervention_strength",
        "recipient_pressure_before",
        "recipient_pressure_after",
        "pressure_reduction",
        "donor_inventory",
        "donor_consumption",
    ]

    for column in cascade_numeric:

        cascade[column] = pd.to_numeric(
            cascade[column],
            errors="coerce"
        )

    for column in intervention_numeric:

        intervention[column] = pd.to_numeric(
            intervention[column],
            errors="coerce"
        )

    invalid_cascade = int(
        cascade[cascade_numeric]
        .isna()
        .any(axis=1)
        .sum()
    )

    invalid_intervention = int(
        intervention[intervention_numeric]
        .isna()
        .any(axis=1)
        .sum()
    )

    print(
        f"Cascade rows with invalid numeric values: "
        f"{invalid_cascade:,}"
    )

    print(
        f"Intervention rows with invalid numeric values: "
        f"{invalid_intervention:,}"
    )

    # ========================================================
    # 6. BASIC CASCADE STATISTICS
    # ========================================================

    print("\n" + "=" * 75)
    print("6. CASCADE BASIC STATISTICS")
    print("=" * 75)

    unique_facilities = cascade[
        "facility_id"
    ].nunique()

    unique_origins = cascade[
        "origin_facility_id"
    ].nunique()

    print(
        f"Unique affected facilities: "
        f"{unique_facilities:,}"
    )

    print(
        f"Unique cascade origins: "
        f"{unique_origins:,}"
    )

    print(
        f"Maximum cascade step: "
        f"{int(cascade['cascade_step'].max())}"
    )

    print(
        "\nRecords by cascade step:"
    )

    step_summary = (
        cascade
        .groupby("cascade_step")
        .agg(
            facilities=(
                "facility_id",
                "nunique"
            ),
            records=(
                "facility_id",
                "count"
            ),
            average_pressure=(
                "propagation_pressure",
                "mean"
            ),
            maximum_pressure=(
                "propagation_pressure",
                "max"
            ),
        )
        .reset_index()
        .sort_values("cascade_step")
    )

    print(
        step_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 7. PRESSURE VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("7. PROPAGATION PRESSURE VALIDATION")
    print("=" * 75)

    pressure = cascade[
        "propagation_pressure"
    ]

    print(
        f"Minimum propagation pressure: "
        f"{pressure.min():.4f}"
    )

    print(
        f"Maximum propagation pressure: "
        f"{pressure.max():.4f}"
    )

    print(
        f"Mean propagation pressure: "
        f"{pressure.mean():.4f}"
    )

    print(
        f"Median propagation pressure: "
        f"{pressure.median():.4f}"
    )

    negative_pressure = int(
        (pressure < 0).sum()
    )

    above_one = int(
        (pressure > 1).sum()
    )

    print(
        f"Negative-pressure records: "
        f"{negative_pressure:,}"
    )

    print(
        f"Pressure records > 1: "
        f"{above_one:,}"
    )

    # ========================================================
    # 8. COMPONENT VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("8. PROPAGATION COMPONENT VALIDATION")
    print("=" * 75)

    for column in [
        "spatial_propagation",
        "supplier_propagation",
        "propagation_pressure",
    ]:

        values = cascade[column]

        print(
            f"\n{column}:"
        )

        print(
            f"  minimum = {values.min():.4f}"
        )

        print(
            f"  maximum = {values.max():.4f}"
        )

        print(
            f"  mean    = {values.mean():.4f}"
        )

    # ========================================================
    # 9. PRESSURE FORMULA CONSISTENCY
    # ========================================================

    print("\n" + "=" * 75)
    print("9. PRESSURE CONSISTENCY CHECK")
    print("=" * 75)

    calculated_pressure = (
        cascade["spatial_propagation"]
        + cascade["supplier_propagation"]
    )

    difference = (
        cascade["propagation_pressure"]
        - calculated_pressure
    ).abs()

    print(
        "Checking whether:"
    )

    print(
        "propagation_pressure = "
        "spatial_propagation + supplier_propagation"
    )

    print(
        f"Maximum difference: "
        f"{difference.max():.8f}"
    )

    print(
        f"Mean difference: "
        f"{difference.mean():.8f}"
    )

    consistent = bool(
        (difference <= 1e-6).all()
    )

    print(
        f"Pressure formula consistent: "
        f"{'YES' if consistent else 'NO'}"
    )

    # ========================================================
    # 10. INITIAL -> NEW PRESSURE CHECK
    # ========================================================

    print("\n" + "=" * 75)
    print("10. INITIAL / NEW PRESSURE ANALYSIS")
    print("=" * 75)

    cascade["pressure_change"] = (
        cascade["new_pressure"]
        - cascade["initial_pressure"]
    )

    increased = int(
        (
            cascade["pressure_change"]
            > 0
        ).sum()
    )

    unchanged = int(
        (
            cascade["pressure_change"]
            == 0
        ).sum()
    )

    decreased = int(
        (
            cascade["pressure_change"]
            < 0
        ).sum()
    )

    print(
        f"Pressure increased: "
        f"{increased:,}"
    )

    print(
        f"Pressure unchanged: "
        f"{unchanged:,}"
    )

    print(
        f"Pressure decreased: "
        f"{decreased:,}"
    )

    print(
        f"Average initial pressure: "
        f"{cascade['initial_pressure'].mean():.4f}"
    )

    print(
        f"Average new pressure: "
        f"{cascade['new_pressure'].mean():.4f}"
    )

    # ========================================================
    # 11. THRESHOLD ANALYSIS
    # ========================================================

    print("\n" + "=" * 75)
    print("11. CASCADE THRESHOLD ANALYSIS")
    print("=" * 75)

    threshold = 0.50

    cascade["above_threshold"] = (
        cascade["propagation_pressure"]
        >= threshold
    )

    threshold_count = int(
        cascade["above_threshold"].sum()
    )

    threshold_fraction = (
        threshold_count
        / len(cascade)
    )

    print(
        f"Threshold: {threshold:.2f}"
    )

    print(
        f"Records at/above threshold: "
        f"{threshold_count:,}"
    )

    print(
        f"Fraction at/above threshold: "
        f"{threshold_fraction:.2%}"
    )

    threshold_by_step = (
        cascade
        .groupby("cascade_step")
        .agg(
            records=(
                "facility_id",
                "count"
            ),
            above_threshold=(
                "above_threshold",
                "sum"
            ),
            threshold_fraction=(
                "above_threshold",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        "\nThreshold crossing by cascade step:"
    )

    print(
        threshold_by_step.to_string(
            index=False
        )
    )

    # ========================================================
    # 12. DISTANCE VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("12. GEOGRAPHIC PROPAGATION VALIDATION")
    print("=" * 75)

    distance = cascade[
        "distance_from_origin_km"
    ]

    print(
        f"Minimum distance: "
        f"{distance.min():.2f} km"
    )

    print(
        f"Maximum distance: "
        f"{distance.max():.2f} km"
    )

    print(
        f"Mean distance: "
        f"{distance.mean():.2f} km"
    )

    print(
        f"Median distance: "
        f"{distance.median():.2f} km"
    )

    invalid_distance = int(
        (distance < 0).sum()
    )

    print(
        f"Negative-distance records: "
        f"{invalid_distance:,}"
    )

    # ========================================================
    # 13. SUPPLIER PROPAGATION
    # ========================================================

    print("\n" + "=" * 75)
    print("13. SIMULATED SUPPLIER PROPAGATION")
    print("=" * 75)

    supplier_column = (
        "shared_simulated_supplier"
    )

    supplier_counts = (
        cascade[supplier_column]
        .value_counts(dropna=False)
    )

    print(
        supplier_counts.to_string()
    )

    print(
        "\nIMPORTANT: supplier relationships "
        "are simulated, not observed."
    )

    # ========================================================
    # 14. INTERVENTION VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("14. INTERVENTION COUNTERFACTUAL VALIDATION")
    print("=" * 75)

    intervention["calculated_reduction"] = (
        intervention[
            "recipient_pressure_before"
        ]
        -
        intervention[
            "recipient_pressure_after"
        ]
    )

    intervention["reduction_difference"] = (
        intervention[
            "calculated_reduction"
        ]
        -
        intervention[
            "pressure_reduction"
        ]
    ).abs()

    print(
        f"Valid intervention records: "
        f"{len(intervention):,}"
    )

    print(
        f"Average recipient pressure BEFORE: "
        f"{intervention['recipient_pressure_before'].mean():.4f}"
    )

    print(
        f"Average recipient pressure AFTER: "
        f"{intervention['recipient_pressure_after'].mean():.4f}"
    )

    print(
        f"Average pressure reduction: "
        f"{intervention['pressure_reduction'].mean():.4f}"
    )

    positive_reductions = int(
        (
            intervention["pressure_reduction"]
            > 0
        ).sum()
    )

    print(
        f"Interventions reducing pressure: "
        f"{positive_reductions:,}"
    )

    print(
        f"Maximum reduction calculation error: "
        f"{intervention['reduction_difference'].max():.8f}"
    )

    # ========================================================
    # 15. FEASIBILITY VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("15. FEASIBILITY VALIDATION")
    print("=" * 75)

    feasibility = intervention[
        "feasibility_score"
    ]

    transit = intervention[
        "transit_feasibility_probability"
    ]

    print(
        f"Feasibility range: "
        f"{feasibility.min():.4f} - "
        f"{feasibility.max():.4f}"
    )

    print(
        f"Transit probability range: "
        f"{transit.min():.4f} - "
        f"{transit.max():.4f}"
    )

    invalid_feasibility = int(
        (
            (feasibility < 0)
            |
            (feasibility > 1)
        ).sum()
    )

    invalid_transit = int(
        (
            (transit < 0)
            |
            (transit > 1)
        ).sum()
    )

    print(
        f"Feasibility outside [0,1]: "
        f"{invalid_feasibility:,}"
    )

    print(
        f"Transit probability outside [0,1]: "
        f"{invalid_transit:,}"
    )

    print(
        "\nDonor safety status:"
    )

    print(
        intervention[
            "donor_has_safety_surplus"
        ]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        "\nIntervention status:"
    )

    print(
        intervention[
            "intervention_status"
        ]
        .value_counts(dropna=False)
        .to_string()
    )

    # ========================================================
    # 16. SAVE CASCADE VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("16. SAVING VALIDATION OUTPUTS")
    print("=" * 75)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    cascade_summary = pd.DataFrame(
        [
            {
                "metric": "cascade_records",
                "value": len(cascade),
            },
            {
                "metric": "unique_facilities",
                "value": unique_facilities,
            },
            {
                "metric": "unique_origins",
                "value": unique_origins,
            },
            {
                "metric": "maximum_cascade_step",
                "value": cascade[
                    "cascade_step"
                ].max(),
            },
            {
                "metric": "minimum_pressure",
                "value": pressure.min(),
            },
            {
                "metric": "maximum_pressure",
                "value": pressure.max(),
            },
            {
                "metric": "mean_pressure",
                "value": pressure.mean(),
            },
            {
                "metric": "median_pressure",
                "value": pressure.median(),
            },
            {
                "metric": "records_above_0_50",
                "value": threshold_count,
            },
            {
                "metric": "fraction_above_0_50",
                "value": threshold_fraction,
            },
            {
                "metric": "pressure_formula_consistent",
                "value": consistent,
            },
            {
                "metric": "negative_pressure_records",
                "value": negative_pressure,
            },
            {
                "metric": "negative_distance_records",
                "value": invalid_distance,
            },
        ]
    )

    cascade_summary.to_csv(
        CASCADE_VALIDATION_OUTPUT,
        index=False
    )

    print(
        f"Saved: {CASCADE_VALIDATION_OUTPUT}"
    )

    # ========================================================
    # 17. SAVE INTERVENTION VALIDATION
    # ========================================================

    intervention_summary = pd.DataFrame(
        [
            {
                "metric": "valid_interventions",
                "value": len(intervention),
            },
            {
                "metric": "mean_pressure_before",
                "value": intervention[
                    "recipient_pressure_before"
                ].mean(),
            },
            {
                "metric": "mean_pressure_after",
                "value": intervention[
                    "recipient_pressure_after"
                ].mean(),
            },
            {
                "metric": "mean_pressure_reduction",
                "value": intervention[
                    "pressure_reduction"
                ].mean(),
            },
            {
                "metric": "positive_reductions",
                "value": positive_reductions,
            },
            {
                "metric": "maximum_reduction_error",
                "value": intervention[
                    "reduction_difference"
                ].max(),
            },
            {
                "metric": "invalid_feasibility_values",
                "value": invalid_feasibility,
            },
            {
                "metric": "invalid_transit_values",
                "value": invalid_transit,
            },
        ]
    )

    intervention_summary.to_csv(
        INTERVENTION_VALIDATION_OUTPUT,
        index=False
    )

    print(
        f"Saved: {INTERVENTION_VALIDATION_OUTPUT}"
    )

    # ========================================================
    # 18. FINAL STATUS
    # ========================================================

    print("\n" + "=" * 75)
    print("STAGE 11E COMPLETE")
    print("=" * 75)

    print(
        "Cascade propagation outputs were checked "
        "for structural consistency."
    )

    print(
        "Intervention counterfactual outputs were "
        "checked for pressure reduction consistency."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The cascade and supplier network are simulated "
        "prototype assumptions."
    )

    print(
        "This validation therefore measures internal "
        "consistency and plausible behavior, not "
        "real-world cascade prediction accuracy."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()