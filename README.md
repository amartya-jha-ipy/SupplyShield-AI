# SupplyShield AI

### AI-Powered Early Detection and Response System for Medicine Shortages

SupplyShield AI is an AI-driven prototype designed to detect **emerging medicine shortages before they become widespread**, identify facilities at elevated risk, detect regional shortage patterns, simulate shortage propagation, and evaluate potential redistribution interventions.

The system combines **machine learning, spatial analysis, anomaly detection, demand forecasting, risk fusion, explainable AI, and intervention simulation** into a unified decision-support platform.

---

## 🚨 Problem

A medicine shortage at one healthcare facility may initially appear to be an isolated inventory issue. However, declining stock across multiple nearby facilities can indicate a broader supply disruption.

Shortages can spread because of:

* Increasing or anomalous demand
* Delayed replenishment
* Uneven inventory distribution
* Regional shortage patterns
* Limited visibility across healthcare facilities
* Geographic constraints on redistribution

Without a shared analytical view, surplus inventory at one facility may remain unused while another facility approaches a stockout.

**SupplyShield AI aims to provide an early-warning layer for these situations.**

---

## 💡 Solution

SupplyShield AI analyzes historical facility-level medicine data and generates a multi-signal shortage risk assessment.

The platform:

1. Predicts the probability of a future stockout.
2. Detects unusual demand and inventory behavior.
3. Identifies spatial and regional shortage signals.
4. Combines multiple signals into a final shortage-risk score.
5. Simulates how shortage pressure could propagate across nearby facilities.
6. Identifies potential donor facilities with available inventory.
7. Estimates redistribution feasibility.
8. Provides explainable evidence behind each risk alert.
9. Presents the results through an interactive dashboard.

---

## 🏗️ System Architecture

```text
                    Historical Medicine Data
                              │
                              ▼
                    Data Preparation
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
        Demand / Inventory             Spatial Analysis
             Analysis                       │
                │                           │
                ▼                           ▼
       Anomaly Detection          Regional Shortage Signal
                │                           │
                └─────────────┬─────────────┘
                              ▼
                    Stockout Prediction
                              │
                              ▼
                     Risk Signal Fusion
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
        Explainable Risk            Cascade Simulation
                │                           │
                └─────────────┬─────────────┘
                              ▼
                  Redistribution Simulation
                              │
                              ▼
                    SupplyShield Dashboard
```

---

## 🧠 Machine Learning Pipeline

### 1. Stockout Risk Prediction

A LightGBM classification model is used to estimate the probability that a facility will experience a stockout in the following period.

The model uses historical inventory, consumption, temporal, and facility-level information.

**Prototype performance:**

| Metric  |     Result |
| ------- | ---------: |
| ROC-AUC | **0.7268** |
| PR-AUC  | **0.6865** |

---

### 2. Demand & Inventory Anomaly Detection

The system identifies unusual demand and inventory behavior that may indicate emerging supply pressure.

Anomaly signals are incorporated into the overall shortage-risk assessment rather than being treated as standalone proof of a shortage.

---

### 3. Spatial Shortage Detection

Nearby facilities are analyzed to identify geographically correlated shortage conditions.

The prototype evaluates whether stockout events are concentrated around a facility or region.

**Spatial validation result:**

```text
Stockout rate with regional signal    : 55.79%
Stockout rate without regional signal: 19.05%

Relative signal ratio ≈ 2.93×
```

This indicates that the regional signal provides useful information for identifying areas experiencing elevated shortage pressure.

---

### 4. Risk Signal Fusion

Multiple evidence sources are combined into a final shortage-risk score.

Signals include:

* ML-predicted stockout probability
* Current stockout status
* Spatial shortage pressure
* Demand anomalies
* Regional shortage signals
* Other observed evidence

**Final fusion validation:**

| Metric  |     Result |
| ------- | ---------: |
| ROC-AUC | **0.6706** |
| PR-AUC  | **0.6201** |

---

## 🔬 Explainable AI

SupplyShield AI does not only output a risk score.

For every high-priority facility, the system provides evidence explaining **why the facility was flagged**.

Example evidence:

```text
ML predicted shortage risk: 94.2%
Current stockout: YES
Nearby pressure: 90.0%
Regional shortage signal: DETECTED
Demand pattern: ANOMALOUS
Evidence signals: 4
```

This allows a decision-maker to investigate the underlying signals instead of relying on an unexplained prediction.

---

## 🔄 Shortage Cascade Simulation

The system can simulate how shortage pressure may propagate from one facility to nearby facilities.

The prototype combines:

* Geographic proximity
* Spatial pressure
* Simulated supplier relationships
* Propagation pressure

The cascade module produces:

```text
Origin Facility
      │
      ▼
Step 1 ──► Nearby facilities
      │
      ▼
Step 2 ──► Additional facilities
      │
      ▼
Step 3 ──► Further propagation
```

> **Important:** Supplier relationships and propagation behavior in this prototype are simulated assumptions rather than observed real-world supplier-network data.

---

## 📦 Redistribution & Intervention Simulation

When a facility is identified as being at risk, SupplyShield AI evaluates potential donor facilities.

The prototype considers:

* Donor inventory
* Safety surplus
* Distance
* Recipient shortage pressure
* Transit feasibility
* Intervention strength

Example:

```text
Recipient Facility
        │
        │ shortage pressure
        ▼
Potential Donor Facilities
        │
        ├── Inventory availability
        ├── Safety surplus
        ├── Distance
        └── Transit feasibility
                │
                ▼
       Simulated Intervention
```

The intervention results are **simulation outputs**, not observed logistics transactions.

---

## 📊 Dashboard

The web dashboard provides:

### System Overview

* Number of facilities
* Active risk alerts
* High-risk facilities
* Regional shortage signals

### Risk Overview

* Very Low
* Low
* Moderate
* High

### Facility Investigation

Each facility can be investigated individually to view:

* Risk score
* ML probability
* Spatial pressure
* Stock status
* Demand behavior
* Regional signals
* Supporting evidence
* Explainability
* Potential redistribution options

### Cascade Analysis

The dashboard also presents simulated shortage propagation and intervention results.

---

## 🗂️ Project Structure

```text
SupplyShield-AI/
│
├── data/
│   ├── real/
│   │   └── S2_Dhis2Data.csv
│   │
│   └── synthetic/
│       ├── config.yaml
│       ├── demand_anomalies.csv
│       ├── demand_forecasts.csv
│       ├── regional_anomalies.csv
│       ├── supplier_disruptions.csv
│       ├── supply_chain_data.csv
│       └── synchronization_results.csv
│
├── notebooks/
│   ├── detect_spatial_shortage.py
│   ├── prepare_prototype_data.py
│   ├── profile_real_data.py
│   ├── select_prototype_medicine.py
│   ├── train_shortage_risk.py
│   ├── validate_anomaly_signal.py
│   ├── validate_cascade.py
│   ├── validate_feasibility.py
│   ├── validate_final_risk.py
│   ├── validate_model.py
│   └── validate_spatial_signal.py
│
├── src/
│   ├── backend/
│   │   └── app.py
│   │
│   ├── frontend/
│   │   ├── index.html
│   │   ├── facility.html
│   │   ├── style.css
│   │   ├── app.js
│   │   └── facility.js
│   │
│   ├── ml/
│   │   ├── anomaly_clusterer.py
│   │   ├── cascade_simulator.py
│   │   ├── demand_forecaster.py
│   │   ├── feasibility_scorer.py
│   │   ├── final_risk_fusion.py
│   │   ├── sync_detector.py
│   │   └── xai_explainer.py
│   │
│   └── synthetic_data_generator.py
│
└── README.md
```

---

## 🛠️ Technology Stack

### Machine Learning

* Python
* LightGBM
* Scikit-learn
* Pandas
* NumPy

### Backend

* Python
* Flask
* RES
