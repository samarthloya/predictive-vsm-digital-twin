# Predictive Value Stream Mapping (VSM) & Condition Monitoring using Machine Learning

## Project Overview

This project bridges traditional Lean Manufacturing with Industrie 4.0 principles by upgrading the static Value Stream Mapping (VSM) methodology into a dynamic, predictive algorithm. The project is built in two phases to demonstrate both physical process modeling and real-world data handling:

1. **Phase 1 (Simulation):** A digital twin for a continuous flow surface treatment and powder coating line, modeling physical constraints like batch-loading pretreatment and geometric hanger capacities.
2. **Phase 2 (Real-World Data):** An integration of the open-source AI4I 2020 Predictive Maintenance dataset to monitor real-time CNC sensor telemetry and forecast machine breakdowns before they become bottlenecks.

## Technical Architecture

* **Data Engineering:**
* *Synthetic Generator:* Python (NumPy, Pandas) script simulating realistic powder coating shift data (batch vs. continuous flow constraints).
* *Real-World Dataset:* Implementation of the AI4I 2020 Predictive Maintenance dataset to handle actual industrial sensor data (temperature, torque, RPM, tool wear).


* **Predictive ML Engine:** Scikit-Learn dual-model architecture. Utilizes a Random Forest Regressor to forecast lead times, and a Random Forest Classifier leveraging `predict_proba` to generate a continuous "Machine Health Score" (Predicted Uptime) rather than a simple binary fail state.
* **Digital Twin Dashboards:** Local Streamlit web applications providing interactive UIs for production planners to visualize the predicted Value Stream flow and monitor real-time sensor anomalies.

## Key Lean Manufacturing Insights

Through Random Forest feature importance extraction, the models mathematically validated two critical shop-floor realities:

1. **Physical Flow:** In the surface coating model, **geometric hanger capacity** and **total volume** were the highest drivers of lead time variance, superseding micro-stoppages or color-change setups.
2. **Condition Monitoring:** In the real-world CNC model, **torque** and **rotational speed variance** were the leading indicators of impending machine failure, allowing planners to reroute the VSM before the station's uptime drops to 0%.

## Local Installation & Execution

1. Clone the repository.
2. Install dependencies:

```bash
pip install pandas numpy scikit-learn streamlit

```

**To run the Synthetic Powder Coating Model:**

```bash
python vsm_data_gen.py
python vsm_ml_engine.py
python -m streamlit run app.py

```

**To run the Real-World Predictive Maintenance Monitor:**

1. Ensure `ai4i2020.csv` is in the root directory.
2. Train the real-data model and launch the dashboard:

```bash
python -m streamlit run app_ai4i2020.py

```