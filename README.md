# Predictive Value Stream Mapping (VSM) & Condition Monitoring

## Project Overview

This project bridges traditional Lean Manufacturing with Industrie 4.0 principles by upgrading the static Value Stream Mapping (VSM) methodology into a dynamic, predictive algorithm. The project is split into two operational models:

1. **Synthetic Digital Twin:** Simulates a continuous flow surface treatment and powder coating line to predict total manufacturing lead times and dynamically identify shifting bottlenecks based on physical daily production variables.
2. **Real-World Condition Monitoring:** Utilizes the publicly available AI4I 2020 Predictive Maintenance dataset to forecast machine breakdowns. It translates binary failure predictions into a continuous probabilistic "Machine Health Score" to prevent unplanned downtime.

## Technical Architecture

* **Data Pipelines:** Python (NumPy, Pandas) for both simulating physical constraint data (batch-loading pretreatment, overhead conveyor hanger capacities) and preprocessing real-world multivariate CNC sensor data.
* **Predictive ML Engine:** Scikit-Learn utilizing Random Forest Regressors for lead time forecasting, and Random Forest Classifiers for both bottleneck identification and predictive maintenance (PdM).
* **Digital Twin Dashboards:** Interactive Streamlit web applications providing UIs for production planners to visualize predicted Value Stream flow and monitor real-time machine health degradation.

## Key Lean Manufacturing Insights

* **Physical Constraints (Synthetic Model):** Feature importance extraction validated that geometric hanger capacity and total batch volume are the highest drivers of lead time variance. Optimizing physical part-loading configurations significantly supersedes efforts to reduce color-change setup delays.
* **Predictive Maintenance (AI4I Model):** Machine learning analysis proved that torque and rotational speed variance are the heaviest predictors of impending machine failure, allowing planners to schedule proactive maintenance before a station reaches 0% uptime and becomes a bottleneck.

## Local Installation & Execution

1. Clone the repository.
2. Install dependencies:

```bash
pip install pandas numpy scikit-learn streamlit joblib

```

### Option A: Run the Synthetic Powder Coating Simulation

3. Generate the physical constraint dataset:

```bash
python vsm_data_gen.py

```

4. Train the predictive models:

```bash
python vsm_ml_engine.py

```

5. Launch the interactive dashboard:

```bash
python -m streamlit run app.py

```

### Option B: Run the Real-World AI4I PdM Model

3. Ensure the downloaded `ai4i2020.csv` file is in your project directory.
4. Train the predictive maintenance model:

```bash
python vsm_ml_engine.py

```

*(Note: Ensure your `vsm_ml_engine.py` is configured for the AI4I dataset, or create a separate `ai4i_ml_ai4i2020.py` script).*
5. Launch the continuous health monitoring dashboard:

```bash
python -m streamlit run app_ai4i2020.py

```