# Predictive Value Stream Mapping (VSM) using Machine Learning

## Project Overview

This project bridges traditional Lean Manufacturing with Industrie 4.0 principles by upgrading the static Value Stream Mapping (VSM) methodology into a dynamic, predictive algorithm. Built as a digital twin for a continuous flow surface treatment and powder coating line, the machine learning model predicts total manufacturing lead times and dynamically identifies shifting bottlenecks based on physical daily production variables.

## Technical Architecture

* **Data Simulation:** Python (NumPy, Pandas) simulates shift data, modeling physical realities like batch-loading pretreatment, overhead conveyor hanger capacities, and dual-booth protocols.
* **Predictive ML Engine:** Scikit-Learn dual-model architecture utilizing a Random Forest Regressor to forecast lead time and a Random Forest Classifier to predict the bottleneck workstation.
* **Digital Twin Dashboard:** A local Streamlit web application providing an interactive UI for production planners to visualize the predicted Value Stream flow.

## Key Lean Manufacturing Insights

Through feature importance extraction, the Random Forest model mathematically validated that **geometric hanger capacity** and **total volume** are the highest drivers of lead time variance. The algorithm demonstrated that optimizing physical part-loading configurations significantly supersedes efforts to reduce micro-stoppages or color-change setup delays in dictating overall throughput.

## Local Installation & Execution

1. Clone the repository.
2. Install dependencies:
```bash
pip install pandas numpy scikit-learn streamlit

```


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