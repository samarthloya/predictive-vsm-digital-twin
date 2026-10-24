import streamlit as st
import pandas as pd
import joblib

# 1. Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('pdm_model.pkl')

model = load_model()

# 2. Dashboard UI
st.set_page_config(page_title="Industrie 4.0 VSM Monitor", layout="wide")
st.title("⚙️ Predictive Value Stream Monitor")
st.markdown("Real-time CNC sensor tracking for bottleneck prevention.")

# 3. Sensor Inputs (Sidebar)
st.sidebar.header("Live Sensor Feeds")
air_temp = st.sidebar.slider("Air Temperature [K]", 295.0, 305.0, 298.0, 0.1)
process_temp = st.sidebar.slider("Process Temperature [K]", 305.0, 315.0, 310.0, 0.1)
rpm = st.sidebar.slider("Rotational Speed [rpm]", 1100, 2900, 1500)
torque = st.sidebar.slider("Torque [Nm]", 10.0, 80.0, 40.0, 0.1)
tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 250, 50)

# 4. Run Inference: Use Probability for Continuous Degradation
input_data = pd.DataFrame([[air_temp, process_temp, rpm, torque, tool_wear]], 
                          columns=['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])

# predict_proba returns an array: [Probability of Class 0, Probability of Class 1]
# We want the probability of Class 1 (Failure)
failure_probability = model.predict_proba(input_data)[0][1]

# Convert the failure probability into a continuous Uptime/Health Score
predicted_uptime_percent = (1.0 - failure_probability) * 100

# 5. Display KPI Dashboard with 3-Tier Status
st.subheader("Station Status: CNC Milling Center")

if predicted_uptime_percent >= 85.0:
    st.success("✅ **STATUS: NORMAL**")
    st.write("Machine is operating within tolerance. VSM flow remains stable.")
    st.metric("Predicted Uptime", f"{predicted_uptime_percent:.1f}%")

elif 50.0 <= predicted_uptime_percent < 85.0:
    st.warning("⚠️ **STATUS: DEGRADED (Middle Ground)**")
    st.write("Sensor anomalies detected. Machine is wearing down. Micro-stoppages are likely, which will degrade Takt time.")
    st.metric("Predicted Uptime", f"{predicted_uptime_percent:.1f}%")

else:
    st.error("🚨 **CRITICAL: FAILURE IMMINENT**")
    st.write("Severe sensor deviations. Machine failure predicted. This station is now the primary bottleneck.")
    st.metric("Predicted Uptime", f"{predicted_uptime_percent:.1f}% (Action Required)")

st.divider()
st.markdown("### VSM Impact Analysis")
st.write("By tracking the continuous probability of failure (Machine Health Score), production planners can schedule maintenance during the **Degraded** phase, preventing a complete drop to 0% uptime and preserving the factory's overall throughput.")