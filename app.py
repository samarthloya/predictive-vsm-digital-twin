import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 1. Configure the Dashboard
st.set_page_config(page_title="Predictive VSM Dashboard", layout="wide")
st.title("🏭 Predictive Value Stream Mapping (VSM)")
st.markdown("Dynamic Digital Twin for Continuous Flow Powder Coating")

# 2. Train & Cache Models
@st.cache_resource
def load_models():
    df = pd.read_csv('predictive_vsm_data.csv')
    X = df[['Order_Quantity', 'Parts_Per_Hanger', 'Color_Change', 'Booth_Swap_Available', 'Uptime_Coating', 'Uptime_Curing']]
    
    # Regression for Lead Time (Minutes)
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X, df['Total_Lead_Time_Mins'])
    
    # Classification for Bottleneck
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X, df['Bottleneck_Station'])
    
    return regressor, classifier

reg_model, clf_model = load_models()

# 3. Sidebar Configuration (User Inputs)
st.sidebar.header("Shift Parameters")
order_qty = st.sidebar.slider("Order Quantity", 1000, 5000, 2500, step=100)
parts_per_hanger = st.sidebar.slider("Parts per Hanger (Size Dependent)", 12, 50, 30, step=1)

st.sidebar.divider()
st.sidebar.subheader("Coating Setup")
color_change_input = st.sidebar.selectbox("Color Change Required?", ["No", "Yes"])
color_val = 1 if color_change_input == "Yes" else 0

if color_val == 1:
    booth_swap_input = st.sidebar.selectbox("2nd Booth Available for Swap?", ["Yes", "No"])
    swap_val = 1 if booth_swap_input == "Yes" else 0
    if swap_val == 1:
        st.sidebar.caption("Estimated setup delay: 10 minutes")
    else:
        st.sidebar.caption("Estimated setup delay: 35-45 minutes (Deep clean)")
else:
    swap_val = 0 

st.sidebar.divider()
st.sidebar.subheader("Equipment Availability")
uptime_coating = st.sidebar.slider("Coating Uptime (%)", 0.85, 0.98, 0.95, 0.01)
uptime_curing = st.sidebar.slider("Curing Uptime (%)", 0.90, 0.99, 0.98, 0.01)

# 4. Run ML Inference
input_data = pd.DataFrame([[order_qty, parts_per_hanger, color_val, swap_val, uptime_coating, uptime_curing]], 
                          columns=['Order_Quantity', 'Parts_Per_Hanger', 'Color_Change', 'Booth_Swap_Available', 'Uptime_Coating', 'Uptime_Curing'])

pred_lead_time_mins = reg_model.predict(input_data)[0]
pred_bottleneck = clf_model.predict(input_data)[0]

stations = {1: "Pre-Treatment", 2: "Powder Coating", 3: "Curing Oven", 4: "Inspection"}
bottleneck_name = stations[pred_bottleneck]

# 5. Display KPI Dashboard
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Total Lead Time", f"{pred_lead_time_mins / 60:.1f} Hours")
col2.metric("Predicted Bottleneck", bottleneck_name)
col3.metric("Production Volume", f"{order_qty} Units")

st.divider()

# 6. Dynamic VSM Flow Visualizer
st.subheader("Predicted Value Stream Flow")
flow_cols = st.columns(4)
for i, col in enumerate(flow_cols, start=1):
    with col:
        if i == pred_bottleneck:
            st.error(f"**Stage {i}:\n{stations[i]}**\n\n⚠️ BOTTLENECK")
        else:
            st.success(f"**Stage {i}:\n{stations[i]}**\n\n✅ Flow Stable")