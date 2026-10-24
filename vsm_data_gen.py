import pandas as pd
import numpy as np

np.random.seed(42)
num_orders = 5000

# 1. Order & Pre-Treatment (Batch Process)
order_quantity = np.random.randint(1000, 5000, size=num_orders)
pre_treat_batch_size = np.random.randint(500, 1000, size=num_orders) 
num_batches = np.ceil(order_quantity / pre_treat_batch_size)

loading_time_per_batch = np.random.uniform(20, 40, size=num_orders)
wash_time_per_batch = 20
actual_time_pretreatment = num_batches * (loading_time_per_batch + wash_time_per_batch)

# 2. Overhead Conveyor Logic (Hanger Capacity)
# 0 = Small Parts, 1 = Large Parts
is_large_part = np.random.choice([0, 1], size=num_orders)

# Small parts: 35-50 per hanger. Large parts: 12-20 per hanger.
parts_per_hanger = np.where(is_large_part == 0, 
                            np.random.randint(35, 51, size=num_orders),
                            np.random.randint(12, 21, size=num_orders))

# Calculate how many total hangers are needed for the order
num_hangers = np.ceil(order_quantity / parts_per_hanger)

# 3. Coating (Hanger Flow)
color_change = np.random.choice([0, 1], size=num_orders, p=[0.8, 0.2])
booth_swap_available = np.random.choice([0, 1], size=num_orders, p=[0.5, 0.5])

setup_delay_mins = np.zeros(num_orders)
for i in range(num_orders):
    if color_change[i] == 1:
        if booth_swap_available[i] == 1:
            setup_delay_mins[i] = 10
        else:
            setup_delay_mins[i] = np.random.uniform(35, 45)

ct_coating_per_hanger = np.random.uniform(1.5, 2.5, size=num_orders)
uptime_coating = np.random.uniform(0.85, 0.98, size=num_orders)
actual_time_coating = ((ct_coating_per_hanger * num_hangers) / uptime_coating) + setup_delay_mins

# 4. Curing (Hanger Flow)
ct_curing_per_hanger = np.random.uniform(2.0, 3.0, size=num_orders)
uptime_curing = np.random.uniform(0.90, 0.99, size=num_orders)
actual_time_curing = (ct_curing_per_hanger * num_hangers) / uptime_curing

# 5. Inspection (Belt Conveyor - Size Dependent Rate)
# Large parts: 40-50 parts/min. Small parts: 70-90 parts/min.
inspection_rate_per_min = np.where(is_large_part == 1,
                                   np.random.randint(40, 51, size=num_orders),
                                   np.random.randint(70, 91, size=num_orders))

actual_time_inspection = order_quantity / inspection_rate_per_min

# 6. Total Lead Time
total_lead_time = actual_time_pretreatment + actual_time_coating + actual_time_curing + actual_time_inspection

# 7. Identify Bottleneck
stations_actual_times = np.vstack((actual_time_pretreatment, actual_time_coating, actual_time_curing, actual_time_inspection))
bottleneck_station = np.argmax(stations_actual_times, axis=0) + 1 

# Compile DataFrame
df_vsm = pd.DataFrame({
    'Order_Quantity': order_quantity,
    'Parts_Per_Hanger': parts_per_hanger,
    'Color_Change': color_change,
    'Booth_Swap_Available': booth_swap_available,
    'Uptime_Coating': uptime_coating.round(3),
    'Uptime_Curing': uptime_curing.round(3),
    'Total_Lead_Time_Mins': total_lead_time.round(1),
    'Bottleneck_Station': bottleneck_station
})

df_vsm.to_csv('predictive_vsm_data.csv', index=False)
print("Updated hanger-capacity and inspection data generated. Saved as 'predictive_vsm_data.csv'")