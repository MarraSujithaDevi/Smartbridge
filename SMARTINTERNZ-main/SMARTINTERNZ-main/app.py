import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# Clean & preprocess
df.dropna(inplace=True)
df['date_time'] = pd.to_datetime(df['date_time'])
df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hour'] = df['date_time'].dt.hour
df['minute'] = df['date_time'].dt.minute
df['second'] = df['date_time'].dt.second
df['holiday'] = df['holiday'].astype('category').cat.codes
df['weather_main'] = df['weather_main'].astype('category').cat.codes

# Features
features = ['holiday', 'temp', 'rain_1h', 'snow_1h', 'weather_main',
            'year', 'month', 'day', 'hour', 'minute', 'second']
X = df[features]
y = df['traffic_volume']

# Scale + train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                    test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# ========== Streamlit App ==========
st.set_page_config(page_title="Traffic Volume Estimation", layout="centered")
st.markdown("<h1 style='text-align: center;'>🚦 Traffic Volume Estimation</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# UI layout
col1, col2 = st.columns(2)

with col1:
    holiday = st.selectbox("🗓️ Holiday", ["None", "Holiday"])
    temp = st.slider("🌡️ Temperature (°C)", -10.0, 50.0, 20.0)
    rain = st.slider("🌧️ Rain (mm)", 0.0, 50.0, 0.0)
    snow = st.slider("❄️ Snow (mm)", 0.0, 50.0, 0.0)

with col2:
    weather = st.selectbox("☁️ Weather", ["Clear", "Clouds", "Rain", "Snow"])
    year = st.number_input("Year", min_value=2010, max_value=2030, value=2025)
    month = st.number_input("Month", 1, 12, 6)
    day = st.number_input("Day", 1, 31, 20)
    hour = st.slider("Hour", 0, 23, 12)
    minute = st.slider("Minute", 0, 59, 0)
    second = st.slider("Second", 0, 59, 0)

# Map values
holiday_val = 1 if holiday == "Holiday" else 0
weather_map = {"Clear": 0, "Clouds": 1, "Rain": 2, "Snow": 3}
weather_val = weather_map[weather]

# Predict button
if st.button("📊 Predict Traffic Volume"):
    input_data = [[
        holiday_val, temp, rain, snow, weather_val,
        year, month, day, hour, minute, second
    ]]
    input_scaled = scaler.transform(input_data)
    prediction = int(model.predict(input_scaled)[0])

    st.success(f"🚗 Predicted Traffic Volume: **{prediction:,} vehicles**")

    # Hourly simulation
    hourly_traffic = []
    for h in range(24):
        sim_data = [[holiday_val, temp, rain, snow, weather_val,
                     year, month, day, h, minute, second]]
        sim_scaled = scaler.transform(sim_data)
        pred = model.predict(sim_scaled)[0]
        hourly_traffic.append(pred)

    # Show chart
    st.markdown("### 📈 Simulated Traffic Volume Over 24 Hours")
    fig, ax = plt.subplots()
    ax.plot(range(24), hourly_traffic, color="orange", marker='o')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Traffic Volume")
    ax.set_title("Traffic Simulation for 24 Hours")
    st.pyplot(fig)

    # Additional message
    peak_hour = np.argmax(hourly_traffic)
    st.info(f"🚨 **Peak Hour Traffic** is expected at **{peak_hour}:00 hrs** with ~**{int(hourly_traffic[peak_hour])}** vehicles.")
