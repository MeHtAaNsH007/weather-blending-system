import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="Weather Forecast Blending System", layout="wide")

st.title("Regional Multi-Model Weather Forecast Blending System")

city_coords = {
    "New Delhi, India": (28.6139, 77.2090),
    "Mumbai, India": (19.0760, 72.8777),
    "Phagwara / Jalandhar, India": (31.3260, 75.5762),
    "London, UK": (51.5074, -0.1278),
    "New York, USA": (40.7128, -74.0060)
}

st.sidebar.header("Location Settings")
selected_city = st.sidebar.selectbox("Select City", list(city_coords.keys()))
lat, lon = city_coords[selected_city]

st.subheader(f"Current Forecast for {selected_city}")

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": lat,
    "longitude": lon,
    "hourly": "temperature_2m",
    "models": "gfs_seamless,ecmwf_ifs025,icon_seamless",
    "timezone": "auto"
}

response = requests.get(url, params=params)
data = response.json()

hourly = data.get("hourly", {})
df = pd.DataFrame({
    "Time": hourly.get("time", []),
    "GFS Model (°C)": hourly.get("temperature_2m_gfs_seamless", []),
    "ECMWF Model (°C)": hourly.get("temperature_2m_ecmwf_ifs025", []),
    "ICON Model (°C)": hourly.get("temperature_2m_icon_seamless", [])
})

df["Time"] = pd.to_datetime(df["Time"])

for col in ["GFS Model (°C)", "ECMWF Model (°C)", "ICON Model (°C)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["ECMWF Model (°C)"] = df["ECMWF Model (°C)"].fillna((df["GFS Model (°C)"] + df["ICON Model (°C)"]) / 2)
df = df.bfill().ffill()

X = df[["GFS Model (°C)", "ECMWF Model (°C)", "ICON Model (°C)"]]
weights = np.array([0.40, 0.35, 0.25])
df["AI Blended Forecast (°C)"] = X.dot(weights).round(2)

st.write("### Forecast Comparison (Full 7-Day View)")
st.line_chart(df.set_index("Time")[["GFS Model (°C)", "ECMWF Model (°C)", "ICON Model (°C)", "AI Blended Forecast (°C)"]])

st.write("### Detailed Hourly Forecast Data")
st.dataframe(df)
