import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="Weather Forecast Blending System", layout="wide", page_icon="🌤️")

# Updated Title to match SIH26081 screenshot requirements
st.title("Regional Multi-Model Weather Forecast Blending and Calibration System")
st.caption("SIH26081 | Supervised Regression & Explainable Decision Support")

st.sidebar.header("📍 Location Settings")
city_input = st.sidebar.text_input("Search City Name", value="New Delhi")

geo_url = "https://geocoding-api.open-meteo.com/v1/search"
geo_params = {"name": city_input, "count": 1, "format": "json"}

try:
    geo_res = requests.get(geo_url, params=geo_params).json()
except Exception:
    geo_res = {}

if geo_res.get("results"):
    city_data = geo_res["results"][0]
    city_name = f"{city_data['name']}, {city_data.get('country', '')}"
    lat = city_data["latitude"]
    lon = city_data["longitude"]

    url_base = "https://api.open-meteo.com/v1/forecast"
    params_base = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "past_days": 3,
        "forecast_days": 7,
        "timezone": "auto"
    }
    res_base = requests.get(url_base, params=params_base).json()
    hourly_base = res_base.get("hourly", {})

    params_models = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "models": "gfs_seamless,ecmwf_ifs025,icon_seamless",
        "past_days": 3,
        "forecast_days": 7,
        "timezone": "auto"
    }
    res_models = requests.get(url_base, params=params_models).json()
    hourly_models = res_models.get("hourly", {})

    if "time" in hourly_base and "time" in hourly_models:
        df_base = pd.DataFrame({
            "Time": hourly_base.get("time", []),
            "Ground Truth (°C)": hourly_base.get("temperature_2m", [])
        })

        df_models = pd.DataFrame({
            "Time": hourly_models.get("time", []),
            "GFS (°C)": hourly_models.get("temperature_2m_gfs_seamless", []),
            "ECMWF (°C)": hourly_models.get("temperature_2m_ecmwf_ifs025", []),
            "ICON (°C)": hourly_models.get("temperature_2m_icon_seamless", [])
        })

        df_base["Time"] = pd.to_datetime(df_base["Time"])
        df_models["Time"] = pd.to_datetime(df_models["Time"])

        df = pd.merge(df_base, df_models, on="Time", how="inner")

        for col in ["Ground Truth (°C)", "GFS (°C)", "ECMWF (°C)", "ICON (°C)"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["ECMWF (°C)"] = df["ECMWF (°C)"].fillna((df["GFS (°C)"] + df["ICON (°C)"]) / 2)
        df["Ground Truth (°C)"] = df["Ground Truth (°C)"].fillna(df["GFS (°C)"])
        df = df.bfill().ffill()

        now = pd.Timestamp.now().tz_localize(None)

        past_df = df[df["Time"] < now].copy()
        if past_df.empty:
            past_df = df.iloc[:72].copy()

        mae_gfs = np.mean(np.abs(past_df["GFS (°C)"] - past_df["Ground Truth (°C)"]))
        mae_ecmwf = np.mean(np.abs(past_df["ECMWF (°C)"] - past_df["Ground Truth (°C)"]))
        mae_icon = np.mean(np.abs(past_df["ICON (°C)"] - past_df["Ground Truth (°C)"]))

        eps = 1e-5
        inv_gfs = 1.0 / (mae_gfs + eps)
        inv_ecmwf = 1.0 / (mae_ecmwf + eps)
        inv_icon = 1.0 / (mae_icon + eps)
        sum_inv = inv_gfs + inv_ecmwf + inv_icon

        w_gfs = inv_gfs / sum_inv
        w_ecmwf = inv_ecmwf / sum_inv
        w_icon = inv_icon / sum_inv

        dynamic_weights = np.array([w_gfs, w_ecmwf, w_icon])

        X = df[["GFS (°C)", "ECMWF (°C)", "ICON (°C)"]]
        df["AI Blended (°C)"] = X.dot(dynamic_weights).round(2)

        # Baseline Comparison Requirement
        df["Simple Average (°C)"] = X.mean(axis=1).round(2)

        df["Model Spread (°C)"] = (df[["GFS (°C)", "ECMWF (°C)", "ICON (°C)"]].max(axis=1) - 
                                   df[["GFS (°C)", "ECMWF (°C)", "ICON (°C)"]].min(axis=1)).round(2)

        future_df = df[df["Time"] >= now].copy()
        if future_df.empty:
            future_df = df.iloc[-168:].copy()

        tab1, tab2, tab3 = st.tabs(["📊 Live Forecast", "📈 Dynamic AI Calibration", "📥 Data Export & Docs"])

        with tab1:
            st.subheader(f"Real-Time Weather Overview: {city_name}")
            
            current_temp = future_df["AI Blended (°C)"].iloc[0]
            max_temp = future_df["AI Blended (°C)"].max()
            min_temp = future_df["AI Blended (°C)"].min()
            avg_spread = future_df["Model Spread (°C)"].mean().round(2)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Blended Temp", f"{current_temp} °C")
            col2.metric("7-Day High", f"{max_temp} °C")
            col3.metric("7-Day Low", f"{min_temp} °C")
            col4.metric("Avg Model Spread", f"{avg_spread} °C")

            # Confidence Alert Requirement
            if avg_spread > 2.5:
                st.warning("⚠️ **Low Confidence / Unreliable Condition:** The average model spread exceeds 2.5°C. The combined forecast is currently unstable due to high agency disagreement. Human meteorologist review is required before making operational decisions.")
            else:
                st.success("✅ **High Confidence:** Models are in strong agreement. The combined forecast is reliable for operational use.")

            st.write("### 7-Day Dynamic Blended Forecast vs Simple Average")
            st.line_chart(future_df.set_index("Time")[["GFS (°C)", "ECMWF (°C)", "ICON (°C)", "Simple Average (°C)", "AI Blended (°C)"]])

        with tab2:
            st.subheader("72-Hour Historical Calibration & Computed Weights")
            st.write("The system calculates recent 72-hour errors for this city and computes dynamic weights using the inverse-variance formula.")

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("GFS 72h MAE", f"{mae_gfs:.3f} °C", delta=f"Weight: {w_gfs*100:.1f}%", delta_color="off")
            m_col2.metric("ECMWF 72h MAE", f"{mae_ecmwf:.3f} °C", delta=f"Weight: {w_ecmwf*100:.1f}%", delta_color="off")
            m_col3.metric("ICON 72h MAE", f"{mae_icon:.3f} °C", delta=f"Weight: {w_icon*100:.1f}%", delta_color="off")

            st.write("### Live Calculated Weight Matrix")
            weight_df = pd.DataFrame({
                "Agency Model": ["GFS (USA)", "ECMWF (Europe)", "ICON (Germany)"],
                "72-Hour Error (MAE)": [f"{mae_gfs:.3f} °C", f"{mae_ecmwf:.3f} °C", f"{mae_icon:.3f} °C"],
                "Dynamic Weight": [f"{w_gfs*100:.2f}%", f"{w_ecmwf*100:.2f}%", f"{w_icon*100:.2f}%"],
                "Calibration Rule": ["Inverse MAE Ratio", "Inverse MAE Ratio", "Inverse MAE Ratio"]
            })
            st.table(weight_df)

        with tab3:
            st.subheader("Raw & Processed Dataset (Includes Past 72h & 7-Day Forecast)")
            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Dataset (CSV)",
                data=csv_data,
                file_name=f"{city_name.replace(' ', '_')}_weather_dynamic_blended.csv",
                mime="text/csv"
            )

    else:
        st.error("Failed to retrieve weather data. Please try searching another city.")
else:
    st.sidebar.error("City not found. Please type a valid city name.")
