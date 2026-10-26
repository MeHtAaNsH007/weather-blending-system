# Regional Multi-Model Weather Forecast Blending and Calibration System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://weather-blending-system-a7br9net8eufv49icdx8o2.streamlit.app/)

An AI-assisted meteorological forecasting platform that ingests predictions from multiple global numerical weather prediction (NWP) models, performs dynamic inverse-error calibration against recent ground-truth data, and presents an interpretable blended forecast dashboard.

**Problem Statement:** SIH26081  
**Core Methodology:** Supervised Regression, Dynamic Inverse-MAE Model Calibration, Explainable AI (XAI), and Decision Support.

---

## 🛠️ System Architecture & Workflow

1. **Multi-Model Data Ingestion:**
   * **GFS (USA - NOAA):** Global Forecast System (Seamless feed)
   * **ECMWF (Europe):** Integrated Forecasting System (IFS 0.25° grid)
   * **ICON (Germany - DWD):** Icosahedral Nonhydrostatic Model (Seamless feed)
   * **Ground Truth:** Real-time observed 2m surface temperature (`past_days=3`)

2. **Real-Time Dynamic Inverse-MAE Calibration:**
   Instead of using static hardcoded model weights, the system computes the 72-hour historical Mean Absolute Error ($\text{MAE}$) for each agency model relative to actual ground observations for the requested location:

   $$\text{MAE}_i = \frac{1}{72} \sum_{t=1}^{72} \vert{}T_{\text{model}_i}(t) - T_{\text{actual}}(t)\vert{}$$

   Dynamic model weights ($w_i$) are assigned using the **Inverse-Variance Ratio**:

   $$w_i = \frac{\frac{1}{\text{MAE}_i + \epsilon}}{\sum_{j=1}^{n} \frac{1}{\text{MAE}_j + \epsilon}}$$

   *The model with the lowest recent forecast error automatically receives the highest weight for the operational forecast.*

3. **Baseline Comparison & Uncertainty Quantification:**
   * **Baseline Model:** Computes an unweighted Simple Average across all agencies to measure forecast improvement.
   * **Model Spread Analysis:** Tracks $\max(T) - \min(T)$ across input models.
   * **Confidence Calibration Alert:** Triggers an automated operational review alert if average model spread exceeds $2.5^\circ\text{C}$.

---

## 📊 Features

* **Real-Time Live Dashboard:** Interactive line chart displaying individual agency predictions, Simple Average baseline, and AI Blended forecast over a 7-day window.
* **Explainable AI Calibration Matrix:** Tabbed transparent breakdown showing exact 72-hour MAE metrics and real-time computed percentage weights for GFS, ECMWF, and ICON.
* **Operational Decision Support:** Automatic banner indicating forecast confidence (High Confidence vs Low Confidence requiring meteorologist review).
* **Data Export:** Clean CSV export for combined historical actuals and future predictions.

---

## 🚀 Live Demo

Access the deployed application here:  
👉 [Weather Blending & Calibration System Live App](https://weather-blending-system-a7br9net8eufv49icdx8o2.streamlit.app/)

---

## ⚙️ Local Setup & Running Instructions

### Prerequisites
* Python 3.9+ installed
* Git

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/weather-blending-system.git
   cd weather-blending-system
