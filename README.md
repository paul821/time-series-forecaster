# Streamlit Forecasting App

A time series forecasting application built with Streamlit, implementing standard methods from Operations Engineering:
- Moving Average
- Simple Exponential Smoothing (SES)
- Holt's Linear Trend
- Winter's Multiplicative Method

## Features
- **Interactive Data Input**: Manually edit data or paste from spreadsheets.
- **Optimization**: Automatically find optimal smoothing parameters ($\alpha, \beta, \gamma$) to minimize MSE.
- **Configurable Prediction Intervals**: Set your desired confidence level (e.g., 90%, 95%, 99%).
- **Visualizations**: Interactive Plotly charts showing historical data, fitted values, and forecasts with uncertainty bounds.

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment
To deploy on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push this repository to GitHub.
2. Connect your GitHub account to Streamlit Cloud.
3. Select this repository and the `app.py` file.
4. Click **Deploy**.
