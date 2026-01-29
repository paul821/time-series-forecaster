import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from forecasting import ForecastingModels

st.set_page_config(page_title="Forecasting App", layout="wide")
st.title("Time Series Forecasting App")

# --------------------------
# Session State Initialization
# --------------------------
if "data" not in st.session_state:
    # Default initial data (e.g., from PDF Example 3.2 or random)
    # Let's use Example 3.2 data (Wheelchairs) as default
    default_data = [57, 254, 335, 307, 347, 450, 434, 479, 621, 597, 641, 793]
    st.session_state.data = pd.DataFrame({"Demand": default_data})

# --------------------------
# Sidebar - Configuration
# --------------------------
st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox(
    "Choose Forecasting Method",
    ["Moving Average", "Simple Exponential Smoothing", "Holt's Linear Trend", "Winter's Multiplicative"]
)

st.sidebar.header("Parameters")

# Variable containers
params = {}
optimize = False
metric_to_optimize = "MSE"

if model_type == "Moving Average":
    window_size = st.sidebar.number_input("Window Size (N)", min_value=1, value=3)
    params["window_size"] = window_size
    
elif model_type in ["Simple Exponential Smoothing", "Holt's Linear Trend", "Winter's Multiplicative"]:
    optimization_mode = st.sidebar.radio("Parameter Mode", ["Manual", "Optimize"])
    
    if optimization_mode == "Optimize":
        optimize = True
        metric_to_optimize = st.sidebar.selectbox("Metric to Minimize", ["MSE", "MAD", "MAPE", "BIAS"])
    
    if model_type == "Simple Exponential Smoothing":
        if not optimize:
            alpha = st.sidebar.slider("Alpha (Level)", 0.0, 1.0, 0.2)
            params["alpha"] = alpha
            
    elif model_type == "Holt's Linear Trend":
        if not optimize:
            alpha = st.sidebar.slider("Alpha (Level)", 0.0, 1.0, 0.2)
            beta = st.sidebar.slider("Beta (Trend)", 0.0, 1.0, 0.1)
            params["alpha"] = alpha
            params["beta"] = beta
            
    elif model_type == "Winter's Multiplicative":
        season_length = st.sidebar.number_input("Periods per Season (N)", min_value=2, value=4)
        params["season_length"] = season_length
        
        if not optimize:
            alpha = st.sidebar.slider("Alpha (Level)", 0.0, 1.0, 0.2)
            beta = st.sidebar.slider("Beta (Trend)", 0.0, 1.0, 0.1)
            gamma = st.sidebar.slider("Gamma (Seasonal)", 0.0, 1.0, 0.1)
            params["alpha"] = alpha
            params["beta"] = beta
            params["gamma"] = gamma

forecast_horizon = st.sidebar.number_input("Forecast Horizon", min_value=1, value=4)

st.sidebar.markdown("### Uncertainty")
confidence_interval_pct = st.sidebar.number_input("Prediction Interval (%)", min_value=0.1, max_value=99.9, value=95.0, step=0.1)
confidence_level = confidence_interval_pct / 100.0

# --------------------------
# Main Panel - Data Input
# --------------------------
st.subheader("Data Input")

col1, col2 = st.columns(2)
with col1:
    if st.button("Add Row"):
        # Add a new row with 0 or last value
        new_index = len(st.session_state.data)
        st.session_state.data.loc[new_index] = {"Demand": 0}
        
with col2:
    if st.button("Delete Last Row"):
        if len(st.session_state.data) > 0:
            st.session_state.data = st.session_state.data.iloc[:-1]

edited_df = st.data_editor(st.session_state.data, num_rows="dynamic", key="data_editor")
# Sync editor changes back to session state if they happen
if not edited_df.equals(st.session_state.data):
    st.session_state.data = edited_df

data_values = st.session_state.data["Demand"].dropna().values

if len(data_values) < 2:
    st.warning("Please enter at least 2 data points.")
    st.stop()

# --------------------------
# Forecasting Logic
# --------------------------
fm = ForecastingModels()
results = None
opt_params = []

if optimize:
    with st.spinner("Optimizing parameters..."):
        method_map = {
            "Simple Exponential Smoothing": "SES",
            "Holt's Linear Trend": "Holt",
            "Winter's Multiplicative": "Winter"
        }
        short_name = method_map[model_type]
        season_len = params.get("season_length", None)
        
        opt_vals = fm.optimize_params(short_name, data_values, metric=metric_to_optimize, season_length=season_len)
        
        if opt_vals is not None:
            st.sidebar.success(f"Optimized Parameters: {np.round(opt_vals, 3)}")
            if short_name == "SES":
                params["alpha"] = opt_vals[0]
            elif short_name == "Holt":
                params["alpha"] = opt_vals[0]
                params["beta"] = opt_vals[1]
            elif short_name == "Winter":
                params["alpha"] = opt_vals[0]
                params["beta"] = opt_vals[1]
                params["gamma"] = opt_vals[2]

# Run Algo
if model_type == "Moving Average":
    results = fm.moving_average(data_values, params["window_size"], forecast_horizon, confidence_level=confidence_level)
elif model_type == "Simple Exponential Smoothing":
    results = fm.ses(data_values, params["alpha"], forecast_horizon, confidence_level=confidence_level)
elif model_type == "Holt's Linear Trend":
    results = fm.holts(data_values, params["alpha"], params["beta"], forecast_horizon, confidence_level=confidence_level)
elif model_type == "Winter's Multiplicative":
    if len(data_values) < 2 * params["season_length"]:
        st.error(f"Need at least {2*params['season_length']} data points for Winter's initialization.")
        st.stop()
    results = fm.winters(data_values, params["alpha"], params["beta"], params["gamma"], params["season_length"], forecast_horizon, confidence_level=confidence_level)
    if "error" in results:
        st.error(results["error"])
        st.stop()

# --------------------------
# Display Results
# --------------------------

# Metrics
st.subheader("Model Performance (Training metrics)")
metrics = results["metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("MSE", f"{metrics['MSE']:.2f}")
c2.metric("MAD", f"{metrics['MAD']:.2f}")
c3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
c4.metric("BIAS", f"{metrics['BIAS']:.2f}")

# Forecast Table
st.subheader("Detailed Results")

n = len(data_values)
fitted = results["fitted"]
level = results.get("level", np.full(n, np.nan))
trend = results.get("trend", np.full(n, np.nan))
seasonals = results.get("seasonals", np.full(n, np.nan))

df_res = pd.DataFrame({
    "Period": range(1, n + 1),
    "Actual": data_values,
    "Fitted": fitted,
    "Error": fitted - data_values,
    "Level (L)": level,
    "Trend (G)": trend,
    "Seasonal (C)": seasonals
})

st.dataframe(df_res)

# Future Forecast Table
st.subheader("Future Forecasts & Prediction Intervals")
future_df = pd.DataFrame({
    "Period": range(n + 1, n + forecast_horizon + 1),
    "Forecast": results["forecast"],
    f"Lower PI ({confidence_interval_pct}%)": results["lower"],
    f"Upper PI ({confidence_interval_pct}%)": results["upper"]
})
st.dataframe(future_df)

# Visualization
st.subheader("Forecast Plot")

fig = go.Figure()

# Actual
fig.add_trace(go.Scatter(x=list(range(1, n+1)), y=data_values, mode='lines+markers', name='Actual', line=dict(color='blue')))

# Fitted
fig.add_trace(go.Scatter(x=list(range(1, n+1)), y=fitted, mode='lines', name='Fitted', line=dict(color='orange', dash='dash')))

# Forecast
future_x = list(range(n+1, n+forecast_horizon+1))
fig.add_trace(go.Scatter(x=future_x, y=results["forecast"], mode='lines+markers', name='Forecast', line=dict(color='green')))

# Prediction Intervals
# Combine historical fitted (no PI?) + Forecast PI
# We only plot PI for forecast
fig.add_trace(go.Scatter(
    x=future_x, 
    y=results["upper"], 
    mode='lines', 
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=future_x, 
    y=results["lower"], 
    mode='lines', 
    line=dict(width=0), 
    fill='tonexty', 
    fillcolor='rgba(0, 255, 0, 0.2)',
    name=f'{confidence_interval_pct}% Prediction Interval'
))

fig.update_layout(
    title="Demand vs Forecast",
    xaxis_title="Period",
    yaxis_title="Demand",
    hovermode="x unified"
)

st.plotly_chart(fig)

st.write("### Methodology Notes")
if model_type == "Winter's Multiplicative":
    st.write(f"Initialized using assumed season length N={params['season_length']}. Requires at least 2 seasons.")
st.write(f"Prediction Intervals calculated with {confidence_interval_pct}% confidence level.")
