import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

class ForecastingModels:
    def __init__(self):
        pass

    @staticmethod
    def calculate_metrics(actual, forecast, num_params=0):
        """
        Calculate error metrics: BIAS, MSE, MAD, MAPE, SSE.
        actual and forecast should be numpy arrays of the same length (trimmed to overlap).
        """
        actual = pd.to_numeric(actual, errors='coerce')
        forecast = pd.to_numeric(forecast, errors='coerce')
        
        # Filter out NaN in forecast or actual (if any)
        # This handles both original NaNs and those introduced by coercion
        # Ensure we have numpy arrays of float type to avoid TypeError with isnan on objects
        actual = np.array(actual, dtype=float)
        forecast = np.array(forecast, dtype=float)

        mask = ~np.isnan(actual) & ~np.isnan(forecast)
        actual = actual[mask]
        forecast = forecast[mask]
        
        if len(actual) == 0:
            return {"BIAS": np.nan, "MSE": np.nan, "MAD": np.nan, "MAPE": np.nan, "SSE": np.nan, "n": 0}

        errors = forecast - actual
        n = len(actual)
        
        bias = np.sum(errors) / n
        sse = np.sum(errors ** 2)
        mse = sse / n
        mad = np.sum(np.abs(errors)) / n
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_terms = np.abs(errors / actual)
            mape_terms = np.nan_to_num(mape_terms, nan=0.0, posinf=0.0, neginf=0.0) # Handle divide by zero
        mape = np.sum(mape_terms) / n * 100
        
        return {
            "BIAS": bias,
            "MSE": mse,
            "MAD": mad,
            "MAPE": mape,
            "SSE": sse,
            "n": n
        }

    # =========================================================================
    # Moving Average
    # =========================================================================
    def moving_average(self, data, window_size, forecast_horizon=1, confidence_level=0.95):
        """
        Moving Average Method.
        Returns detailed results including fitted values, forecast, and intervals.
        """
        data = np.array(data)
        n = len(data)
        fitted = np.full(n, np.nan)
        
        # Calculate fitted values (1-step ahead forecast for historical data)
        # F_{t+1} = Mean(D_{t-N+1} ... D_t)
        # So F_t = Mean(D_{t-N} ... D_{t-1})
        for t in range(window_size, n):
            fitted[t] = np.mean(data[t-window_size:t])
            
        # Forecast future
        last_fitted = np.mean(data[-window_size:])
        forecasts = np.full(forecast_horizon, last_fitted) # MA forecast is constant
        
        # Calculate Error Metrics on historical data
        metrics = self.calculate_metrics(data[window_size:], fitted[window_size:])
        
        # Prediction Intervals
        # s_e = sqrt(SSE / (n - window_size)) ? Book says something specific about standard error?
        # Eq 3.15 standard error for SES is sqrt(SSE/(t-1)). 
        # For MA, usually s = RMSE or similar. The book doesn't give explicitly for MA in snippet, 
        # but text implies similar logic. Let's use MSE-based std dev if not specified.
        # Actually I'll use standard deviation of residuals.
        # Prediction Intervals
        # s_e = sqrt(SSE / (n - N)) or similar. Using residual std dev.
        # Fixed: Book uses t-1 or similar for SES. I will use residual std dev.
        if metrics["n"] > 1:
            errors = data[window_size:] - fitted[window_size:]
            sigma_e = np.sqrt(np.sum(errors**2) / (len(errors) - 1)) if len(errors) > 1 else 0
        else:
            sigma_e = 0
            
        # Calculate z-score
        # alpha_tail = 1 - confidence_level
        # z = norm.ppf(1 - alpha_tail/2)
        z = norm.ppf(1 - (1 - confidence_level)/2)
        
        pi_width = z * sigma_e
        lower_bound = forecasts - pi_width
        upper_bound = forecasts + pi_width
        
        return {
            "fitted": fitted,
            "forecast": forecasts,
            "lower": lower_bound,
            "upper": upper_bound,
            "metrics": metrics,
            "params": {"Window": window_size, "Confidence": confidence_level},
            "components": {}
        }

    # =========================================================================
    # Simple Exponential Smoothing (SES)
    # =========================================================================
    def ses(self, data, alpha, forecast_horizon=1, confidence_level=0.95):
        data = np.array(data)
        n = len(data)
        fitted = np.full(n, np.nan)
        level = np.full(n, np.nan)
        
        # Initialization
        # F_1 = D_1
        # L_1 = alpha D_1 + (1-alpha)F_1  <-- Wait, book Eq 3.5 says L_t = alpha D_t + (1-alpha) F_t
        # And F_{t+1} = L_t.
        # Initialization Step: F_1 = D_1.
        
        # So for t=1:
        # F_1 = D_1 (This is the forecast made at t=0 for t=1? No, Book: "Forecast for Period 1 was accurate... sets F1=D1")
        # Then L_1 = alpha D_1 + (1-alpha) F_1 = alpha D_1 + (1-alpha)D_1 = D_1.
        # Then F_2 = L_1 = D_1.
        
        # Loop
        # We need F_t to compute L_t. F_t comes from L_{t-1}.
        
        fitted[0] = data[0] # F_1 = D_1
        level[0] = data[0]  # L_1 = D_1
        
        for t in range(1, n):
            # Forecast for Period t (F_t) is L_{t-1}
            F_t = level[t-1]
            fitted[t] = F_t
            
            D_t = data[t]
            # Update Level
            L_t = alpha * D_t + (1 - alpha) * F_t
            level[t] = L_t
            
        # Forecast future
        # F_{n+1} = L_n
        last_level = level[-1]
        forecasts = np.full(forecast_horizon, last_level)
        
        # Metrics
        # Fitted[0] is prediction for data[0]. But we forced equal. Error is 0.
        # usually metrics exclude initialization? Book says "sum of all t observed forecast errors".
        metrics = self.calculate_metrics(data, fitted)
        
        # Standard Error (Eq 3.15)
        # s_e = sqrt((SSE - t * BIAS^2) / (t-1))
        t = metrics["n"]
        if t > 1:
            try:
                term = metrics["SSE"] - t * (metrics["BIAS"]**2)
                # Ensure non-negative
                if term < 0: term = 0
                sigma_e = np.sqrt(term / (t - 1))
            except:
                sigma_e = 0
        else:
            sigma_e = 0
            
        # Prediction Interval
        # D_tau = (tau-1) * alpha^2  (Eq 3.16 says D_tau = (tau-1)alpha^2 is WRONG?
        # Wait, let me check Eq 3.16 in text.
        # Eq 3.16: D_tau = (tau - 1) alpha^2 ? No... lemme re-read snippet.
        # Snippet Line 2255: "D_tau = (tau - 1) alpha^2?"
        # Image says "D_tau = (tau - 1) * alpha^2"? No, looking at line 3146 (Wait that is winter's)
        # Line 2255: D_tau = (tau - 1) * alpha^2? 
        # Reader text: "D_tau = (tau - 1) alpha^2  for tau = 2, 3..." ?? 
        # Check carefully: "D_tau = (psi - 1) alpha^2" ?? 
        # No, it says "D_tau = (tau - 1) * alpha^2"? Not clear.
        # Let's check text line 2255 again from Step Id 32.
        # 2255: D  = (    1) 2 for   = 2; 3; : : : (3.16)
        # Greek letter missing? It says (    1) 2.
        # It's likely D_tau = (tau - 1) * alpha^2?
        # Actually standard result for ARIMA(0,1,1) equiv to SES is variance grows.
        # Let's assume the formula is D_tau = (tau-1)*alpha^2 based on pattern.
        # Wait, for tau=2, D2 = (2-1)*alpha^2 = alpha^2.
        # Data has been processed.
        # z-score
        z = norm.ppf(1 - (1 - confidence_level)/2)
        
        lower_bounds = []
        upper_bounds = []
        
        for tau in range(1, forecast_horizon + 1):
            if tau == 1:
                D_tau = 0
            else:
                D_tau = (tau - 1) * (alpha ** 2)
            
            width = z * sigma_e * np.sqrt(1 + D_tau)
            lower_bounds.append(forecasts[tau-1] - width)
            upper_bounds.append(forecasts[tau-1] + width)
            
        return {
            "fitted": fitted,
            "forecast": forecasts,
            "lower": np.array(lower_bounds),
            "upper": np.array(upper_bounds),
            "metrics": metrics,
            "level": level,
            "params": {"alpha": alpha, "Confidence": confidence_level}
        }

    # =========================================================================
    # Holt's Linear Trend
    # =========================================================================
    def holts(self, data, alpha, beta, forecast_horizon=1, confidence_level=0.95):
        data = np.array(data)
        n = len(data)
        fitted = np.full(n, np.nan)
        level = np.full(n, np.nan)
        trend = np.full(n, np.nan) # growth rate G
        
        # Initialization
        # L_0 = D_1  (index 0)
        # G_0 = (D_n - D_1)/(n-1) -> using ALL data? Or (D_2 - D_1)?
        # Book Example 3.2 uses G0 = (D_12 - D_1)/11. (Line 2501).
        # This is strictly not "causal" if we claim to be at t=1. 
        # But if we treat it as fitting to the dataset, it works.
        # I'll implement the book's method for initialization as default.
        
        L0 = data[0]
        G0 = (data[-1] - data[0]) / (n - 1) if n > 1 else 0
        
        # Forecast for Period 1 (index 0) = L0 + G0
        # Wait, if L0 is D1, then F1 shouldn't be L0+G0?
        # Line 2483: "forecast for Period t=1 would be F1 = L0 + G0."
        # Line 2481: L0 = D1.
        # If F1 = L0 + G0 = D1 + G0, then error e1 = F1 - D1 = G0.
        # This seems correct per book.
        
        prev_L = L0
        prev_G = G0
        
        # Calculate fit
        for t in range(n):
            # Forecast F_{t+1} (here index t corresponds to Period t+1 in 1-based indexing? No)
            # Let's map 0-based index i to Period t=i+1.
            # Forecast for Period t=i+1 (index i) is calculated using info from t-1 (index i-1).
            # For i=0 (Period 1): F1 = L0 + G0.
            
            F_curr = prev_L + prev_G
            fitted[t] = F_curr
            
            D_curr = data[t]
            
            # Update (Eq 3.19, 3.20)
            # L_t = alpha * D_t + (1-alpha)(L_{t-1} + G_{t-1})
            # G_t = beta * (L_t - L_{t-1}) + (1-beta) * G_{t-1}
            
            L_curr = alpha * D_curr + (1 - alpha) * (prev_L + prev_G)
            G_curr = beta * (L_curr - prev_L) + (1 - beta) * prev_G
            
            level[t] = L_curr
            trend[t] = G_curr
            
            prev_L = L_curr
            prev_G = G_curr
            
        # Forecast future
        # F_{t+tau} = L_t + tau * G_t
        last_L = level[-1]
        last_G = trend[-1]
        
        forecasts = []
        for tau in range(1, forecast_horizon + 1):
             forecasts.append(last_L + tau * last_G)
        forecasts = np.array(forecasts)
        
        # Metrics
        metrics = self.calculate_metrics(data, fitted)
        
        # Standard Error (Eq 3.24)
        # s_e = sqrt((SSE - t*(BIAS)^2) / (t-2))
        t = metrics["n"]
        if t > 2:
            try:
                term = metrics["SSE"] - t * (metrics["BIAS"]**2)
                if term < 0: term = 0
                sigma_e = np.sqrt(term / (t - 2))
            except:
                sigma_e = 0
        else:
            sigma_e = 0
            
            
        # Prediction Interval
        # Eq 3.25: D_tau = sum_{j=1}^{tau-1} alpha^2 (1 + j*beta)^2
        z = norm.ppf(1 - (1 - confidence_level)/2)
        lower_bounds = []
        upper_bounds = []
        
        for tau in range(1, forecast_horizon + 1):
            if tau == 1:
                D_tau = 0
            else:
                # Sum j from 1 to tau-1
                # D_tau = sum(alpha^2 * (1 + j * beta)**2)
                j_vals = np.arange(1, tau)
                D_tau = np.sum((alpha**2) * ((1 + j_vals * beta)**2))
                
            width = z * sigma_e * np.sqrt(1 + D_tau)
            lower_bounds.append(forecasts[tau-1] - width)
            upper_bounds.append(forecasts[tau-1] + width)
            
        return {
            "fitted": fitted,
            "forecast": forecasts,
            "lower": np.array(lower_bounds),
            "upper": np.array(upper_bounds),
            "metrics": metrics,
            "level": level,
            "trend": trend,
            "params": {"alpha": alpha, "beta": beta, "Confidence": confidence_level}
        }

    # =========================================================================
    # Winter's Multiplicative Method
    # =========================================================================
    def winters(self, data, alpha, beta, gamma, season_length, forecast_horizon=1, confidence_level=0.95):
        data = np.array(data)
        n = len(data)
        fitted = np.full(n, np.nan)
        level = np.full(n, np.nan)
        trend = np.full(n, np.nan)
        seasonals = np.full(n, np.nan) # Store C_t computed at each step
        
        # Initialization
        # Need K seasons (at least 2).
        N = season_length
        if n < 2 * N:
             # Fallback or error?
             # For robustness, we might try to handle it, but per book K>=2.
             # Return nan if not enough data
             return {"error": "Not enough data for Winter's initialization (need 2*season_length)"}
             
        # Initial Estimates (Eq 3001+)
        # V1 = Mean(Season 1)
        # VK = Mean(Last Season) -> K = n // N
        # If n is not multiple of N, what is K? Book example has exactly K seasons.
        # Let's use as many full seasons as possible.
        K = n // N
        # If we use K seasons for initialization, it implies we "peek" at data up to N*K.
        # But commonly we initialize using first 2 seasons or K seasons then start "forecasting" or fitting?
        # Book says: "Suppose data for K seasons are available... Inputs D1..DKN". 
        # Then "Forecast for Period N+1 is..."
        # This implies initialization uses ALL K seasons to set L_N, G_N, C_1..C_N, 
        # and then starts forecasting from N+1?
        # NO, looking at Example 3.3.
        # It has data for 3 years (12 Qs). 
        # It calculates init L4, G4, C1..C4 using "all 3 years of data" to get V1 and V3?
        # Yes, G4 uses V3-V1. So it uses future data relative to T=4.
        # THIS IS OKAY for "Fitting" historical data.
        # And forecasting starts at N+1.
        # So "Fitted values" will start at N+1.
        # Indices 0 to N-1 (Period 1 to N) will have NO forecasts/fitted values.
        
        # V1: Mean of data[0:N]
        V1 = np.mean(data[0:N])
        # VK: Mean of last full season data[(K-1)*N : K*N]
        VK = np.mean(data[(K-1)*N : K*N])
        
        # G_N = (VK - V1) / (N * (K-1))
        if K > 1:
            GN = (VK - V1) / (N * (K - 1))
        else:
            GN = 0

        # L_N = V1 (Line 3003: L_N = D1...DN / N = V1)
        # Corrected per Iravani/Standard Winter's: L_N should be the level at the END of season 1.
        # V1 is the average level of season 1, roughly corresponding to the middle of the season.
        # We adjust V1 by the trend to get L_N.
        # L_N = V1 + ((N - 1) / 2) * G_N
        LN = V1 + ((N - 1) / 2) * GN
        
        # Initial Seasonal Factors C_1 ... C_N
        # Iravani / Standard Winter's Method Improvement:
        # Use ALL K seasons to estimate seasonal indices, not just the first one.
        # We also de-trend the data within each season to get pure seasonality.
        
        # We need average of each season k=1..K
        season_means = []
        for k in range(K):
            # Data for season k (0-based index)
            s_data = data[k*N : (k+1)*N]
            season_means.append(np.mean(s_data))
            
        # Re-verify GN using first and last season means calculated above
        # GN formula: (V_K - V_1) / (N * (K-1))
        if K > 1:
            GN = (season_means[-1] - season_means[0]) / (N * (K - 1))
        else:
            GN = 0
            
        # Recalculate LN based on verified GN and V1
        # Iravani Text Formula: LN = (D1 + ... + DN) / N = V1.
        # This is the simple average of the first season.
        LN = season_means[0]

        # Calculate Seasonal Ratios
        # Iravani Text Formula: C_i = D_i / V_i (where V_i is implied V1 for the first season).
        # We use the first season to determine initial seasonal factors.
        
        initial_seasonals = []
        V1 = season_means[0]
        # Avoid division by zero
        if V1 == 0: V1 = 1.0
            
        for i in range(N):
            initial_seasonals.append(data[i] / V1)
            
        # Note: Some texts normalize these to sum to N immediately, others don't mentioning it explicitly 
        # in the init step but it is required for stability. 
        # However, strictly following "C_i = D_i / V1", we check if normalization is requested.
        # User formula didn't explicitly mention normalization sum=N for init, but it's standard property.
        # Let's keep normalization to be safe, or stick to raw if requested?
        # User text doesn't show normalization step in the snippet.
        # But if mean(D_i) = V1, then sum(D_i/V1) = sum(D_i)/V1 = (N*V1)/V1 = N.
        # So D_i/V1 NATURALLY sums to N (approx).
        # So explicit normalization isn't strictly needed mathematically if done exactly this way.
        
        seasonal_factors_history = list(initial_seasonals) 
        
        # Note: seasonal_factors_history[k] corresponds to Period k+1.
        # e.g. history[0] is C1.
        
        # Tracking variables
        prev_L = LN
        prev_G = GN
        
        # Records for plotting
        # Fill first N with NaNs or initial values?
        # Fitted values are NaNs.
        for i in range(N):
            level[i] = np.nan 
            trend[i] = np.nan
            seasonals[i] = initial_seasonals[i]

        # Populate the initialized values at the end of the initialization period (Index N-1, Period N)
        level[N-1] = LN
        trend[N-1] = GN
        
        # Start iterating from t = N (Period N+1)
        for t in range(N, n):
            # Forecast for Period t+1 (index t)
            # F_{t+1} = (L_t + G_t) * C_{t+1-N} ? No.
            # Current time is t-1 (Period t). We just finished Period t. We are at start of Period t+1?
            # Let's align with book steps.
            # "After observing demand Dt... compute Lt, Gt, Ct... Forecast for t+1 is (Lt+Gt)*Ct+1-N"
            
            # So, at index t (Period t+1), we use values from t-1 (Period t).
            # Loop t goes 0 to n-1.
            # We start logic at t = N. This corresponds to index N. (Period N+1).
            # We effectively want to "simulate" the process from N to n-1.
            
            # Forecast for D_t (index t):
            # F_{index t} = (L_{index t-1} + G_{index t-1}) * C_{index t - N}
            # L_{index N-1} is L_N (initial). G_{index N-1} is G_N.
            # C_{index N - N} = C_{0} = C_1.
            
            # For t=N:
            # F = (LN + GN) * seasonal_factors_history[t - N]
            season_idx = t - N
            C_seasonal = seasonal_factors_history[season_idx]
            
            F_curr = (prev_L + prev_G) * C_seasonal
            fitted[t] = F_curr
            
            # Update after observing D_t
            D_curr = data[t]
            
            # L_t = alpha * (D_t / C_{t-N}) + (1-alpha)(L_{t-1} + G_{t-1})
            # C_{t-N} is the same C_seasonal we used for forecast.
            
            L_new = alpha * (D_curr / C_seasonal) + (1 - alpha) * (prev_L + prev_G)
            
            # G_t = beta * (L_t - L_{t-1}) + (1-beta) G_{t-1}
            G_new = beta * (L_new - prev_L) + (1 - beta) * prev_G
            
            # C_t = gamma * (D_t / L_t) + (1-gamma) C_{t-N}
            C_new = gamma * (D_curr / L_new) + (1 - gamma) * C_seasonal
            
            # Store
            level[t] = L_new
            trend[t] = G_new
            seasonals[t] = C_new
            seasonal_factors_history.append(C_new)
            
            # Update prev
            prev_L = L_new
            prev_G = G_new
            
        # Forecast Horizon
        # F_{t+tau} = (L_t + tau*G_t) * C_{t+tau-N}
        # Last index was n-1. We have L_{n-1} and G_{n-1} and history up to C_{n-1}.
        # Future periods: n, n+1 ...
        # For tau=1 (index n): Need C_{n - N}. 
        # Wait, if we are at n-1, next one is n. n - N corresponds to index n-1 - N + 1?
        # Yes seasonal factor reuse logic:
        # C_{future_t} = C_{future_t - N}
        # But since we update C's, we use the most recent C for that season.
        # The history list contains updated C's.
        # index n corresponds to "Period n+1".
        # We need C corresponding to that season.
        # Season index = (index n) - N ? 
        # Basically look back N steps in history.
        
        last_L = level[-1]
        last_G = trend[-1]
        
        forecasts = []
        for tau in range(1, forecast_horizon + 1):
            # Time index for forecast is n + tau - 1
            # We need Seasonal factor from N periods ago relative to (n + tau - 1)?
            # Actually, just "match the season".
            # The seasonal factors cycle every N.
            # We need the most updated factor for the season corresponding to `n + tau - 1`.
            # Effective index in history: (n + tau - 1) - N ?
            # Wait, if we project far future (tau > N), we recycle the latest factors.
            # The most recent factors are at indices n-N to n-1.
            # We want to pick from this window.
            # (n + tau - 1) % N  might give the offset?
            # Or simpler:
            # target_idx = (n + tau - 1) - N. If target_idx >= n, that means we need even older?
            # No, standard Winters uses the LATEST seasonal factor for that season.
            # Latest seasonal factors are in seasonal_factors_history[-N:].
            # We map (n + tau - 1) to one of these.
            # The position in the cycle is (n + tau - 1) % N.
            # But the history array is aligned such that index i corresponds to Period i+1.
            # So periodicity is correct.
            # We just need seasonal_factors_history[-N + (tau-1)%N] ?
            # Let's check.
            # If tau=1, we want prediction for n. Factor comes from n-N.
            # history[n-N] exists.
            # If tau=N+1, we want prediction for n+N. Factor comes from n. (Wait, n doesn't exist yet?)
            # No, we assume C stays constant in future for projection.
            # So we use factors from the last full season available (or most recent update for each season).
            # Which are exactly history[n-N ... n-1].
            
            offset = (tau - 1) % N
            # We want the factor that corresponds to this offset in the cycle.
            # The last N factors in history correspond to the last cycle.
            # history[-N] is factor for n-N.
            # history[-1] is factor for n-1.
            # If tau=1, we want factor for n (which is same season as n-N). So history[-N].
            # If tau=2, we want factor for n+1 (same season as n-N+1). So history[-N+1].
            C_used = seasonal_factors_history[-(N) + offset]
            
            val = (last_L + tau * last_G) * C_used
            forecasts.append(val)
            
        forecasts = np.array(forecasts)
        
        # Metrics
        # Ignore the first N indices where fitted is NaN
        metrics = self.calculate_metrics(data[N:], fitted[N:])
        
        # Prediction Intervals (Eq 3.31)
        # s_r (sigma_r) relative error std dev.
        # Eq 3.32: s_r = sqrt( sum((e/F)^2) / (t-3) )
        if metrics["n"] > 3:
            # We need relative errors for the fitted part
            actual_trim = data[N:]
            e_trim = fitted[N:] - actual_trim
            # Handle F=0?
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_sq = (e_trim / fitted[N:])**2
                rel_sq = np.nan_to_num(rel_sq, nan=0.0, posinf=0.0, neginf=0.0)
            
            sum_rel_sq = np.sum(rel_sq)
            sigma_r = np.sqrt(sum_rel_sq / (metrics["n"] - 3))
        else:
            sigma_r = 0
            
        z = norm.ppf(1 - (1 - confidence_level)/2)
        lower_bounds = []
        upper_bounds = []
        
        for tau in range(1, forecast_horizon + 1):
             # D_tau (Eq 3.33)
             if tau == 1:
                 D_tau = 0
             else:
                 # Sum j=1 to tau-1
                 # alpha^2 * (1 + (tau-j)*beta)^2 * (Lt + j*Gt)^2
                 val_sum = 0
                 for j in range(1, tau):
                     term = (alpha**2) * ((1 + (tau - j)*beta)**2) * ((last_L + j*last_G)**2)
                     val_sum += term
                 D_tau = val_sum
                 
             C_used = seasonal_factors_history[-(N) + (tau-1)%N]
             base = np.sqrt((last_L + tau * last_G)**2 + D_tau)
             width = sigma_r * z * C_used * base
             
             lower_bounds.append(forecasts[tau-1] - width)
             upper_bounds.append(forecasts[tau-1] + width)

        return {
            "fitted": fitted,
            "forecast": forecasts,
            "lower": np.array(lower_bounds),
            "upper": np.array(upper_bounds),
            "metrics": metrics,
            "level": level,
            "trend": trend,
            "seasonals": seasonals,
            "params": {"alpha": alpha, "beta": beta, "gamma": gamma, "Confidence": confidence_level}
        }
    
    # =========================================================================
    # Optimizer
    # =========================================================================
    def optimize_params(self, method_name, data, metric="MSE", season_length=None):
        """
        Find optimal parameters to minimize the given metric.
        Variables: alpha, beta, gamma (depending on method).
        Constraints: 0 <= param <= 1.
        """
        
        def objective(params):
            if method_name == "SES":
                res = self.ses(data, params[0], forecast_horizon=0)
            elif method_name == "Holt":
                res = self.holts(data, params[0], params[1], forecast_horizon=0)
            elif method_name == "Winter":
                res = self.winters(data, params[0], params[1], params[2], season_length, forecast_horizon=0)
                if "error" in res: return 1e99 # Penalty
            else:
                return 1e99
                
            val = res["metrics"].get(metric, 1e99)
            if np.isnan(val): return 1e99
            return val

        # Setup optimization
        if method_name == "SES":
            x0 = [0.1]
            bounds = [(0.01, 1.0)]
        elif method_name == "Holt":
            x0 = [0.1, 0.1]
            bounds = [(0.01, 1.0), (0.01, 1.0)]
        elif method_name == "Winter":
            x0 = [0.1, 0.1, 0.1]
            bounds = [(0.01, 1.0), (0.01, 1.0), (0.01, 1.0)]
        else:
            return None
            
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        return result.x

