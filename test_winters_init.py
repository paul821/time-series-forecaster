
import unittest
import numpy as np
import pandas as pd
from forecasting import ForecastingModels

class TestWintersInit(unittest.TestCase):
    def test_winters_initialization(self):
        fm = ForecastingModels()
        
        # Create data for 2 seasons (N=4, K=2) with trend
        # Season 1: 10, 20, 30, 40 (Mean = 25)
        # Season 2: 50, 60, 70, 80 (Mean = 65)
        # Trend per season diff = 40.
        # Season length N=4.
        
        data = [10, 20, 30, 40, 50, 60, 70, 80]
        N = 4
        
        # Original V1 = 25
        # Original VK (V2) = 65
        # GN = (V2 - V1) / (N * (K-1)) = (65 - 25) / (4 * 1) = 40 / 4 = 10.
        
        # Old LN = V1 = 25.
        # New LN = V1 + ((N-1)/2) * GN = 25 + (3/2)*10 = 25 + 1.5*10 = 25 + 15 = 40.
        # This makes sense because at the end of season 1 (t=4), the value is 40.
        
        # We need to expose or infer internal state. 
        # winters() returns "level" array.
        # Note: level[0]...level[N-1] are initialized to NaN in current code? 
        # Let's check code:
        # for i in range(N): level[i] = NaN
        # Then loop starts at t=N (Period N+1 = Period 5).
        # Inside loop at t=N:
        # F_curr = (prev_L + prev_G) * ...
        # where prev_L = LN.
        # So we can't directly see LN in the returned level array at index N-1.
        # BUT, we can check level[N] (first calculated level).
        # Or better, we can modify the test to subclass or inspect using a debugger, but simplest is to check the Forecast/Fitted values?
        # Actually, let's just inspect the result object if we can or infer from first forecast.
        
        # Let's assume we can infer from the first fitted value at t=N (Period 5).
        # F_{t=4} (index 4) = (LN + GN) * C_{index 4-N} = (LN + GN) * C_0
        # C_0 = D_0 / V1 = 10 / 25 = 0.4.
        # If LN = 40, GN = 10 -> (40+10)*0.4 = 50 * 0.4 = 20.
        # Wait, index 4 is Period 5. Data[4] is 50.
        # Forecast is 20? 
        # Let's recheck seasonality logic.
        # C_i = D_i / V1.
        # Season 1: 10/25=0.4, 20/25=0.8, 30/25=1.2, 40/25=1.6.  (Avg = 1.0 OK)
        # Season 2: 50/65 approx. 
        
        # With Old LN=25, GN=10 -> (25+10)*0.4 = 35 * 0.4 = 14.
        # With New LN=40, GN=10 -> (40+10)*0.4 = 20.
        
        # Let's run it.
        res = fm.winters(data, alpha=0, beta=0, gamma=0, season_length=4) # Zero parameters to keep init state
        
        fitted_5 = res["fitted"][4] # Index 4 is 5th point
        print(f"Fitted[4] (Period 5): {fitted_5}")
        
        # We expect LN=40, GN=10.
        # F_5 = (40+10) * C_1 = 50 * 0.4 = 20.
        
        actual_LN_implied = (fitted_5 / 0.4) - 10
        print(f"Implied LN: {actual_LN_implied}")
        
        if abs(actual_LN_implied - 40) < 0.001:
             print("Verification SUCCESS: LN matches trend-adjusted expectation.")
        else:
             print(f"Verification FAILED: Expected LN ~ 40, got {actual_LN_implied}")
             self.fail("LN initialization does not match trend-adjusted formula")

if __name__ == '__main__':
    unittest.main()
