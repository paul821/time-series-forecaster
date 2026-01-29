
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
        
        # New LN = 40.0
        # GN = 10.0
        
        res = fm.winters(data, alpha=0, beta=0, gamma=0, season_length=4) 
        
        # Check that Level and Trend are populated at index N-1 (Period 4)
        LN_actual = res["level"][3]
        GN_actual = res["trend"][3]
        
        print(f"Level at index 3 (Period 4): {LN_actual}")
        print(f"Trend at index 3 (Period 4): {GN_actual}")
        
        self.assertFalse(np.isnan(LN_actual), "Level at Period N should not be NaN")
        self.assertFalse(np.isnan(GN_actual), "Trend at Period N should not be NaN")
        
        self.assertAlmostEqual(LN_actual, 40.0, places=3, msg="LN should match trend-adjusted formula")
        self.assertAlmostEqual(GN_actual, 10.0, places=3, msg="GN should match standard formula")

        # Also verify Fitted starts at index N (Period 5)
        fitted_5 = res["fitted"][4]
        # F_5 = (LN + GN) * C_1 = (40+10)*0.4 = 20.
        self.assertAlmostEqual(fitted_5, 20.0, places=3)
        print("Verification SUCCESS.")

if __name__ == '__main__':
    unittest.main()
