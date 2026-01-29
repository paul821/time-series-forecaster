
import unittest
import numpy as np
import pandas as pd
from forecasting import ForecastingModels

class TestForecastingFix(unittest.TestCase):
    def test_calculate_metrics_with_mixed_types(self):
        fm = ForecastingModels()
        
        # Simulate mixed type data that might come from bad user input or pandas object types
        actual = np.array([100, 200, "300", None, 500], dtype=object)
        forecast = np.array([110, 210, 310, 410, "nan"], dtype=object)
        
        # This should not raise TypeError
        try:
            metrics = fm.calculate_metrics(actual, forecast)
            print("Metrics calculated:", metrics)
        except TypeError as e:
            self.fail(f"calculate_metrics raised TypeError with mixed types: {e}")
            
        # Verify correctness (basic check)
        # Valid pairs: (100, 110), (200, 210), (300, 310) -> converted to float
        # None and "nan" should be treated as NaN and filtered out
        # Pair (None, 410) -> Actual is NaN -> Filtered
        # Pair (500, "nan") -> Forecast is NaN -> Filtered
        
        # Expected N = 3
        self.assertEqual(metrics["n"], 3, "Should have 3 valid data points")

if __name__ == '__main__':
    unittest.main()
