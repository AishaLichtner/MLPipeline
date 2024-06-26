import sys
sys.path.append(r'C:\Users\aisha\MLPipeline\MLPipeline\src')
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import preprocess as pr

class TestPreprocess(unittest.TestCase):
    def test_normalize_outliers_interpolate_approach(self):
        # Create sample DataFrame with outliers
        data = {
            "Daily_mileage": [10, 20, 30, 40, 1000],  # Insert an outlier
            "Other_column": [1, 2, 3, 4, 5]
        }
        df = pd.DataFrame(data)

        # Identify outliers
        outliers = df[df["Daily_mileage"] > 100]  # Assume any value above 100 is an outlier

        # Apply normalize_outliers with interpolate approach
        normalized_df = pr.normalize_outliers(df, outliers, col="Daily_mileage", approach="interpolate")

        # Check if the outlier is interpolated
        self.assertFalse(pd.isna(normalized_df["Daily_mileage"]).any(), "Outliers were not interpolated")

    def test_prophet_approach(self):
        
        
        data = {
            "Daily_mileage": [10, 20, 30, 40, 1000],  # Insert an outlier
            "Other_column": [1, 2, 3, 4, 5]
        }
        df = pd.DataFrame(data)

        # Identify outliers
        outliers = df[df["Daily_mileage"] > 100]  # Assume any value above 100 is an outlier

        # Apply normalize_outliers with interpolate approach
        normalized_df = pr.normalize_outliers(df, outliers, col="Daily_mileage", approach="prophet")

        # Check if the outlier is interpolated
        self.assertTrue(pd.isna(normalized_df["Daily_mileage"]).any(), "Outliers were not left as NaN")

    def test_basic_transformation(self):
        # Sample data
        data = {
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Daily_mileage': [100, 150, 200]
        }
        df = pd.DataFrame(data)

        # Expected result
        expected_data = {
            'ds': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'y': [100, 150, 200]
        }
        expected_df = pd.DataFrame(expected_data)

        # Run the function
        result_df = pr.transform_to_prophet_format(df)

        # Assert the result
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_normalize_outliers_different_column_names(self):
        # Sample data with different column names
        data = {
            'Date_col': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Mileage': [100, 150, 200]
        }
        df = pd.DataFrame(data)

        # Expected result
        expected_data = {
            'ds': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'y': [100, 150, 200]
        }
        expected_df = pd.DataFrame(expected_data)

        # Run the function with different column names
        result_df = pr.transform_to_prophet_format(df, date_col='Date_col', dep_var='Mileage')

        # Assert the result
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_empty_dataframe(self):
        # Empty dataframe
        df = pd.DataFrame(columns=['Date', 'Daily_mileage'])

        # Expected result
        expected_df = pd.DataFrame(columns=['ds', 'y'])

        # Run the function
        result_df = pr.transform_to_prophet_format(df)

        # Assert the result
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_identify_outliers(self):
        data = {
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'Daily_mileage': [100, 150, 200, 300, 1000]  # 1000 should be identified as an outlier
        }
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])

        expected_outliers = pd.DataFrame({
            'Date': ['2023-01-05'],
            'Daily_mileage': [1000]
        })
        expected_outliers['Date'] = pd.to_datetime(expected_outliers['Date'])

        outliers = pr.identifiy_outliers(df)

        pd.testing.assert_frame_equal(outliers.reset_index(drop=True), expected_outliers.reset_index(drop=True))

    @patch('preprocess.handle_missing_values')
    def test_handle_missing_rows(self, mock_handle_missing_values):
        data = {
            'Date': ['2023-01-01', '2023-01-03', '2023-01-04'],
            'Daily_mileage': [100, 200, 300],
            'Asset_ID': [1, 1, 1]
        }
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])

        start_date = '2023-01-01'
        end_date = '2023-01-05'
        expected_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        pr.handle_missing_rows(start_date, end_date, df, col="Daily_mileage", approach="zero")

        # Create expected merged_df to compare with
        expected_data = {
            'Date': expected_dates,
            'Daily_mileage': [100, None, 200, 300, None],
            'Asset_ID': [1, 1, 1, 1, 1]
        }
        expected_df = pd.DataFrame(expected_data)

        expected_df['Date'] = pd.to_datetime(expected_df['Date'])

        mock_handle_missing_values.assert_called_once()
        pd.testing.assert_frame_equal(mock_handle_missing_values.call_args[1]['df'], expected_df)


if __name__ == "__main__":
    unittest.main()
