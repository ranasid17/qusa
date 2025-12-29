# QUSA/qusa/features/overnight.py

import pandas as pd


class OvernightCalculator:
    """
    Calculates overnight features for financial time series data.
    """

    def __init__(self, date_col="date", open_col="open", close_col="close"):
        """
        Class constructor.

        Parameters:
            1) date_col (str): Name of the date column.
            2) open_col (str): Name of the opening price column.
            3) close_col (str): Name of the closing price column.
        """

        self.date = date_col
        self.open = open_col
        self.close = close_col

    def calculate_overnight_delta(self, df):
        """
        Calculate the overnight price change for each trading day in the DataFrame.

        Parametesrs:
            1) df (pd.DataFrame): DataFrame containing stock data with 'Date', 'Open', and 'Close' columns.

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with an additional 'Overnight_Delta' column.
        """

        df_mod = df.copy()  # copy the original DataFrame to avoid modifying it directly

        df_mod[self.date] = pd.to_datetime(  # confirm date column as datetime type
            df_mod[self.date]
        )
        df_mod = df_mod.sort_values(by=self.date).reset_index(
            drop=True
        )  # sort by date and reset index

        # calculate overnight change and percentage change
        df_mod["overnight_delta"] = df_mod[self.open] - df_mod[self.close].shift(1)
        df_mod["overnight_delta_pct"] = (
            df_mod["overnight_delta"] / df_mod[self.close].shift(1)
        ) * 100

        return df_mod

    @staticmethod
    def identify_abnormal_delta(df, threshold=2.0):
        """
        Identify abnormal overnight price changes.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data with 'Overnight_Delta' column.
            2) threshold (float): Threshold value for identifying abnormal overnight delta.

        Returns:
            1) df_mod (pd.DataFrame): DataFrame containing days with abnormal overnight delta.
        """

        df_mod = df.copy()  # copy the original DataFrame to avoid modifying it directly

        mean_delta_pct = df_mod[
            "overnight_delta_pct"
        ].mean()  # calculate mean overnight change percentage
        std_dev = df_mod["overnight_delta_pct"].std()  # repeat for std dev

        # calculate z score and label anomalies
        df_mod["z_score"] = (df_mod["overnight_delta_pct"] - mean_delta_pct) / std_dev
        df_mod["abnormal"] = df_mod["z_score"].abs() > threshold

        return df_mod
