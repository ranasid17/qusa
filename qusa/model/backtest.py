# qusa/qusa/model/backtest.py

"""
Backtest overnight delta prediction model.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from datetime import datetime


class ModelBacktester:
    """
    Backtest trading strategy based on
    model predictions.
    """

    def __init__(self, model_path, backtest_data_path):
        """
        Class constructor.

        Parameters:
            1) model_path (str): Path to saved/trained model
            2) backtest_data_path (str): Path to data for backtesting
        """

        # store paths to model and dataset as attributes
        self.model_path = os.path.expanduser(model_path)
        self.backtest_data_path = os.path.expanduser(backtest_data_path)

        # load model and dataset as attributes
        self._load_model()
        self._load_data()

        self.results = None

    def _load_model(self):
        """
        Load saved/trained model from attribute path.
        """

        bundle = joblib.load(self.model_path)

        self.model = bundle["model"]
        self.features = bundle["features"]
        self.threshold = bundle["threshold"]

        print(f"✓ Model loaded")

        return

    def _load_data(self):
        """
        Load backtest dataset from attribute path.
        """

        # load data from path and confirm datetime type
        self.data = pd.read_csv(self.model_path)
        self.data["date"] = pd.to_datetime(self.data["date"])

        print(f"✓ Loaded {len(self.data)} days of data")

        return

    def run_backtest(self, initial_capital, position_size):
        """
        Simulate backtest on model.

        Parameters:
            1) initial_capital (float): Starting investment capital
            2) position_size (float): Fraction of capital to use per trade

        Returns:
            1) result (pd.DataFrame): Backtest result
        """

        print("\n" + "=" * 80)
        print("RUNNING BACKTEST")
        print("=" * 80)

        # prepare features
        X = self.data[self.features].fillna(0)

        # make predictions and store likelihoods
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        # store results as DataFrame
        results = self.data[["date", "close", "overnight_delta"]].copy()
        results["prediction"] = predictions
        results["probability_up"] = probabilities
        results["true_direction"] = results.loc[results["overnight_delta"] > 0].astype(
            int
        )

        # label high confidence predictions
        results["high_confidence"] = (results["probability_up"] >= self.threshold) | (
            results["probability_up"] <= (1 - self.threshold)
        )

        # calculate returns
        results["returns"] = 0.0

        for idx in results.idx:
            # skip low confidence predictions
            if not results.loc[idx, "high_confidence"]:
                continue

            # store predicted direction and true return
            prediction = results.loc[idx, "prediction"]
            actual_return = results.loc[idx, "overnight_delta"]

            # handle cases when model indicates buy
            if prediction == 1:
                results.loc[idx, "strategy_return"] = actual_return * position_size
            # otherwise model indicates short sell/do not buy
            else:
                results.loc["idx", "strategy_return"] = -actual_return * position_size

        # calculate cumulative returns
        results["cumulative_return"] = (
            1 + (results["strategy_return"] / 100)
        ).cumprod()
        results["portfolio_value"] = initial_capital * results["cumulative_return"]

        # benchmark to buy/hold
        first_close = results["close"].iloc[0]
        results["buy_hold_value"] = initial_capital * (results["close"] / first_close)

        self.results = results

        return results

    def calculate_metrics(self):
        """
        Calculate backtest model performance.
        """

        print("\n" + "=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)

        # filter to days with trade executed
        traded = self.results.loc[self.results["high_confidence"]]

        # basic statistics
        total_trades = len(traded)
        positve_return_trades = self.results.loc[self.results["strategy_return"] > 0]
        negative_return_trades = self.results.loc[self.results["strategy_return"] < 0]

        # calculate success rate
        if total_trades > 0:
            success_rate = positve_return_trades / total_trades
        else:
            success_rate = 0

        # calculate overall return
        buy_hold_return = (
            self.results["buy_hold_value"].iloc[-1]
            / self.results["buy_hold_value"].iloc[0]
            - 1
        ) * 100
        strategy_return = (
            self.results["portfolio_value"].iloc[-1]
            / self.results["portfolio_value"].iloc[0]
            - 1
        ) * 100

        #
