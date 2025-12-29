# qusa/qusa/model/backtest.py

"""
Backtest overnight delta prediction model.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


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
        self.data = pd.read_csv(self.backtest_data_path)
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

        for idx in results.index:
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
                results.loc[idx, "strategy_return"] = -actual_return * position_size

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

        Returns:
            1) metrics (dict): Performance metrics
        """

        print("\n" + "=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)

        # filter to days with trade executed
        traded = self.results.loc[self.results["high_confidence"]]

        # basic statistics
        total_trades = len(traded)
        positive_return_trades = self.results.loc[self.results["strategy_return"] > 0]
        negative_return_trades = self.results.loc[self.results["strategy_return"] < 0]

        # calculate success rate
        if total_trades > 0:
            success_rate = positive_return_trades / total_trades
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

        # calculate risk metrics
        daily_return = traded["strategy_return"]  # filter returns on traded days
        volatility = daily_return.std()

        if volatility > 0:
            sharpe_ratio = (daily_return.mean() / volatility) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # calculate draw down
        cumulative = self.results["portfolio_value"]
        running_max = cumulative.expanding().max()
        draw_down = (cumulative - running_max) / running_max * 100
        max_draw_down = draw_down.min()

        # print metrics
        print(f"\nTrading Statistics:")
        print(f"  Total trades: {total_trades}")
        print(f"  Winning trades: {positive_return_trades}")
        print(f"  Losing trades: {negative_return_trades}")
        print(f"  Win rate: {success_rate:.1%}")

        print(f"\nReturn Metrics:")
        print(f"  Strategy return: {strategy_return:+.2f}%")
        print(f"  Buy & Hold return: {buy_hold_return:+.2f}%")
        print(f"  Alpha: {strategy_return - buy_hold_return:+.2f}%")

        print(f"\nRisk Metrics:")
        print(f"  Volatility (daily): {volatility:.2f}%")
        print(f"  Sharpe ratio: {sharpe_ratio:.2f}")
        print(f"  Max draw_down: {max_draw_down:.2f}%")

        # store metrics in dictionary
        metrics = {
            "total_trades": total_trades,
            "success_rate": success_rate,
            "strategy_return": strategy_return,
            "buy_hold_return": buy_hold_return,
            "sharpe_ratio": sharpe_ratio,
            "max_draw_down": max_draw_down,
            "alpha": strategy_return - buy_hold_return,
        }

        return metrics

    def plot_results(self, save_path):
        """
        Plot backtest results.

        Parameters:
            1) save_path (str): Path to save plot
        """

        fig, ax = plt.subplots(3, 1, figsize=(12, 8))

        # plot 1: portfolio value vs buy & hold
        ax[0].plot(
            self.results["date"], self.results["portfolio_value"], label="Strategy"
        )
        ax[0].plot(
            self.results["date"], self.results["buy_hold_value"], label="Buy & Hold"
        )
        ax[0].set_title("Portfolio Value Over Time", fontsize=14)
        ax[0].set_xlabel("Date", fontsize=12)
        ax[0].set_ylabel("Portfolio Value ($)", fontsize=12)
        ax[0].legend()

        # plot 2: draw down
        cumulative = self.results["portfolio_value"]
        running_max = cumulative.expanding().max()
        draw_down = (cumulative - running_max) / running_max * 100
        ax[1].plot(self.results["date"], draw_down, color="darkred")
        ax[1].fill_between(
            self.results["date"],
            draw_down,
            0,
            where=(draw_down < 0),
            color="red",
            alpha=0.3,
        )
        ax[1].set_title("Draw Down Over Time", fontsize=14)
        ax[1].set_xlabel("Date", fontsize=12)
        ax[1].set_ylabel("Draw Down (%)", fontsize=12)

        # plot 3: trade distribution
        traded = self.results.loc[self.results["high_confidence"]]
        ax[2].scatter(
            traded["date"],
            traded["strategy_return"],
            c=traded["strategy_return"].apply(lambda x: "g" if x > 0 else "r"),
        )
        ax[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax[2].set_title("Trade Returns Distribution", fontsize=14)
        ax[2].set_xlabel("Date", fontsize=12)
        ax[2].set_ylabel("Trade Return (%)", fontsize=12)

        plt.tight_layout()

        # save plot
        save_path = os.path.expanduser(save_path)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\n✓ Results saved to {save_path}")

        return
