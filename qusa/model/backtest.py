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

    def run_backtest(self, initial_capital, position_size, transaction_cost):
        """
        Simulate backtest on model with corrected loop logic.
        """
        print("\n" + "=" * 80)
        print(f"RUNNING BACKTEST (Friction: {transaction_cost}%)")
        print("=" * 80)

        # Prepare features and predictions
        x = self.data[self.features].fillna(0)
        predictions = self.model.predict(x)
        probabilities = self.model.predict_proba(x)[:, 1]

        # Setup results DataFrame
        results = self.data[["date", "close", "overnight_delta"]].copy()
        results["predicted_direction"] = predictions
        results["predicted_probability"] = probabilities

        # Determine confidence
        results["high_confidence"] = (
            results["predicted_probability"] >= self.threshold
        ) | (results["predicted_probability"] <= (1 - self.threshold))

        # Initialize return columns with 0.0 to avoid NaN issues later
        results["strategy_return"] = 0.0
        results["gross_strategy_return"] = 0.0

        for idx in results.index:
            if not results.loc[idx, "high_confidence"]:
                continue

            prediction = results.loc[idx, "predicted_direction"]
            actual_return = results.loc[idx, "overnight_delta"]

            # Calculate gross return
            if prediction == 1:
                gross = actual_return * position_size
            else:
                gross = -actual_return * position_size

            # Store values using .at or .loc specifically for the current index
            results.at[idx, "gross_strategy_return"] = gross
            results.at[idx, "strategy_return"] = gross - transaction_cost

        # Calculate cumulative returns
        results["cumulative_return"] = (
            1 + (results["strategy_return"] / 100)
        ).cumprod()
        results["portfolio_value"] = initial_capital * results["cumulative_return"]

        # Benchmark
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

        # handle case with no results
        if self.results is None:
            return {}

        # 1) store first, final equity values for strategy and buy & hold
        first_equity = float(
            self.results["portfolio_value"].iloc[0]
            / self.results["cumulative_return"].iloc[0]
        )
        final_equity_strategy = float(self.results["portfolio_value"].iloc[-1])
        final_equity_buy_hold = float(self.results["buy_hold_value"].iloc[-1])

        # 2) calculate performance for strategy and buy & hold
        ## strategy return
        strategy_gain = final_equity_strategy - first_equity
        strategy_return = strategy_gain / first_equity

        ## buy & hold return
        buy_hold_gain = final_equity_buy_hold - first_equity
        buy_hold_return = buy_hold_gain / first_equity

        ## 2) calculate alpha
        alpha = strategy_return - buy_hold_return

        # 4) calculate win/loss percentage
        ## store total, positive return, and negative return trades
        trades = self.results.loc[self.results["high_confidence"] == True].copy()
        trades_positive_return = trades.loc[trades["strategy_return"] > 0]
        trades_negative_return = trades.loc[trades["strategy_return"] < 0]

        ## 5) calculate win/loss percentages
        ## count total, positive, and negative return trades
        trades_count = len(trades)
        trades_positive_return_count = len(trades_positive_return["strategy_return"])
        trades_negative_return_count = len(trades_negative_return["strategy_return"])
        ## calculate win percentages
        if trades_count > 0:
            win_rate = trades_positive_return_count / trades_count
        else:
            win_rate = 0
        ## calculate loss percentage
        loss_rate = 1 - win_rate

        # 6) calculate risk metrics
        ## annualized volatility (assume 252 trading days)
        daily_std = float(self.results["strategy_return"].std())
        annual_vol = daily_std * np.sqrt(252)

        ## Sharpe ratio (assume risk-free rate = 0.0%)
        daily_mean = float(self.results["strategy_return"].mean())

        if annual_vol != 0:
            sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        ## max draw down
        rolling_max = self.results["portfolio_value"].cummax()
        draw_down = (self.results["portfolio_value"] / rolling_max) - 1
        max_draw_down = float(draw_down.min())

        # 5) compile metrics dictionary
        metrics = {
            "total_trades": trades_count,
            "winning_trades": trades_positive_return_count,
            "losing_trades": trades_negative_return_count,
            "win_percentage": win_rate,
            "loss_percentage": loss_rate,
            "strategy_return": strategy_return,
            "buy_hold_return": buy_hold_return,
            "strategy_gain": strategy_gain,
            "buy_hold_gain": buy_hold_gain,
            "strategy_value": final_equity_strategy,
            "buy_hold_value": final_equity_buy_hold,
            "alpha": alpha,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_draw_down": max_draw_down * 100,
        }

        self._print_results_to_console(metrics)

        return metrics

    @staticmethod
    def _print_results_to_console(metrics):
        """
        Print performance metrics to console.

        Parameters:
            1) metrics (dict): Performance metrics
        """

        print("\n" + "=" * 30)
        print(" BACKTEST SUMMARY")
        print("=" * 30)
        print(f"Total Trades:       {metrics['total_trades']}")
        print(f"Winning Trades:     {metrics['winning_trades']}")
        print(f"Losing Trades:      {metrics['losing_trades']}")
        print(f"Win Rate:           {metrics['win_percentage'] * 100:.2f}%")
        print(f"Loss Rate:          {metrics['loss_percentage'] * 100:.2f}%")
        print("-" * 30)
        print(f"Strategy Return:    {metrics['strategy_return'] * 100:.2f}%")
        print(f"Buy & Hold Return:  {metrics['buy_hold_return'] * 100:.2f}%")
        print(f"Alpha:              {metrics['alpha']:.2f}")
        print("-" * 30)
        print(f"Final Strategy Val: ${metrics['strategy_value']:,.2f}")
        print(f"Final Buy & Hold:   ${metrics['buy_hold_value']:,.2f}")
        print("-" * 30)
        print(f"Strategy Profit:    ${metrics['strategy_gain']:,.2f}")
        print(f"Buy & Hold Profit:  ${metrics['buy_hold_gain']:,.2f}")
        print("-" * 30)
        print(f"Annual Volatility:  {metrics['annual_volatility']:.2f}")
        print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
        print(f"Max Draw down:      {metrics['max_draw_down']:.2f}%")
        print("=" * 30)

        return

    def plot_results(self, save_path):
        """
        Plot backtest results.

        Parameters:
            1) save_path (str): Path to save plot
        """

        fig, ax = plt.subplots(3, 1, figsize=(12, 8))

        # plot 1: portfolio value vs buy & hold
        ax[0].plot(
            self.results["date"],
            self.results["portfolio_value"],
            label="Strategy",
            color="#0f4c5c",
            linestyle="--",
        )
        ax[0].plot(
            self.results["date"],
            self.results["buy_hold_value"],
            label="Buy & Hold",
            color="#9a031e",
        )
        ax[0].set_title("Portfolio Value", fontsize=14)
        ax[0].set_xlabel("Date", fontsize=12)
        ax[0].set_ylabel("Portfolio Value ($)", fontsize=12)
        ax[0].legend()

        # plot 2: draw down
        cumulative = self.results["portfolio_value"]
        running_max = cumulative.expanding().max()
        draw_down = (cumulative - running_max) / running_max * 100
        ax[1].plot(self.results["date"], draw_down, color="#9a031e")
        ax[1].fill_between(
            self.results["date"],
            draw_down,
            0,
            where=(draw_down < 0),
            color="#ef233c",
            alpha=0.3,
        )
        ax[1].set_title("Draw Down", fontsize=14)
        ax[1].set_xlabel("Date", fontsize=12)
        ax[1].set_ylabel("Draw Down (%)", fontsize=12)

        # plot 3: trade distribution
        traded = self.results.loc[self.results["high_confidence"]]
        ax[2].scatter(
            traded["date"],
            traded["strategy_return"],
            c=traded["strategy_return"].apply(
                lambda x: "#0f4c5c" if x > 0 else "#9a031e"
            ),
        )
        ax[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax[2].set_title("Trade Returns Distribution", fontsize=14)
        ax[2].set_xlabel("Date", fontsize=12)
        ax[2].set_ylabel("Trade Return (%)", fontsize=12)

        plt.tight_layout()

        # save plot
        save_path = os.path.expanduser(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\n✓ Results saved to {save_path}")

        return
