# qusa/scripts/model_backtest.py

"""
Run backtest on a trained model for a specified stock ticker.
"""

import json
import sys

from datetime import datetime
from pathlib import Path

# add project root to sys.path to import qusa package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from qusa.model import ModelBacktester, generate_backtest_report
from qusa.utils.config import load_config
from qusa.utils.logger import setup_logger


def save_backtest_artifacts(backtester, metrics, output_dir, ticker, logger, config):
    """
    Save backtest artifacts: metrics, plots, and AI report.

    Parameters:
        1) backtester (ModelBacktester): Backtester instance
        2) metrics (dict): Backtest performance metrics
        3) output_dir (Path): Directory to save artifacts
        4) ticker (str): Stock ticker
        5) logger (logging.Logger): Logger instance
        6) config (dict): Configuration dictionary
    """

    # create output directory if not exists
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    # 1) save results CSV
    csv_path = output_path / f"backtest_results_{ticker}_{timestamp}.csv"
    backtester.results.to_csv(csv_path, index=False)
    logger.info(f"✓ Backtest results saved to {csv_path}")

    # 2) save performance metrics JSON
    serialized_metrics = {
        k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()
    }

    serialized_metrics["ticker"] = ticker
    serialized_metrics["timestamp"] = timestamp

    ## write to file
    metrics_path = output_path / f"backtest_metrics_{ticker}_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(serialized_metrics, f, indent=4)
    logger.info(f"✓ Backtest metrics saved to {metrics_path}")

    # 3) save plots
    plot_path = output_path / f"backtest_plot_{ticker}_{timestamp}.png"
    backtester.plot_results(save_path=str(plot_path))

    ## Note: plot_results in the class prints confirmation,
    ##       but will re-log here for consistency in log file
    logger.info(f"✓ Backtest plot saved to {plot_path}")

    # 4) generate AI report
    llm_config = config.get("llm", {})
    enable_reports = llm_config.get("enable_reports", True)

    if not enable_reports:
        logger.info("AI report generation disabled in config")
        return

    try:
        logger.info("Generating AI-powered backtest report...")

        # extract LLM settings from config
        llm = llm_config.get("model_name", "gemma3:4b")
        temp = llm_config.get("temp", 0.2)
        max_context_rows = llm_config.get("max_context_rows", 100)
        base_output_dir = llm_config.get(
            "output_dir", "~/Projects/qusa/reports"
        ).expanduser()
        subdir = llm_config.get("report_subdir", {}).get("backtest", "backtest")

        report_output_dir = base_output_dir / subdir

        report = generate_backtest_report(
            metrics=metrics,
            results_df=backtester.results,
            ticker=ticker,
            llm_name=llm,
            output_dir=str(report_output_dir),
            temperature=temp,
            max_content_rows=max_context_rows,
        )
        logger.info("✓ AI report generated successfully")

    except Exception as e:
        logger.warning(f"⚠ Could not generate AI report: {e}")
        logger.debug("Full traceback:", exc_info=True)

    return


def main():
    """
    Main execution script.
    """

    # 1) set up configuration and logging
    try:
        config_path = PROJECT_ROOT / "qusa" / "utils" / "config.yaml"
        config = load_config(str(config_path))

        # extract log path from config
        log_path = config.get("backtest", {}).get("log_file", "logs/backtest.log")
        logger = setup_logger("model_backtest", log_path)

        logger.info("✓ Configuration and logging set up")

    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        sys.exit(1)

    # 2) extract settings
    try:
        tickers = config["data"]["tickers"]

        # paths
        model_dir = Path(config["model"]["output"]["model_output_path"]).expanduser()
        processed_data_dir = Path(
            config["data"]["paths"]["processed_data_dir"]
        ).expanduser()
        figures_dir = Path(config["data"]["paths"]["figures_dir"]).expanduser()

        # backtest settings
        initial_capital = config["backtest"]["initial_capital"]
        position_size = config["backtest"]["position_size"]
        transaction_cost = config["backtest"]["transaction_cost"]
        save_results = config["backtest"].get("save_results", True)

    except KeyError as e:
        logger.error(f"✗ Missing configuration key: {e}")
        sys.exit(1)

    # 3) main execution loop
    success_count = 0

    for ticker in tickers:
        logger.info(f"{'-' * 40}")
        logger.info(f"Processing Ticker: {ticker}")

        try:
            # Construct dynamic paths
            model_path = model_dir / f"{ticker.lower()}_model.pkl"
            data_path = processed_data_dir / f"{ticker}_processed.csv"

            # validate paths
            if not model_path.exists():
                logger.warning(f"Skipping {ticker}: Model not found at {model_path}")
                continue  # skip to next ticker
            if not data_path.exists():
                logger.warning(f"Skipping {ticker}: Data not found at {data_path}")
                continue  # skip to next ticker

            # initialize backtester
            logger.info("Initializing backtester...")
            backtester = ModelBacktester(
                model_path=str(model_path), backtest_data_path=str(data_path)
            )

            # execute backtest
            logger.info(f"Running backtest for {ticker}...")
            backtester.run_backtest(
                initial_capital=initial_capital,
                position_size=position_size,
                transaction_cost=transaction_cost,
            )
            logger.info(f"Completed backtest for {ticker}")

            # retrieve performance metrics
            metrics = backtester.calculate_metrics(initial_capital)

            roi = metrics.get("strategy_return", 0.0)
            sharpe = metrics.get("sharpe_ratio", 0.0)
            logger.info(f"Result: Return={roi:.2f}%, Sharpe={sharpe:.2f}")

            # save artifacts if enabled
            if save_results:
                save_backtest_artifacts(
                    backtester, metrics, figures_dir, ticker, logger, config
                )

            success_count += 1

        except Exception as e:
            logger.error(f"✗ Error backtesting {ticker}: {e}", exc_info=True)
            continue  # proceed to next ticker

    # 4) summarize results
    logger.info(f"{'=' * 40}")
    logger.info(f"Successfully back tested {success_count}/{len(tickers)} tickers.")


if __name__ == "__main__":
    sys.exit(main())
