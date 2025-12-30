#!/usr/bin/env python3
# qusa/scripts/run_model_pipeline.py

"""
Master Pipeline Orchestrator for QUSA.
Runs Feature Engineering, Model Training, Evaluation, and Backtesting
for all tickers defined in config.yaml.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qusa.utils.config import load_config
from qusa.utils.logger import setup_logger

import train_model
import evaluate_model
import run_backtest


def main():
    """
    Main execution script.
    """

    # 1) set up configuration and logging
    try:
        config_path = PROJECT_ROOT / "config.yaml"
        config = load_config(str(config_path))

        log_file = PROJECT_ROOT / "logs" / "full_pipeline.log"
        logger = setup_logger("pipeline_orchestrator", log_file=str(log_file))

        tickers = config["data"]["tickers"]
        skip_backtest = config.get("pipeline", {}).get("skip_backtest", False)

        logger.info("✓ Configuration and logging set up")

    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info(f"STARTING FULL QUSA PIPELINE")
    logger.info(f"Tickers: {tickers}")
    logger.info("=" * 80)

    # 2) pipeline execution loop
    for ticker in tickers:
        logger.inf(f"\n>>> PROCESSING TICKER: {ticker} <<<")

        try:
            # PHASE 1: Training
            ## Note: Requires scripts/run_FE_pipeline.py to be run beforehand
            logger.info(f"PHASE 1: Training Model for {ticker}...")
            train_model.main()

            # PHASE 2: Evaluation
            logger.info(f"PHASE 2: Evaluating Model for {ticker}...")
            evaluate_model.main()

            # PHASE 3: Backtesting
            if not skip_backtest:
                logger.info(f"PHASE 3: Backtesting Model for {ticker}...")
                run_backtest.main()
            else:
                logger.info("Skipping Backtesting as per configuration.")

            logger.info(f"✓ Pipeline successful for {ticker}")

        except Exception as e:
            logger.error(f"✗ Pipeline failed for {ticker}: {e}")
            continue  # proceed to next ticker

    logger.info("\n" + "=" * 80)
    logger.info("FULL PIPELINE EXECUTION COMPLETE")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
