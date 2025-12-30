#!/usr/bin/env python3
# qusa/scripts/train_model.py

"""
Train overnight delta direction model.
"""

import os
import sys

from pathlib import Path

# Add the parent directory to the system path to import qusa package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from qusa.model import train_model
from qusa.utils.config import load_config
from qusa.utils.logger import setup_logger


def main():
    """
    Main function to train model.
    """

    # 1) set up configuration and logging
    try:
        config_path = PROJECT_ROOT / "qusa" / "utils" / "config.yaml"
        config = load_config(str(config_path))

        log_dir = config["data"]["paths"].get("predictions_dir", "logs")
        log_file = Path(log_dir).expanduser() / "training_pipeline.log"
        logger = setup_logger("trainer", log_file=str(log_file))

        logger.info("Configuration loaded successfully.")

    except KeyError as e:
        print(f"✗ Missing configuration key: {e}")
        sys.exit(1)

    # 2) extract training settings
    try:
        tickers = config["data"]["tickers"]

        # paths
        model_dir = Path(config["model"]["output"]["model_output_path"]).expanduser()
        processed_data_dir = Path(
            config["data"]["paths"]["processed_data_dir"]
        ).expanduser()
        model_output_dir = Path(
            config["model"]["output"]["model_output_path"]
        ).expanduser()

        # confirm directories exist
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # build model training parameter dictionary
        model_params = config["model"]["parameters"]
        model_config = {
            "max_depth": model_params.get("max_depth", 5),
            "min_samples_leaf": model_params.get("min_samples_leaf", 10),
            "min_samples_split": model_params.get("min_samples_split", 20),
            "class_weight": model_params.get("class_weight", "balanced"),
            "random_state": model_params.get("random_state", 42),
            "test_size": model_params.get("test_size", 0.25),
            "cv": model_params.get("cv", 5),
            "probability_threshold": model_params.get("probability_threshold", 0.6),
        }

        logger.info(f"Training Tickers: {tickers}")
        logger.info(f"Model Parameters: {model_config}")

    except KeyError as e:
        logger.error(f"✗ Missing configuration key: {e}")
        sys.exit(1)

    # 3) main training loop
    success_count = 0

    for ticker in tickers:
        logger.info(f"{'-' * 40}")
        logger.info(f"Processing Ticker: {ticker}")

        try:
            # Construct dynamic paths
            model_save_path = model_output_dir / f"{ticker.lower()}_model.pkl"
            data_path = processed_data_dir / f"{ticker}_processed.csv"

            # validate paths
            if not data_path.exists():
                logger.warning(f"Skipping {ticker}: Data not found at {data_path}")
                continue  # skip to next ticker

            # train model
            ## Note: train_model function handles its own internal prints,
            ##       but we wrap it to catch logic or data errors.
            model = train_model(
                data_path=str(data_path),
                save_path=str(model_save_path),
                config=model_config,
            )

            # log metrics
            metrics = getattr(model, "metrics", {})
            acc = metrics.get("accuracy", 0.0)
            logger.info(f"Model training metrics for {ticker}: Accuracy = {acc:.4f}")
            logger.info(f"   Model saved to: {model_save_path}")

            success_count += 1

        except Exception as e:
            logger.error(f"✗ Error training model for {ticker}: {e}")
            continue  # proceed to next ticker

    # 4) summary
    logger.info("=" * 80)
    logger.info(
        f"BATCH TRAINING COMPLETE: {success_count}/{len(tickers)} models trained."
    )
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
