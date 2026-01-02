#!/usr/bin/env python3
# scripts/model_evaluation.py
"""
Evaluate trained model on test data.
"""

import sys

from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from qusa.model import evaluate_model, generate_evaluation_report
from qusa.utils.config import load_config
from qusa.utils.logger import setup_logger


def main():
    """
    Main execution script.
    """

    # 1) set up configuration and logging
    try:
        config_path = PROJECT_ROOT / "qusa" / "utils" / "config.yaml"
        config = load_config(str(config_path))

        # extract log path from config
        log_path = config.get("prediction", {}).get("log_file", "logs/evaluation.log")
        logger = setup_logger("model_evaluation", log_path)

        logger.info("✓ Configuration and logging set up")

    except FileNotFoundError as e:
        print(f"✗ Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        sys.exit(1)

    # 2) store key parameters
    try:
        tickers = config["data"]["tickers"]
        model_dir = Path(config["model"]["output"]["model_output_path"]).expanduser()
        eval_data_dir = Path(config["data"]["paths"]["processed_data_dir"]).expanduser()

        # LLM config
        llm_config = config.get("llm", {})
        enable_reports = llm_config.get("enable_reports", True)
        llm = llm_config.get("model_name", "gemma3:4b")
        temp = llm_config.get("temperature", 0.2)
        max_context_rows = llm_config.get("max_context_rows", 100)
        base_output_dir = Path(
            llm_config.get("output_dir", "~/Projects/qusa/reports")
        ).expanduser()
        subdir = llm_config.get("report_subdir", {}).get("evaluation", "evaluation")

        report_output_dir = base_output_dir / subdir

        logger.info(f"Starting evaluation for {len(tickers)} tickers: {tickers}")
        logger.info(f"LLM Reports: {'Enabled' if enable_reports else 'Disabled'}")

    except KeyError as e:
        logger.error(f"✗ Missing configuration key: {e}")
        sys.exit(1)

    # 3) main evaluation loop
    success_count = 0

    for ticker in tickers:
        logger.info(f"{'=' * 40}")
        logger.info(f"Processing Ticker: {ticker}")

        try:
            # build dynamic paths from pre-defined structure
            ## structure: {ticker.lower()}_model.pkl and {ticker}_processed.csv
            model_path = model_dir / f"{ticker.lower()}_model.pkl"
            eval_data_path = eval_data_dir / f"{ticker}_processed.csv"

            # validate paths before calling evaluation
            if not model_path.exists():
                logger.warning(f"Skipping {ticker}: Model not found at {model_path}")
                continue  # skip to next ticker
            if not eval_data_path.exists():
                logger.warning(f"Skipping {ticker}: Model not found at {model_path}")
                continue  # skip to next ticker

            # evaluate model
            logger.info(f"Loading model from: {model_path.name}")
            metrics = evaluate_model(
                model_path=str(model_path), eval_data_path=str(eval_data_path)
            )
            logger.info(f"Evaluation results: {metrics}")

            # generate AI report
            if enable_reports:
                try:
                    logger.info("Generating AI-powered evaluation report...")

                    report = generate_evaluation_report(
                        metrics=metrics,
                        ticker=ticker,
                        llm_name=llm,
                        output_dir=str(report_output_dir),
                        temperature=temp,
                        max_context_rows=max_context_rows,
                    )
                    logger.info("✓ Evaluation report generated")

                except Exception as e:
                    logger.warning(f"⚠ Report generation failed: {e}")
                    logger.debug("Full traceback:", exc_info=True)

                success_count += 1

        except Exception as e:
            logger.error(f"✗ Error evaluating {ticker}: {str(e)}", exc_info=True)
            continue  # proceed to next ticker

    # 4) summarize results
    logger.info(f"{'=' * 40}")
    logger.info(f"Successfully evaluated {success_count}/{len(tickers)} tickers.")


if __name__ == "__main__":
    sys.exit(main())
