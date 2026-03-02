#!/usr/bin/env python3
# scripts/add_MC_features.py

"""
Add Monte Carlo features to processed data.
Runs after run_FE_pipeline.py to enhance existing features.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qusa.features.monte_carlo import MonteCarloFeatures
from qusa.utils.config import load_config
from qusa.utils.logger import setup_logger
import pandas as pd


def main():
    """
    Main execution function.
    """

    # Setup logging
    logger = setup_logger(
        "add_mc_features", log_file=str(PROJECT_ROOT / "logs" / "add_mc_features.log")
    )

    print("=" * 80)
    print("ADDING MONTE CARLO FEATURES")
    print("=" * 80)

    # Load configuration
    try:
        config_path = PROJECT_ROOT / "qusa" / "utils" / "config.yaml"
        config = load_config(str(config_path))
        logger.info("✓ Configuration loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load configuration: {e}")
        return 1

    # Check if MC features are enabled
    mc_config = config.get("monte_carlo", {})
    if not mc_config.get("enabled", False):
        print("\n⚠ Monte Carlo features disabled in config")
        logger.info("Monte Carlo features disabled - exiting")
        return 0

    # Display configuration
    print("\nConfiguration:")
    print(f"  Window size: {mc_config.get('window_size', 252)} days")
    print(f"  Iterations: {mc_config.get('iterations', 1000):,}")
    print(f"  Random seed: {mc_config.get('random_seed', 42)}")
    print(f"  Min threshold: {mc_config.get('min_data_threshold', 252)} rows")
    print(f"  Features: {len(mc_config.get('features', []))}")

    logger.info(
        f"MC Config: window={mc_config.get('window_size')}, "
        f"iterations={mc_config.get('iterations')}, "
        f"seed={mc_config.get('random_seed')}"
    )

    # Get tickers and paths
    try:
        tickers = config["data"]["tickers"]
        processed_dir = Path(config["data"]["paths"]["processed_data_dir"]).expanduser()
        logger.info(f"Processing {len(tickers)} ticker(s): {tickers}")
    except KeyError as e:
        logger.error(f"✗ Missing configuration key: {e}")
        return 1

    # Initialize MC feature calculator
    mc_calculator = MonteCarloFeatures(config=mc_config)

    # Process each ticker
    success_count = 0

    for ticker in tickers:
        print(f"\n{'─' * 80}")
        print(f"Processing: {ticker}")
        print("─" * 80)

        try:
            # Load processed data
            data_path = processed_dir / f"{ticker}_processed.csv"

            if not data_path.exists():
                logger.warning(f"Skipping {ticker}: File not found at {data_path}")
                print(f"⚠ File not found: {data_path}")
                continue

            print(f"\nLoading processed data...")
            df = pd.read_csv(data_path)
            df["date"] = pd.to_datetime(df["date"])

            original_rows = len(df)
            original_cols = len(df.columns)
            print(f"✓ Loaded {data_path.name}")
            print(f"  Rows: {original_rows:,}")
            print(f"  Columns: {original_cols}")

            logger.info(
                f"Loaded {ticker}: {original_rows} rows, {original_cols} columns"
            )

            # Calculate MC features
            print(f"\nCalculating Monte Carlo features...")
            start_time = datetime.now()

            df_enhanced = mc_calculator.add_mc_features(df, price_col="close")

            elapsed = (datetime.now() - start_time).total_seconds()

            # Validate features
            validation = mc_calculator.validate_features(df_enhanced)

            print(f"✓ MC features calculated in {elapsed:.1f}s")
            print(f"\nValidation:")
            print(f"  Total rows: {validation['total_rows']:,}")
            print(f"  Valid MC rows: {validation['valid_rows']:,}")
            print(f"  NaN rows (threshold): {validation['nan_rows']:,}")

            if validation["errors"]:
                print(f"\n⚠ Validation warnings:")
                for error in validation["errors"]:
                    print(f"    - {error}")
                    logger.warning(f"{ticker}: {error}")
            else:
                print(f"  ✓ No validation errors")

            logger.info(
                f"{ticker}: {validation['valid_rows']} valid MC rows, "
                f"{validation['nan_rows']} NaN rows"
            )

            # Print feature summary
            mc_calculator.print_feature_summary(df_enhanced)

            # Save enhanced data
            print(f"\nSaving enhanced data...")
            df_enhanced.to_csv(data_path, index=False)

            new_cols = len(df_enhanced.columns)
            added_cols = new_cols - original_cols

            print(f"✓ Saved to {data_path.name}")
            print(f"  Rows: {len(df_enhanced):,} (unchanged)")
            print(f"  Columns: {new_cols} (+{added_cols} MC features)")

            logger.info(f"✓ {ticker} complete: {added_cols} features added")

            success_count += 1

        except Exception as e:
            logger.error(f"✗ Error processing {ticker}: {e}", exc_info=True)
            print(f"\n✗ Error: {e}")
            continue

    # Summary
    print("\n" + "=" * 80)
    if success_count == len(tickers):
        print("✓ MC FEATURES ADDED SUCCESSFULLY")
        logger.info(f"✓ All {success_count} ticker(s) processed successfully")
    else:
        print(f"⚠ COMPLETED WITH WARNINGS: {success_count}/{len(tickers)} successful")
        logger.warning(f"Only {success_count}/{len(tickers)} tickers processed")
    print("=" * 80)

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
