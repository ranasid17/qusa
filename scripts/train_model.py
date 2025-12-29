# qusa/scripts/train_model.py

"""
Train overnight delta direction model.

Usage:
    python scripts/train_model.py --ticker AMZN
    python scripts/train_model.py --ticker AAPL --config custom_config.yaml
"""

import os
import sys

from pathlib import Path

# Add the parent directory to the system path to import qusa package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from qusa.model import train_model


def main():
    """
    Main function to train model.
    """
