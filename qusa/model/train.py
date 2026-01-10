# qusa/qusa/model/train.py

"""
Train model to predict overnight price direction.
"""

import joblib
import os
import pandas as pd

from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Add Monte Carlo features to SAFE_FEATURES
from qusa.features.monte_carlo import MonteCarloFeatures


# define allowed features for training
SAFE_FEATURES = [
    "52_week_high_proximity",
    "52_week_low_proximity",
    "atr_pct",
    "close_position",
    "rsi",
    "volume_ratio",
    "day_of_week",
    "day_of_month",
    "month_of_year",
    "first_5d_month",
    "final_5d_month",
    "is_monday",
    "is_tuesday",
    "is_wednesday",
    "is_thursday",
    "is_friday",
    "is_jan",
    "is_feb",
    "is_mar",
    "is_apr",
    "is_may",
    "is_jun",
    "is_jul",
    "is_aug",
    "is_sep",
    "is_oct",
    "is_nov",
    "is_dec",
]


# Get MC feature names
MC_FEATURES = MonteCarloFeatures.get_feature_names(horizons=[1, 3, 7, 15, 30, 45])

# Append to SAFE_FEATURES
SAFE_FEATURES.extend(MC_FEATURES)

# confirm no duplicate features
SAFE_FEATURES = list(dict.fromkeys(SAFE_FEATURES))

# define leakage features
CONFOUND_FEATURES = [
    "overnight_delta",  # target feature
    "overnight_delta_pct",  # target feature
    "date",  # not a feature
    "z_score",  # derived from target
    "abnormal",  # derived from target
    "intraday_returns",  # calculated next day
    "intraday_return_strong_positive",  # calculated next day
    "intraday_return_strong_negative",  # calculated next day
]


class OvernightDirectionModel:
    """
    Decision tree model to predict overnight price movement.
    """

    def __init__(self, config=None):
        """
        Class constructor.

        Parameters:
            1) config (dict): Model configuration
        """

        self.config = config
        self.model = None
        self.feature_names = SAFE_FEATURES
        self.trained_date = None
        self.metrics = {}

    @staticmethod
    def load_data(data_path):
        """
        Load and prepare data

        Parameters:
            1) data_path (str): Path to data for model
        """

        print("Loading data...")
        data = pd.read_csv(os.path.expanduser(data_path))

        ###
        # Store positive overnight delta as target feature
        # Drop rows with missing target feature
        # Remove confounding features
        ###

        print("Preparing data...")
        data["target"] = (data["overnight_delta"] > 0).astype(int)
        data = data.dropna(subset=["overnight_delta"])
        data = data.drop(columns=CONFOUND_FEATURES, errors="ignore")

        print(f"✓ Loaded {len(data)} rows")

        return data

    def prepare_features(self, data):
        """
        Prepare features for training from dataset

        Parameters:
            1) data (type): fill here

        Returns:
            1) X (type): fill here
            2) y (type): fill here
        """

        # filter out non-safe features and fill missing values
        X = data[self.feature_names].fillna(0)
        y = data["target"]

        return X, y

    def train(self, X_train, y_train):
        """
        Train model.

        Parameters:
            1) X_train (type): fill here
            2) y_train (type): fill here
        """

        print("Training model...")

        # initialize model
        self.model = DecisionTreeClassifier(
            max_depth=self.config["max_depth"],
            min_samples_leaf=self.config["min_samples_leaf"],
            min_samples_split=self.config["min_samples_split"],
            class_weight=self.config["class_weight"],
            random_state=self.config["random_state"],
        )

        # cross validation
        cv_score = cross_val_score(self.model, X_train, y_train, cv=self.config["cv"])

        print(f"✓ CV accuracy: {cv_score.mean():.3f} (+/- {cv_score.std():.3f})")

        # fit model and store timestamp
        self.model.fit(X_train, y_train)
        self.trained_date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

        print(f"✓ Model trained")
        print(f"  - Tree depth: {self.model.get_depth()}")
        print(f"  - Leaves: {self.model.get_n_leaves()}")

        return self

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Parameters:
            1) X_test (type): fill here
            2) y_test (type): fill here
        """

        print("\nEvaluating model...")

        # predict labels for test set and store probabilities
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:1]

        # calculate performance metrics and store as attribute
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        self.metrics = {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        }

        print(f"✓ Test accuracy: {accuracy:.3f}")
        print(f"\nConfusion Matrix:")
        print(cm)

        # store feature importance
        importance = pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)

        print(f"\nTop 5 Important Features:")

        for ft, imp in importance.items():
            print(f"  {ft:30s}: {imp:.4f}")

        self.metrics["feature_importance"] = importance.to_dict()

        return self.metrics

    def save_model(self, save_path):
        """
        Save model bundle to input path.

        Parameters:
            1) save_path (str): Path to save bundle

        Returns:
            1) save_path (str): Path to save bundle
        """

        bundle = {
            "model": self.model,
            "features": self.feature_names,
            "threshold": self.config["probability_threshold"],
            "target": "overnight_delta_positive",
            "trained_date": self.trained_date,
            "config": self.config,
            "metrics": self.metrics,
        }

        # confirm path to save model bundle exists
        save_path = os.path.expanduser(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(bundle, save_path)

        print(f"\n✓ Model saved to: {save_path}")

        return save_path


def train_model(data_path, save_path, config=None):
    """
    Train overnight delta prediction model.

    Parameters:
        1) data_path (str): Path to processed data for training
        2) save_path (str): Path to save model
        3) config (dict): Model configuration
    """

    print("=" * 80)
    print("OVERNIGHT DIRECTION MODEL TRAINING")
    print("=" * 80)

    # initialize model
    model = OvernightDirectionModel(config=config)

    # load data
    data = model.load_data(data_path)

    # prepare features
    X, y = model.prepare_features(data)

    # split dataset into train/test sets without shuffling
    test_size = model.config["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # train, evaluate, save model
    model.train(X_train, y_train)
    model.evaluate(X_test, y_test)
    model.save_model(save_path)

    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)

    return model
