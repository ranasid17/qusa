# qusa/qusa/model/interpreter.py
"""
Interpret and explain trained models.
"""

import joblib
import numpy as np
import pandas as pd

from pathlib import Path


class ModelInterpreter:
    """
    Extract insights and explanations from trained models.
    """

    def __init__(self, model_path, config=None):
        """
        Initialize interpreter with model and optional config.

        Parameters:
            1) model_path (str): Path to saved model bundle
            2) config (dict, optional): Configuration dictionary
        """

        self.model_path = Path(model_path).expanduser().resolve()
        self._load_model()

        if config:
            interp_config = config.get("interpretation", {})
            self.top_n_features = interp_config.get("top_n_features", 10)
            self.max_decision_depth = interp_config.get("max_decision_depth", 10)
            self.low_importance_threshold = interp_config.get(
                "low_importance_threshold", 0.01
            )
            self.high_correlation_threshold = interp_config.get(
                "high_correlation_threshold", 0.9
            )
            self.low_confidence_threshold = interp_config.get(
                "low_confidence_threshold", 0.6
            )

        else:
            print("Pass config to initialize object")

    def _load_model(self):
        """
        Load the trained model from the specified path.
        """

        # load the model bundle
        bundle = joblib.load(self.model_path)

        # extract components and store as attributes
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.threshold = bundle.get("threshold", 0.6)
        self.trained_date = bundle.get("trained_date", "Unknown")
        self.metrics = bundle.get("metrics", {})

        return

    def analyze_feature_importance(self, data=None):
        """
        Analyze and return the top N features by importance.

        Parameters:
            1) data (pd.DataFrame, optional): Data for correlation analysis

        Returns:
            1) analysis (dict): Feature importance analysis
        """

        if not hasattr(self.model, "feature_importances_"):
            return {"error": "Model does not support feature importance"}

        # store feature importance from model
        importance = pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)

        # Identify low-importance features
        low_importance = importances[importances < self.low_importance_threshold]

        return

    @staticmethod
    def _categorize_features(features):
        """
        Group features by type.

        Parameters:
            1) features (list): List of feature names.

        Returns:
            1) categories (dict): Dict of feature categories mapping to feature names.
        """

        # define feature categories
        categories = {
            "technical_indicators": [],
            "calendar": [],
            "volatility": [],
            "momentum": [],
            "other": [],
        }

        # iterate across feature names and assign to categories
        for ft in features:
            if any(x in ft for x in ["rsi", "volume", "proximity"]):
                categories["technical_indicators"].append(ft)
            elif any(x in ft for x in ["day_", "month_", "is_"]):
                categories["calendar"].append(ft)
            elif any(x in ft for x in ["atr", "volatility"]):
                categories["volatility"].append(ft)
            elif any(x in ft for x in ["close_position", "intraday"]):
                categories["momentum"].append(ft)
            else:
                categories["other"].append(ft)

        return {k: v for k, v in categories.items() if v}

    @staticmethod
    def _calculate_concentration(importance):
        """
        Calculate importance concentration metric.

        Parameters:
            1) importance (pd.Series): Series of feature importance.

        Returns:
            1) concentration (dict): Concentration metrics.
        """

        # sort and calculate cumulative sum of importance
        importance_sorted = importance.sort_values(ascending=False)
        cum_sum = importance_sorted.cumsum()
        importance_total = importance_sorted.sum()

        concentration = {
            "top_1_pct": (
                float(importance_sorted.iloc[0] / importance_total)
                if importance_total > 0
                else 0
            ),
            "top_3_pct": (
                float(cum_sum.iloc[:3].iloc[-1] / importance_total)
                if (importance_total > 0) and (len(cum_sum) >= 3)
                else 0
            ),
            "top_5_pct": (
                float(cum_sum.iloc[:5].iloc[-1] / importance_total)
                if (importance_total > 0) and (len(cum_sum) >= 5)
                else 0
            ),
            "top_10_pct": (
                float(cum_sum.iloc[:10].iloc[-1] / importance_total)
                if (importance_total > 0) and (len(cum_sum) >= 10)
                else 0
            ),
        }

        return concentration

    def extract_decision_rules(self, max_depth):
        """
        Extract human-readable decision rules from the model.

        Parameters:
            1) max_depth (int): Maximum depth of rules to extract.

        Returns:
            1) rules (str): Text representation of decision rules.
        """

        if not hasattr(self.model, "tree_"):
            raise ValueError("Model does not support decision rule extraction.")

        try:
            rules = export_text(
                self.model, feature_names=self.features, max_depth=max_depth
            )
            return rules
        except Exception as e:
            raise RuntimeError(f"Error extracting decision rules: {e}")

    def identify_model_limitations(self, top_n):
        """
        Identify potential limitations and biases in the model.

        Parameters:
            1) top_n (int): Number of top features to consider for analysis.

        Returns:
            1) limitations (dict): Dict of identified model limitations.
        """

        # define limitations dict
        limitations = {"warnings": [], "recommendations": [], "risk_factors": []}

        architecture = self.extract_model_architecture()
        feature_analysis = self.analyze_feature_importance(top_n)

        # check for overfitting risk
        if architecture.get("max_depth", 0) > 10:
            limitations["warnings"].append(
                f"Deep tree (depth={architecture['max_depth']}) may overfit to training data"
            )

        if architecture.get("n_leaves", 0) > 100:
            limitations["warnings"].append(
                f"High leaf count ({architecture['n_leaves']}) suggests overfitting risk"
            )

        # check feature usage
        unused_count = feature_analysis.get("n_features_unused", 0)
        unused_pct = unused_count / len(self.features) * 100

        if unused_pct > 50:
            limitations["warnings"].append(
                f"{unused_pct:.0f}% of features unused - consider feature selection"
            )

        # check feature concentration
        concentration = feature_analysis.get("importance_concentration", {})
        if concentration.get("top_1_pct", 0) > 0.5:
            limitations["warnings"].append(
                f"Single feature dominates ({concentration['top_1_pct']:.1%}) - model may be fragile"
            )

        # check class imbalance
        if self.config.get("class_weight") != "balanced":
            limitations["risk_factors"].append(
                "Class weights not balanced - may bias toward majority class"
            )

        # check sample size
        min_leaf = getattr(self.model, "min_samples_leaf", 1)
        if min_leaf < 10:
            limitations["warnings"].append(
                f"Small min_samples_leaf ({min_leaf}) may cause overfitting"
            )

        # generate recommendations
        if limitations["warnings"]:
            limitations["recommendations"].extend(
                [
                    "Consider cross-validation to verify generalization",
                    "Test on out-of-sample data from different market regimes",
                    "Monitor performance degradation over time",
                ]
            )

        if concentration.get("top_3_pct", 0) < 0.3:
            limitations["recommendations"].append(
                "Low feature concentration suggests good feature diversity"
            )

        return limitations

    def analyze_prediction_patterns(self, data_path):
        """
        Analyze how model makes predictions on given dataset.

        Parameters:
            1) data_path (str): Path to the dataset CSV file.

        Returns:
            1) analysis (dict): Dict of prediction pattern analysis.
        """

        try:
            # load dataset
            data = pd.read_csv(os.path.expanduser(data_path))
            x = data[self.features].fillna(0)

            # generate predictions and probabilities
            predictions = self.model.predict(x)
            probabilities = self.model.predict_proba(x)[:, 1]

            # store analysis results in dict
            analysis = {
                "prediction_distribution": {
                    "positive_pct": float(predictions.mean()),
                    "negative_pct": float(1 - predictions.mean()),
                },
                "confidence_distribution": {
                    "high_confidence_pct": float(
                        ((probabilities > 0.7) | (probabilities < 0.3)).mean()
                    ),
                    "low_confidence_pct": float(
                        ((probabilities >= 0.4) & (probabilities <= 0.6)).mean()
                    ),
                },
                "probability_stats": {
                    "mean": float(probabilities.mean()),
                    "std": float(probabilities.std()),
                    "min": float(probabilities.min()),
                    "max": float(probabilities.max()),
                },
            }

            return analysis

        except Exception as e:
            return {"error": f"Error analyzing prediction patterns: {e}"}

    def generate_interpretation_summary(self, data_path, top_n, max_depth):
        """
        Generate complete model interpretation summary.

        Parameters:
            1) data_path (str): Path to dataset CSV for prediction pattern analysis.
            2) top_n (int): Number of top features to consider for analysis.
            3) max_depth (int): Maximum depth of decision rules to extract.

        Returns:
            1) summary (dict): Comprehensive model interpretation summary.
        """

        summary = {
            "model_architecture": self.extract_model_architecture(),
            "feature_analysis": self.analyze_feature_importance(top_n=top_n),
            "decision_rules": self.extract_decision_rules(max_depth=max_depth),
            "limitations": self.identify_model_limitations(top_n=top_n),
            "training_info": {
                "trained_date": self.trained_date,
                "config": self.config,
                "metrics": self.metrics,
            },
        }

        if data_path:
            summary["prediction_patterns"] = self.analyze_prediction_patterns(
                data_path=data_path
            )

        return summary
