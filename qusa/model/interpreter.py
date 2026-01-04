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

        # label low-importance features
        low_importance = importance[importance < self.low_importance_threshold]

        # assemble into dict
        analysis = {
            "top_features": importance.head(self.top_n_features).to_dict(),
            "all_features": importance.to_dict(),
            "low_importance_features": low_importance.to_dict(),
            "importance_concentration": importance.head(5).sum(),
        }

        # correlation analysis when data provided
        if data is not None:
            # list comprehension to store feature names from input DataFrame
            available_features = [f for f in self.features if f in data.columns]

            # correlate features when more than one feature exists
            if len(available_features) > 1:
                corr_matrix = data[available_features].corr().abs()
            # otherwise return analysis without correlated features
            else:
                return analysis

            highly_correlated_features = []

            # store feature pairs with correlation above threshold
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.high_correlation_threshold:
                        highly_correlated_features.append(
                            {
                                "feature_1": corr_matrix.columns[i],
                                "feature_2": corr_matrix.columns[j],
                                "correlation": corr_matrix.iloc[i, j],
                            }
                        )
            # add correlated features as (key, value) pair in dict
            analysis["high_correlation_pairs"] = highly_correlated_features

        return analysis

    def extract_decision_rules(self):
        """
        Extract human-readable decision rules from the model.

        Returns:
            1) rules (list): Decision rules with conditions and outcomes
        """

        if not hasattr(self.model, "tree_"):
            raise ValueError("Model does not support decision rule extraction.")

        # store model architecture and features
        tree = self.model.tree_
        feature_names = self.features

        rules = []

        def recurse(node, depth, conditions):
            """
            Internal method to recursively analyze tree.

            Parameters:
                1) node (int): xxx
                2) depth (int): xxx
                3) conditions (list): xxx
            """

            # limit depth to avoid complex rules
            if depth > self.max_decision_depth:
                return

            # record label decision when at leaf node
            if tree.feature[node] == -2:
                class_counts = tree.value[node][0]
                predicted_class = np.argmax(class_counts)
                confidence = class_counts[predicted_class] / class_counts.sum()

                rules.append(
                    {
                        "conditions": conditions.copy(),
                        "prediction": "UP" if predicted_class == 1 else "DOWN",
                        "confidence": float(confidence),
                        "samples": int(class_counts.sum()),
                    }
                )
                return

            # split at internal node
            feature = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]

            # recursively store decision rules for left branches
            left_conditions = conditions + [f"{feature} <= {threshold:.3f}"]
            recurse(tree.children_left[node], depth + 1, left_conditions)

            # recursively store decision rules for right branches
            right_conditions = conditions + [f"{feature} > {threshold:.3f}"]
            recurse(tree.children_right[node], depth + 1, right_conditions)

        # start recursive tree analysis at top level node
        recurse(0, 0, [])

        # Sort by sample size (most common paths first)
        rules.sort(key=lambda x: x["samples"], reverse=True)

        return rules

    def identify_model_limitations(self, evaluation_metrics=None):
        """
        Identify potential model weaknesses, limitations, and biases.

        Parameters:
            1) evaluation_metrics (dict, optional): Test set evaluation metrics

        Returns:
            1) limitations (dict): Identified limitations and concerns
        """

        # define limitations dict
        limitations = {
            "low_importance_features": [],
            "performance_concerns": [],
            "confidence_issues": [],
            "data_quality_concerns": [],
        }

        # interpret feature importance
        if hasattr(self.model, "feature_importances_"):
            # extract importance as a Series and filter to low importance
            importance = pd.Series(self.model.feature_importances_, index=self.features)
            low_imp = importance[importance < self.low_importance_threshold]

            # determine if over 50% of features have low importance
            if len(low_imp) > len(self.features) * 0.5:
                limitations["low_importance_features"].append(
                    f"{len(low_imp)} features have very low importance (<{self.low_importance_threshold})"
                )

        # interpret performance metrics
        if evaluation_metrics:
            accuracy = evaluation_metrics.get("accuracy", 0)

            # handle model with low accuracy
            if accuracy < 0.55:
                limitations["performance_concerns"].append(
                    f"Low accuracy ({accuracy:.3f}) suggests weak predictive power"
                )

            high_confidence_coverage = evaluation_metrics.get(
                "high_confidence_coverage", 0
            )

            # handle model with low coverage
            if high_confidence_coverage < 0.3:
                limitations["confidence_issues"].append(
                    f"Low high-confidence coverage ({high_confidence_coverage:.1%}) limits actionable predictions"
                )

            high_confidence_accuracy = evaluation_metrics.get(
                "high_confidence_accuracy", 0
            )

            # handle model with low accuracy in confident predictions
            if high_confidence_accuracy < self.low_confidence_threshold:
                limitations["confidence_issues"].append(
                    f"High-confidence predictions have low accuracy ({high_confidence_accuracy:.3f})"
                )

        # interpret tree-specific limitations
        if hasattr(self.model, "tree_"):
            # store tree parameters
            depth = self.model.get_depth()
            leaves = self.model.get_n_leaves()

            if depth < 2:
                limitations["data_quality_concerns"].append(
                    "Shallow tree suggests limited signal in features"
                )

            elif depth > 15:
                limitations["data_quality_concerns"].append(
                    "Shallow tree suggests limited signal in features"
                )

            if leaves < 5:
                limitations["data_quality_concerns"].append(
                    "Few leaf nodes suggest model may be too simple"
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

    def generate_interpretation_summary(self, data=None, evaluation_metrics=None):
        """
        Generate comprehensive model interpretation report.

        Parameters:
            1) data (pd.DataFrame, optional): Training/test data for analysis
            2) evaluation_metrics (dict, optional): Evaluation metrics

        Returns:
            1) summary (dict): Complete interpretation summary
        """

        # build dictionary holding results
        summary = {
            "model_info": {
                "model_path": str(self.model_path),
                "trained_date": self.trained_date,
                "n_features": len(self.features),
                "prediction_threshold": self.threshold,
            },
            "feature_importance": self.analyze_feature_importance(data),
            "decision_rules": self.extract_decision_rules(),
            "limitations": self.identify_model_limitations(evaluation_metrics),
            "training_metrics": self.metrics,
        }

        return summary

    def print_interpretation(self, data=None, evaluation_metrics=None):
        """
        Print human-readable interpretation to console.

        Parameters:
            1) data (pd.DataFrame, optional): Training/test data
            2) evaluation_metrics (dict, optional): Evaluation metrics
        """

        # call method to assemble summary results
        summary = self.generate_interpretation_summary(data, evaluation_metrics)

        print("\n" + "=" * 80)
        print("MODEL INTERPRETATION")
        print("=" * 80)

        # Model Info
        print(f"\nModel: {summary['model_info']['model_path']}")
        print(f"Trained: {summary['model_info']['trained_date']}")
        print(f"Features: {summary['model_info']['n_features']}")

        # Feature Importance
        print("\n" + "-" * 80)
        print(f"TOP {self.top_n_features} FEATURES")
        print("-" * 80)
        for feat, imp in list(summary["feature_importance"]["top_features"].items())[
            : self.top_n_features
        ]:
            print(f"  {feat:30s}: {imp:.4f}")

        # Decision Rules (top 5)
        print("\n" + "-" * 80)
        print("KEY DECISION RULES")
        print("-" * 80)
        for i, rule in enumerate(summary["decision_rules"][:5], 1):
            print(
                f"\nRule {i} ({rule['samples']} samples, {rule['confidence']:.1%} confidence):"
            )
            print(f"  Prediction: {rule['prediction']}")
            print(f"  Conditions:")
            for cond in rule["conditions"]:
                print(f"    - {cond}")

        # Limitations
        print("\n" + "-" * 80)
        print("MODEL LIMITATIONS")
        print("-" * 80)

        # extract limitations section
        limitations = summary["limitations"]
        has_issues = False

        # iterate across issues if exist and re-format
        for category, issues in limitations.items():
            if issues:
                has_issues = True
                print(f"\n{category.replace('_', ' ').title()}:")
                for issue in issues:
                    print(f"  ⚠ {issue}")

        if not has_issues:
            print("  ✓ No major limitations identified")

        print("\n" + "=" * 80)

        return
