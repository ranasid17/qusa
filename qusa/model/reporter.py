# qusa/qusa/model/reporter.py

"""
Generate intelligent reports using local LLM for trading system analysis.
Includes deep model interpretation, feature analysis, and limitation detection.
"""
import json
import joblib
import ollama
import os
import pandas as pd

from datetime import datetime
from pathlib import Path
from sklearn.tree import export_text
from typing import Literal


class ModelInterpreter:
    """
    Deep dive into model internals, feature importance, and decision logic.
    """

    def __init__(self, model_path):
        """
        Initialize the ModelInterpreter with a trained model.

        Parameters:
            1) model_path (str): Path to the trained model file.
        """

        self.model_path = os.path.expanduser(model_path)
        self._load_model()

    def _load_model(self):
        """
        Load the trained model from the specified path.
        """

        # load the model bundle
        bundle = joblib.load(self.model_path)

        # extract components and store as attributes
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.config = bundle.get("config", {})
        self.metrics = bundle.get("metrics", {})
        self.trained_date = bundle.get("trained_date", "Unknown")

        return

    def extract_model_architecture(self):
        """
        Extract and return the model architecture as dictionary.

        Returns:
            1) architecture (dict): Dict representation of the model architecture.
        """

        # expect a decision tree model
        if hasattr(self.model, "tree_"):
            tree = self.model.tree_
            architecture = {
                "model_type": "DecisionTreeClassifier",
                "max_depth": self.model.get_depth(),
                "n_leaves": self.model.get_n_leaves(),
                "n_nodes": tree.node_count,
                "n_features_used": len([f for f in tree.feature if f >= 0]),
                "total_features": len(self.features),
                "hyperparameters": {
                    "max_depth": self.model.max_depth,
                    "min_samples_leaf": self.model.min_samples_leaf,
                    "min_samples_split": self.model.min_samples_split,
                    "class_weight": self.model.class_weight,
                },
            }
            return architecture

        # otherwise, return model type only
        else:
            return {"model_type": type(self.model).__name__}

    def analyze_feature_importance(self, top_n):
        """
        Analyze and return the top N features by importance.

        Parameters:
            1) top_n (int): Number of top features to return.

        Returns:
            1) top_features (dict): Dict of tuples (feature_name, importance_score).
        """

        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model does not support feature importance analysis.")

        # Gini importance
        importance = pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)

        # calculate feature statistics
        used_features = importance[importance > 0]
        unused_features = importance[importance <= 0]

        # label feature categories
        feature_categories = self._categorize_features(self.features)
        category_importance = {}

        for cat, ft in feature_categories.items():
            cat_importance = importance[importance.index.isin(ft.index)].sum()
            category_importance[cat] = float(cat_importance)

        top_features = {
            "top_features": importance.head(top_n).to_dict(),
            "n_features_used": len(used_features),
            "n_features_unused": len(unused_features),
            "unused_features": unused_features.index.tolist(),
            "category_importance": category_importance,
            "importance_concentration": self._calculate_concentration(importance),
        }

        return top_features

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
            if any(x in ft for x in ["rsi", "atr", "volume", "proximity"]):
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
        unused_pct = feature_analysis.get("unused_pct", 0) / len(self.features) * 100
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
        min_leaf = self.model.get("hyperparameters", {}).get("min_samples_leaf", 1)
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


class StrategyReporter:
    """
    Generate intelligent reports for trading strategies using local LLM.
    """

    # Define report types for class to interpret
    REPORT_TYPES = Literal[
        "backtest",
        "evaluation",
        "training",
        "model_interpretation",
    ]

    PROMPTS = {
        "backtest": """You are a Senior Quantitative Analyst with 15+ years of experience in algorithmic trading.

    Your task is to analyze the following backtesting results for an overnight gap trading strategy and provide a 
    professional, actionable report.

    **BACKTESTING METRICS:**
    {metrics}

    **STATISTICAL SUMMARY OF RESULTS:**
    {stats_summary}

    **RECENT TRADING ACTIVITY (Last 5 Trades):**
    {recent_trades}

    **INSTRUCTIONS:**
    1. Provide a concise Executive Summary (2-3 sentences)
    2. Analyze the strategy's risk-adjusted performance (Sharpe ratio, draw down, volatility)
    3. Evaluate the trading characteristics (win rate, trade frequency, position sizing)
    4. Identify key strengths and weaknesses
    5. Provide 3-5 specific, actionable recommendations for improvement
    6. Conclude with an overall assessment: Is this strategy viable for live trading?

    **FORMAT:** Use clear Markdown formatting with headers, bullet points, and emphasis where appropriate.
    **TONE:** Professional, data-driven, honest about limitations.
    **LENGTH:** 400-600 words maximum.

    Begin your analysis:""",
        "evaluation": """You are a Machine Learning Engineer specializing in financial prediction models.

    Analyze the following model evaluation metrics for an overnight price direction classifier:

    **MODEL METRICS:**
    {metrics}

    **CONFUSION MATRIX ANALYSIS:**
    - True Negatives: {tn}
    - False Positives: {fp}
    - False Negatives: {fn}
    - True Positives: {tp}

    **HIGH CONFIDENCE PREDICTIONS:**
    - Coverage: {coverage:.1%}
    - Accuracy: {hc_accuracy:.3f}

    **CALIBRATION DATA:**
    {calibration}

    **INSTRUCTIONS:**
    1. Evaluate model accuracy, precision, and recall
    2. Assess the confusion matrix - are there systematic biases?
    3. Analyze calibration - is the model well-calibrated?
    4. Evaluate the high-confidence prediction strategy
    5. Identify overfitting or underfitting concerns
    6. Recommend specific improvements (feature engineering, hyperparameters, etc.)

    **FORMAT:** Markdown with clear sections
    **TONE:** Technical but accessible
    **LENGTH:** 350-500 words

    Begin your analysis:""",
        "training": """You are a Senior ML Engineer reviewing a model training pipeline.

    Analyze the following training results:

    **TRAINING CONFIGURATION:**
    {config}

    **TRAINING METRICS:**
    {metrics}

    **FEATURE IMPORTANCE (Top 10):**
    {feature_importance}

    **CROSS-VALIDATION RESULTS:**
    {cv_results}

    **INSTRUCTIONS:**
    1. Evaluate the model architecture choices (depth, leaf size, etc.)
    2. Assess whether the model is appropriately regularized
    3. Review feature importance - are the right features driving predictions?
    4. Check for signs of overfitting (train vs. test performance)
    5. Recommend hyperparameter adjustments if needed
    6. Suggest additional features or data preprocessing steps

    **FORMAT:** Markdown
    **TONE:** Technical, focused on model improvement
    **LENGTH:** 300-500 words

    Begin your analysis:""",
        "model_interpretation": """You are a Senior ML Engineer and Model Interpretability Specialist.

    Provide a comprehensive interpretation of this machine learning model used for overnight stock price direction prediction.

    **MODEL ARCHITECTURE:**
    {architecture}

    **FEATURE IMPORTANCE ANALYSIS:**
    {feature_importance}

    **TOP DECISION RULES:**
    ```
    {decision_rules}
    ```

    **IDENTIFIED LIMITATIONS:**
    {limitations}

    **PREDICTION PATTERNS:**
    {prediction_patterns}

    **FEATURE CATEGORIES:**
    {feature_categories}

    **INSTRUCTIONS:**
    1. **Model Mechanism**: Explain in plain language HOW this model makes predictions
    2. **Key Drivers**: What are the 3-5 most important features and WHY they matter
    3. **Decision Logic**: Describe the decision-making process using the extracted rules
    4. **Feature Categories**: Which types of features (technical, calendar, momentum) dominate?
    5. **Limitations & Biases**: Identify specific weaknesses, overfitting risks, and blind spots
    6. **Interpretability**: How explainable/transparent is this model for traders?
    7. **Recommendations**: Specific improvements to model architecture, features, or training

    **CRITICAL FOCUS AREAS:**
    - Feature concentration: Is the model too reliant on 1-2 features?
    - Unused features: Why aren't certain features being used?
    - Decision complexity: Is the tree too deep/complex?
    - Generalization concerns: Will this work in different market conditions?

    **FORMAT:** Clear Markdown with sections
    **TONE:** Technical but accessible to quantitative traders
    **LENGTH:** 500-700 words

    Begin your analysis:""",
    }

    def __init__(self, llm_name, output_dir, temperature, max_context_rows):
        """
        Initialize the StrategyReporter with LLM settings.

        Parameters:
            1) llm_name (str): Local LLM model name to use (e.g., "llama2-7b").
            2) output_dir (str): Directory to save generated reports.
            3) temperature (float): Temperature setting for LLM generation.
            4) max_context_rows (int): Max rows of data to include in context.
        """

        self.model_name = llm_name
        self.output_dir = Path(output_dir).expanduser()
        self.temperature = temperature
        self.max_context_rows = max_context_rows

        # confirm output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # confirm Ollama connection
        self._verify_ollama_connection()

    def _verify_ollama_connection(self):
        """
        Verify connection to local Ollama LLM server and model availability.
        """

        try:
            # check server status and model list
            models = ollama.list()
            available = [m["name"] for m in models.get("models", [])]

            # warn if specified model not found
            if not any(self.model_name in m for m in available):
                print(f"⚠ Warning: Model '{self.model_name}' not found.")
                print(f"Available models: {', '.join(available)}")
                print(f"Pull it with: ollama pull {self.model_name}")

        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama. Is it running?\n"
                f"Start with: ollama serve\n"
                f"Error: {e}"
            )

    @staticmethod
    def _prepare_metrics_summary(df):
        """
        Prepare a summary of key metrics from a DataFrame.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing metrics data.

        Returns:
            1) summary (str): Formatted summary string.
        """

        summary_lines = [
            f"**Total Observations:** {len(df)}",
            f"**Date Range:** {df['date'].min()} to {df['date'].max()}",
            "",
            "**Statistical Summary:**",
            "```",
            df.describe().to_string(),
            "```",
        ]

        return "\n".join(summary_lines)

    def _prepare_recent_trades(self, df):
        """
        Extract recent trades from DataFrame.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing trade data.

        Returns:
            1) recent_trades (str): Formatted string of recent trades.
        """

        # store recent trade features
        cols = ["date", "overnight_delta", "strategy_return", "portfolio_value"]
        cols_available = [c for c in cols if c in df.columns]

        recent_trades = df[cols_available].tail(self.max_context_rows)

        return f"```\n{recent_trades.to_string(index=False)}\n```"

    def _generate_prompt(self, report_type, metrics, df, model_interpretation):
        """
        Generate the prompt for the LLM based on report type and data.

        Parameters:
            1) report_type (str): Type of report to generate.
            2) metrics (dict): Metrics data for the report.
            3) df (pd.DataFrame): DataFrame containing relevant data.
            4) model_interpretation (dict): Model interpretation data if applicable.

        Returns:
            1) prompt (str): fill here
        """

        # select prompt template based on report type
        template = self.PROMPTS[report_type]

        # handle backtest report prompt
        if report_type == "backtest":
            prompt = template.format(
                metrics=self._prepare_metrics_summary(metrics),
                stats_summary=self._prepare_metrics_summary(df),
                recent_trades=self._prepare_recent_trades(df),
            )
            return prompt

        # handle evaluation report prompt
        elif report_type == "evaluation":
            prompt = template.format(
                metrics=(self._prepare_metrics_summary(metrics)),
                tn=metrics.get("true_negatives", 0),
                fp=metrics.get("false_positives", 0),
                fn=metrics.get("false_negatives", 0),
                tp=metrics.get("true_positives", 0),
                coverage=metrics.get("high_confidence_coverage", 0),
                hc_accuracy=metrics.get("high_confidence_accuracy", 0),
                calibration=metrics.get("calibration_data", "N/A"),
            )
            return prompt

        # handle training report
        elif report_type == "training":
            feature_importance = metrics.get("feature_importance", {})
            most_important_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            feature_string = "\n".join(
                [f"- {feat}: {imp:.4f}" for feat, imp in most_important_features]
            )

            prompt = template.format(
                config=json.dumps(metrics.get("config", {}), indent=2),
                metrics=self._prepare_metrics_summary(metrics),
                feature_importance=feature_string,
                cv_results=metrics.get("cv_score", "N/A"),
            )

            return prompt

        elif report_type == "model_interpretation":
            if not model_interpretation:
                raise ValueError(
                    "model_interpretation data required for this report type"
                )

            # format feature importance
            feature_analysis = model_interpretation.get("feature_analysis", {})
            top_features = feature_analysis.get("top_features", {})
            feature_string = "\n".join(
                [f"- {k}: {v:.4f}" for k, v in list(top_features.items())[:10]]
            )

            # format categories
            category_importance = feature_analysis.get("category_importance", {})
            category_string = "\n".join(
                [f"- {k}: {v:.2%}" for k, v in category_importance.items()]
            )

            prompt = template.format(
                architecture=json.dumps(
                    model_interpretation.get("architecture", {}), indent=2
                ),
                feature_importance=feature_string,
                decision_rules=json.dumps(
                    model_interpretation.get("decision_rules", "N/A"), indent=2
                ),
                limitations=json.dumps(
                    model_interpretation.get("limitations", {}), indent=2
                ),
                prediction_patterns=json.dumps(
                    model_interpretation.get("prediction_patterns", {}), indent=2
                ),
                feature_categories=category_string,
            )

            return prompt
        else:
            raise ValueError(f"Unsupported report type: {report_type}")

    def _call_llm(self, prompt):
        """
        Pass prompt to Ollama-hosted LLM with error handling.

        Parameters:
            1) prompt (str): pass to LLM
        """

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature, "num_predict": 1000},
            )

            return response["message"]["content"]

        except Exception as e:
            error_msg = f"## LLM Generation Failed\n\n**Error:** {str(e)}\n\n"
            error_msg += "**Fallback:** Using raw metrics summary.\n\n"
            return error_msg

    def generate_report(
        self, report_type, metrics, df, ticker, model_path, data_path, save
    ):
        """
        Generate report using Ollama-hosted LLM.

        Parameters:
            1) report_type (str): Type of report to produce
            2) metrics (dict): Model performance metrics
            3) df (pd.DataFrame): DataFrame holding model results
            4) model_path (str): Path to model (for interpretation report)
            5) data_path (str): Path to data (for interpretation report)
            6) save (bool): Save report to file when True

        Returns:
            1) report (str): Markdown string report generated by LLM
        """

        print(f"\n{'=' * 80}")
        print(f"GENERATING {report_type.upper()} REPORT")
        print(f"{'=' * 80}")
        print(f"Model: {self.model_name}")
        print(f"Ticker: {ticker}")

        # extract detailed analysis for model interpretation
        model_interpretation = None
        if report_type == "model_interpretation" and model_path:
            print("Extracting model interpretation...")
            interpreter = ModelInterpreter(model_path)
            model_interpretation = interpreter.generate_interpretation_summary(
                data_path
            )

        # generate prompt
        prompt = self._generate_prompt(report_type, metrics, df, model_interpretation)

        # call LLM
        print("Calling LLM...")
        llm_analysis = self._call_llm(prompt)

        # build report components
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_components = [
            f"# {report_type.replace('_', ' ').title()} Report: {ticker}",
            f"**Generated:** {timestamp}",
            f"**Model:** {self.model_name}",
            "",
            "---",
            "",
            llm_analysis,
            "",
            "---",
            "",
        ]

        # add model interpretation index when available
        if model_interpretation:
            report_components.extend(
                [
                    "## Appendix: Detailed Model Analysis",
                    "",
                    "### Architecture",
                    "```json",
                    json.dumps(model_interpretation.get("architecture", {}), indent=2),
                    "```",
                    "",
                    "### Top Features by Importance",
                    "```json",
                    json.dumps(
                        dict(
                            list(
                                model_interpretation.get("feature_analysis", {})
                                .get("top_features", {})
                                .items()
                            )
                        ),
                        indent=2,
                    ),
                    "```",
                    "",
                    "### Decision Rules (Top Levels)",
                    "```",
                    model_interpretation.get("decision_rules", "N/A")[:2000],
                    "```",
                    "",
                ]
            )

        # add metrics
        report_components.extend(
            [
                "## Raw Metrics",
                "```json",
                json.dumps(metrics, indent=2, default=str),
                "```",
            ]
        )

        # assemble components into full report
        report = "\n".join(report_components)

        # save when available
        if save:
            filename = (
                f"{ticker}_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            filepath = self.output_dir / filename

            with open(filepath, "w") as f:
                f.write(report)

            print(f"✓ Report saved: {filepath}")

        print(f"{'=' * 80}\n")

        return report


# Convenience functions
def generate_backtest_report(metrics, results_df, ticker, **kwargs):
    """
    Generate backtest report.

    Parameters:
        1) metrics (dict): Dict of model performance metrics
        2) results_df (pd.DataFrame): DataFrame of model outputs
        3) ticker (str): Ticker model trained/evaluated/back-tested against

    Returns:
        1) report (str): Markdown file of report

    """
    reporter = StrategyReporter(**kwargs)
    report = reporter.generate_report("backtest", metrics, results_df, ticker)
    return report


def generate_evaluation_report(metrics, ticker, **kwargs):
    """
    Generate evaluation report.

    Parameters:
        1) metrics (dict): Dict of model performance metrics
        2) ticker (str): Ticker model trained/evaluated/back-tested against

    Returns:
        1) report (str): Markdown file of report
    """
    reporter = StrategyReporter(**kwargs)
    report = reporter.generate_report("evaluation", metrics, None, ticker)
    return report


def generate_training_report(metrics, ticker, **kwargs):
    """
    Generate training report.

    Parameters:
        1) metrics (dict): Dict of model performance metrics
        2) ticker (str): Ticker model trained/evaluated/back-tested against

    Returns:
        1) report (str): Markdown file of report
    """
    reporter = StrategyReporter(**kwargs)
    report = reporter.generate_report("training", metrics, None, ticker)
    return report


def generate_model_interpretation_report(model_path, data_path, ticker, **kwargs):
    """
    Generate deep model interpretation report.

    Parameters:
        model_path: Path to trained model
        data_path: Path to data for analysis
        ticker: Stock ticker

    Returns:
        report (str): Markdown report with model interpretation
    """
    reporter = StrategyReporter(**kwargs)
    report = reporter.generate_report(
        "model_interpretation",
        metrics={},
        ticker=ticker,
        model_path=model_path,
        data_path=data_path,
    )
    return report
