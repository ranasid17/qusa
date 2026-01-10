import pandas as pd
from ..model.monte_carlo_stock_model import get_mc_features_multiple_horizons


class MonteCarloFeatures:
    """
    Generate Monte Carlo simulation features for stock price forecasting.
    """
    
    def __init__(self, config=None):
        """
        Initialize Monte Carlo feature generator.
        
        Parameters:
            config (dict): Configuration dictionary with MC settings
        """
        self.config = config or {}
        self.horizons = self.config.get("horizons", [1, 3, 7, 15, 30, 45])
        self.iterations = self.config.get("iterations", 1000)
        self.start_date = self.config.get("start_date", "2020-01-01")
    
    def add_all(self, df, ticker):
        """
        Add all Monte Carlo features to the DataFrame.
        
        Parameters:
            df (pd.DataFrame): Stock data DataFrame
            ticker (str): Stock ticker symbol
        
        Returns:
            pd.DataFrame: DataFrame with MC features added
        """
        df_mod = df.copy()
        
        # Get all MC features for all horizons
        mc_features = get_mc_features_multiple_horizons(
            ticker=ticker,
            start_date=self.start_date,
            horizons=self.horizons,
            iterations=self.iterations
        )
        
        # Add features as columns (same value for all rows)
        for feature_name, feature_value in mc_features.items():
            df_mod[feature_name] = feature_value
        
        return df_mod
    
    @staticmethod
    def get_feature_names(horizons=None):
        """
        Get list of Monte Carlo feature names.
        
        Parameters:
            horizons (list): List of forecast horizons
        
        Returns:
            list: Feature names
        """
        horizons = horizons or [1, 3, 7, 15, 30, 45]
        features = []
        
        for days in horizons:
            features.extend([
                f'mc_{days}d_expected_value',
                f'mc_{days}d_return_pct',
                f'mc_{days}d_prob_breakeven',
                f'mc_{days}d_current_price',
                f'mc_{days}d_q0_5',
                f'mc_{days}d_q1',
                f'mc_{days}d_q5',
                f'mc_{days}d_q10',
                f'mc_{days}d_q25',
                f'mc_{days}d_q50',
                f'mc_{days}d_q75',
                f'mc_{days}d_q95',
            ])
        
        return features