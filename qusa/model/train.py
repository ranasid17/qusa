# qusa/qusa/model/train.py

"""
Train model to predict overnight price direction. 
"""

import joblib 
import numpy as np 
import os 
import pandas as pd 

from datetime import datetime 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier


# define allowed features for training 
SAFE_FEATURES = [
    '52_week_high_proximity', 
    '52_week_low_proximity',
    'atr_pct', 
    'close_position'
    'rsi', 
    'volume_ratio', 
    'day_of_week', 
    'day_of_month', 
    'month_of_year', 
    'first_5d_month', 
    'final_5d_month',
    'is_monday', 
    'is_tuesday', 
    'is_wednesday', 
    'is_thursday', 
    'is_friday',
    'is_jan', 
    'is_feb', 
    'is_mar', 
    'is_apr", '
    'is_may', 
    'is_jun',
    'is_jul', 
    'is_aug', 
    'is_sep', 
    'is_oct', 
    'is_nov', 
    'is_dec',
]

# confirm no duplicate features 
SAFE_FEATURES = list(dict.fromkeys(SAFE_FEATURES))

# define leakage features 
CONFOUND_FEATURES = [
    'overnight_delta',                      # target feature 
    'overnight_delta_pct',                  # target feature 
    'date',                                 # not a feature 
    'z_score',                              # derived from target 
    'abnormal',                             # derived from target 
    'intraday_returns',                     # calculated next day 
    'intraday_return_strong_positive',      # calculated next day 
    'intraday_return_strong_negative'       # calculated next day 
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
    def load_data(self, data_path): 
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
        data['target'] = data.loc[data['overnight_delta']>0].astype(int)
        data = data.dropna(subset=['overnight_delta'])
        data = data.drop(columns=CONFOUND_FEATURES, errors='ignore')

        print(f"✓ Loaded {len(data)} rows")

        return data 
    

    def prepare_features(self, data): 
        return 