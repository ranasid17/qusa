# qusa/qusa/model/predict.py

"""
Use trained model to make live
predictions on most-recent trading
day data.
"""

import joblib
import os
import pandas as pd

from datetime import datetime


class LivePredictor:
    """
    Make predictions on live market data.
    """

    def __init__(self, model_path):
        """
        Class constructor.

        Parameters;
            1) model_path (str): Path to saved model
        """

        self.model_path = os.path.expanduser(model_path)
        self._load_model()


    def _load_model(self):
        """
        Load trained model.
        """

        bundle = joblib.load(self.model_path)

        self.model = bundle['model']
        self.features = bundle['features']
        self.threshold = bundle['threshold']
        self.trained_date = bundle['trained_date', 'Unknown']

        print(f"✓ Model loaded (trained: {self.trained_date})")

        return


    def predict(self, data):
        """
        Make prediction on most-recent data

        Parameters:
            1) data (pd.DataFrame): Data with features

        Returns:
            1) prediction (dict): Results
        """

        # get latest row
        latest = data.tail(1)

        # extract features
        X = latest[self.features].fillna(0)

        # predict labels and probabilities
        y_pred = self.model.predict(X)[0]
        y_prob = self.model.predict_proba(X)[0,1]

        # interpret prediction
        if y_pred == 1:
            direction = "UP ⬆"
        else:
            direction = "DOWN ⬇"

        # interpret prediction probability
        if (y_prob >= self.threshold) or (y_prob <= (1 - self.threshold)):
            confidence = "HIGH"
        else:
            confidence = "LOW"

