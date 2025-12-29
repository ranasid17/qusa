# qusa/scripts/most_recent_day.py

"""
Fetch most recent completed trading day's OHLCV data from Polygon.
Intended for model inference / smoke testing.
"""

import os
import pandas as pd
import requests

from datetime import datetime, timedelta, timezone


class MostRecentDayFetcher:
    """
    Docstring for MostRecentDayFetcher
    """

    def __init__(self, api_key=None):
        """
        Class constructor.

        Parameters:
            1) api_key (str, opt): Massive API key
        """

        self.api_key = api_key or os.getenv("POLYGON_API_KEY")

        # handle case where API key not loaded
        if not self.api_key:
            raise ValueError("Polygon API key not provided")

        self.base_url = "https://api.polygon.io"

    @staticmethod
    def _get_most_recent_trading_day():
        """
        Return YYYY-MM-DD for most recent USA trading day.

        Returns:
            1) day (type): YYYY-MM-DD string of most recent trading day.
        """

        # store today's time in UTC
        today = datetime.now(timezone.utc).date()

        # assume most recent trading day is yesterday
        day = today - timedelta(days=1)

        # roll back weekends
        while day.weekday() >= 5:
            # subtract additional day if Saturday (5), Sunday (6)
            day -= timedelta(days=1)

        return day.isoformat()

    def fetch_daily_bar(self, ticker):
        """
        Fetch most recent completed daily bar for given ticker.

        Parameters:
            1) ticker (str): Stock ticker symbol.

        Returns:
            1) pd.DataFrame: DataFrame containing OHLCV data for most recent trading day.
        """

        # store most recent trading day
        date = self._get_most_recent_trading_day()

        # define request URL and parameters
        url = f"{self.base_url}/v1/open-close/{ticker}/{date}"
        params = {"adjusted": "true", "apiKey": self.api_key}

        # make request to API
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        # parse response JSON
        data = response.json()

        # handle case where API did not return valid data
        if data.get("status") != "OK":
            raise ValueError(f"API returned non-OK status: {data.get('status')}")

        else:
            row = {
                "ticker": ticker,
                "date": date,
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "close": data["close"],
                "volume": data["volume"],
            }

        return pd.DataFrame([row])
