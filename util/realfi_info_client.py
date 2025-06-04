import os
import requests
import time
from typing import List, Dict, Optional
import pandas as pd
from dotenv import load_dotenv


class RealFiInfoClient:
    """Client for interacting with the RealFi.info API"""

    def __init__(self):
        load_dotenv()
        self.base_url = os.getenv('REALFI_BASE_URL', 'https://api.realfi.info')
        self.auth_header = os.getenv('REALFI_API_TOKEN', '')
        self.session = requests.Session()

        if self.auth_header:
            self.session.headers.update({'Authorization': self.auth_header})

    def get_latest_price(self, symbols: List[str]) -> Dict:
        """
        Get the latest price for one or more cryptocurrencies using their unique unit IDs

        Args:
            symbols: List of unique unit IDs (policy ID + hex-encoded name)

        Returns:
            Dictionary with price data
        """
        try:
            response = self.session.get(
                f"{self.base_url}/prices/latest",
                params={"symbols": symbols}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching latest prices: {e}")
            return {}

    def get_historical_prices(self, symbol: str, resolution: str = "1D",
                              from_ts: Optional[int] = None, to_ts: Optional[int] = None) -> Dict:
        """
        Get historical prices for a cryptocurrency

        Args:
            symbol: Unique unit ID of the cryptocurrency
            resolution: Time resolution (e.g., 1W, 1D, 1h, 15m)
            from_ts: Start timestamp (Unix epoch)
            to_ts: End timestamp (Unix epoch)

        Returns:
            Dictionary with historical price data
        """
        try:
            params = {
                "symbol": symbol,
                "resolution": resolution
            }

            if from_ts:
                params["from"] = from_ts
            if to_ts:
                params["to"] = to_ts

            response = self.session.get(
                f"{self.base_url}/prices/historical/candles",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching historical prices for {symbol}: {e}")
            return {}

    def get_top_tokens_by_volume(self, limit: int = 10) -> Dict:
        """
        Get the top tokens by trading volume

        Args:
            limit: Number of tokens to return

        Returns:
            Dictionary with top tokens data
        """
        try:
            response = self.session.get(
                f"{self.base_url}/tokens/top-by-volume",
                params={"limit": limit}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching top tokens: {e}")
            return {}

    def get_top_tokens_by_volume_df(self, limit: int = 10) -> pd.DataFrame:
        """
        Get top Cardano tokens with their symbols, unit IDs, and volumes as a DataFrame

        Args:
            limit: Number of tokens to return

        Returns:
            DataFrame with columns: symbol, unit_id, volume
        """
        # Fetch data from API
        data = self.get_top_tokens_by_volume(limit)['assets']

        # # Hardcoded fallback (commented out)
        data = [
            {"nativeName": "SNEK", "unit": "279c909f348e533da5808898f87f9a14bb2c3dfbbacccd631d927a3f534e454b", "totalVolume": 312398.744331},
            {"nativeName": "WorldMobileTokenX", "unit": "e5a42a1a1d3d1da71b0449663c32798725888d2eb0843c4dabeca05a576f726c644d6f62696c65546f6b656e58", "totalVolume": 161003.063478},
            {"nativeName": "STRIKE", "unit": "f13ac4d66b3ee19a6aa0f2a22298737bd907cc95121662fc971b5275535452494b45", "totalVolume": 138247.992063},
        ]

        if not data:
            return pd.DataFrame(columns=['symbol', 'unit_id', 'volume'])

        # Extract assets list and rename keys to match expected columns
        df_data = [
            {
                'symbol': asset.get('nativeName', ''),
                'unit_id': asset.get('unit', ''),
                'volume': asset.get('totalVolume', 0.0)
            }
            for asset in data
        ]

        # Convert to DataFrame
        df = pd.DataFrame(df_data)

        if df.empty:
            return pd.DataFrame(columns=['symbol', 'unit_id', 'volume'])

        # Ensure volume is numeric
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        # Ensure expected columns are returned
        return df[['symbol', 'unit_id', 'volume']]

    def get_historical_data_df(self, symbol: str, days: int = 365, resolution: str = "1D") -> pd.DataFrame:
        """
        Get historical data as a pandas DataFrame

        Args:
            symbol: Token unit ID
            days: Number of days of historical data
            resolution: Time resolution

        Returns:
            DataFrame with OHLCV data
        """
        to_ts = int(time.time())
        from_ts = to_ts - (days * 24 * 60 * 60)

        data = self.get_historical_prices(symbol, resolution, from_ts, to_ts)

        if not data:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data)

        if df.empty:
            return df

        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Ensure we have the expected columns
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Convert to numeric
        for col in expected_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.sort_index()