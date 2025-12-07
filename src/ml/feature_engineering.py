"""Feature engineering for ML models"""

import pandas as pd
import numpy as np
from typing import List

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Creates features for ML models"""

    def __init__(self):
        logger.info("FeatureEngineer initialized")

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        df = df.copy()

        # Returns
        for period in [1, 5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(period)

        # Price momentum
        df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1

        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_5d'] = df['hl_range'].rolling(5).mean()

        # Close position in range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        return df

    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        df = df.copy()

        if 'volume' not in df.columns:
            return df

        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ratio_5d'] = df['volume'] / df['volume'].rolling(5).mean()
        df['volume_ratio_20d'] = df['volume'] / df['volume'].rolling(20).mean()

        # Price-volume correlation
        df['pv_corr_10d'] = df['close'].rolling(10).corr(df['volume'])

        return df

    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        df = df.copy()

        # Already calculated in data_manager, but ensure they exist
        if 'sma_20' not in df.columns:
            df['sma_20'] = df['close'].rolling(20).mean()
        if 'sma_50' not in df.columns:
            df['sma_50'] = df['close'].rolling(50).mean()

        # Price relative to moving averages
        df['price_to_sma20'] = df['close'] / df['sma_20'] - 1
        df['price_to_sma50'] = df['close'] / df['sma_50'] - 1

        # Moving average crossovers
        df['sma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)

        # Volatility
        df['volatility_10d'] = df['close'].pct_change().rolling(10).std()
        df['volatility_20d'] = df['close'].pct_change().rolling(20).std()

        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter

            # Cyclical encoding
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def create_target_variables(self, df: pd.DataFrame, horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """Create target variables for prediction"""
        df = df.copy()

        for horizon in horizons:
            # Future returns
            df[f'target_return_{horizon}d'] = df['close'].pct_change(horizon).shift(-horizon)

            # Future direction (up/down)
            df[f'target_direction_{horizon}d'] = (df[f'target_return_{horizon}d'] > 0).astype(int)

            # Future volatility
            df[f'target_volatility_{horizon}d'] = (
                df['close'].pct_change().rolling(horizon).std().shift(-horizon)
            )

        return df

    def create_all_features(self, df: pd.DataFrame, include_targets: bool = True) -> pd.DataFrame:
        """Create all features"""
        logger.info("Creating all features")

        df = self.create_price_features(df)
        df = self.create_volume_features(df)
        df = self.create_technical_features(df)
        df = self.create_time_features(df)

        if include_targets:
            df = self.create_target_variables(df)

        # Drop rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Feature engineering complete: {len(df)}/{initial_rows} rows kept")

        return df

    def get_feature_columns(self, df: pd.DataFrame, exclude_targets: bool = True) -> List[str]:
        """Get list of feature columns"""
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                        'trade_count', 'vwap']

        if exclude_targets:
            exclude_cols.extend([col for col in df.columns if col.startswith('target_')])

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
