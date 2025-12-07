"""Historical data management and caching"""

from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from pathlib import Path
import pickle

from src.alpaca.client import AlpacaClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataManager:
    """Manages historical data fetching and caching"""

    def __init__(self, alpaca_client: AlpacaClient, cache_dir: str = "data/cache"):
        self.alpaca_client = alpaca_client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DataManager initialized")

    def _get_cache_path(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> Path:
        """Generate cache file path"""
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        return self.cache_dir / f"{symbol}_{start_str}_{end_str}_{timeframe}.pkl"

    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        try:
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
        return None

    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path):
        """Save data to cache"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            logger.debug(f"Data cached to {cache_path}")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")

    def get_historical_data(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1Day",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get historical data for a symbol

        Args:
            symbol: Stock symbol
            start: Start date (default: 1 year ago)
            end: End date (default: now)
            timeframe: Data timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            use_cache: Whether to use cached data
        """
        if start is None:
            start = datetime.now() - timedelta(days=365)
        if end is None:
            end = datetime.now()

        # Check cache
        cache_path = self._get_cache_path(symbol, start, end, timeframe)
        if use_cache:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                logger.debug(f"Loaded {symbol} data from cache")
                return cached_data

        # Fetch from Alpaca
        try:
            df = self.alpaca_client.get_historical_bars(symbol, start, end, timeframe)

            if not df.empty:
                logger.info(f"Fetched {len(df)} bars for {symbol} from Alpaca")
                if use_cache:
                    self._save_to_cache(df, cache_path)
                return df
        except Exception as e:
            logger.warning(f"Alpaca fetch failed for {symbol}: {e}")

        # Fallback to yfinance
        try:
            logger.info(f"Falling back to yfinance for {symbol}")
            ticker = yf.Ticker(symbol)

            # Map timeframe to yfinance interval
            interval_map = {
                "1Min": "1m",
                "5Min": "5m",
                "15Min": "15m",
                "1Hour": "1h",
                "1Day": "1d",
            }
            interval = interval_map.get(timeframe, "1d")

            df = ticker.history(start=start, end=end, interval=interval)

            if not df.empty:
                # Standardize column names
                df.reset_index(inplace=True)
                df.columns = df.columns.str.lower()
                if 'date' in df.columns:
                    df.rename(columns={'date': 'timestamp'}, inplace=True)
                elif 'datetime' in df.columns:
                    df.rename(columns={'datetime': 'timestamp'}, inplace=True)

                logger.info(f"Fetched {len(df)} bars for {symbol} from yfinance")
                if use_cache:
                    self._save_to_cache(df, cache_path)
                return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from yfinance: {e}")

        return pd.DataFrame()

    def get_multiple_symbols(
        self,
        symbols: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1Day"
    ) -> dict:
        """Get historical data for multiple symbols"""
        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, start, end, timeframe)
            if not df.empty:
                data[symbol] = df
        return data

    def calculate_returns(self, df: pd.DataFrame, period: int = 1) -> pd.Series:
        """Calculate returns from price data"""
        if 'close' in df.columns:
            return df['close'].pct_change(period)
        return pd.Series()

    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate rolling volatility"""
        returns = self.calculate_returns(df)
        return returns.rolling(window=window).std()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to dataframe"""
        try:
            import pandas_ta as ta

            df_copy = df.copy()

            # Moving averages
            df_copy['sma_20'] = ta.sma(df_copy['close'], length=20)
            df_copy['sma_50'] = ta.sma(df_copy['close'], length=50)
            df_copy['ema_12'] = ta.ema(df_copy['close'], length=12)
            df_copy['ema_26'] = ta.ema(df_copy['close'], length=26)

            # RSI
            df_copy['rsi'] = ta.rsi(df_copy['close'], length=14)

            # MACD
            macd = ta.macd(df_copy['close'])
            if macd is not None:
                df_copy = pd.concat([df_copy, macd], axis=1)

            # Bollinger Bands
            bbands = ta.bbands(df_copy['close'], length=20)
            if bbands is not None:
                df_copy = pd.concat([df_copy, bbands], axis=1)

            # Volume indicators
            if 'volume' in df_copy.columns:
                df_copy['volume_sma'] = ta.sma(df_copy['volume'], length=20)

            # ATR (Average True Range)
            atr = ta.atr(df_copy['high'], df_copy['low'], df_copy['close'], length=14)
            if atr is not None:
                df_copy['atr'] = atr

            logger.debug("Technical indicators added")
            return df_copy

        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol"""
        try:
            quote = self.alpaca_client.get_latest_quote(symbol)
            if quote:
                return (quote['bid'] + quote['ask']) / 2
        except Exception as e:
            logger.warning(f"Error getting latest price from Alpaca: {e}")

        # Fallback to yfinance
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"Error getting latest price: {e}")

        return None

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data"""
        if symbol:
            for file in self.cache_dir.glob(f"{symbol}_*.pkl"):
                file.unlink()
            logger.info(f"Cache cleared for {symbol}")
        else:
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            logger.info("All cache cleared")
