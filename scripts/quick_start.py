#!/usr/bin/env python3
"""Quick start script for testing the trading bot"""

from datetime import datetime, timedelta
from src.alpaca.client import AlpacaClient
from src.data.data_manager import DataManager
from src.data.database import DatabaseManager
from src.strategy.ai_strategy import AITradingStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def quick_test():
    """Quick test of the trading bot functionality"""

    logger.info("Starting quick test...")

    # Initialize components
    alpaca_client = AlpacaClient()
    data_manager = DataManager(alpaca_client)
    db_manager = DatabaseManager()
    strategy = AITradingStrategy(alpaca_client, data_manager, db_manager)

    # Test symbol
    symbol = 'AAPL'

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing with {symbol}")
    logger.info(f"{'='*60}\n")

    # 1. Test data fetching
    logger.info("1. Testing data fetching...")
    df = data_manager.get_historical_data(symbol, days_back=30)
    logger.info(f"   Fetched {len(df)} bars of data")

    # 2. Test technical indicators
    logger.info("\n2. Testing technical indicators...")
    df = data_manager.add_technical_indicators(df)
    logger.info(f"   Added {len(df.columns)} columns")

    # 3. Test ML model training
    logger.info("\n3. Testing ML model training...")
    results = strategy.train_models(symbol, days_back=180)
    logger.info(f"   Model trained: {results}")

    # 4. Test strategy analysis
    logger.info("\n4. Testing strategy analysis...")
    analysis = strategy.analyze_symbol(symbol)
    logger.info(f"   Recommendation: {analysis['recommendation']}")
    logger.info(f"   Confidence: {analysis['confidence']:.1%}")
    logger.info(f"   Overall Score: {analysis['overall_score']:.2f}")
    logger.info(f"   ML Score: {analysis['scores'].get('ml', 0):.2f}")
    logger.info(f"   Technical Score: {analysis['scores'].get('technical', 0):.2f}")
    logger.info(f"   Sentiment Score: {analysis['scores'].get('sentiment', 0):.2f}")

    # 5. Test account status
    logger.info("\n5. Testing account status...")
    account = alpaca_client.get_account()
    logger.info(f"   Portfolio Value: ${account['portfolio_value']:,.2f}")
    logger.info(f"   Cash: ${account['cash']:,.2f}")

    logger.info(f"\n{'='*60}")
    logger.info("Quick test completed successfully!")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    quick_test()
