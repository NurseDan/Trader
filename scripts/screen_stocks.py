#!/usr/bin/env python3
"""Stock screening script"""

from src.alpaca.client import AlpacaClient
from src.data.data_manager import DataManager
from src.data.database import DatabaseManager
from src.strategy.ai_strategy import AITradingStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Popular stock lists
STOCK_LISTS = {
    'faang': ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL'],
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMD', 'INTC'],
    'sp500_top': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ'],
    'ev': ['TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV'],
    'ai': ['NVDA', 'AMD', 'GOOGL', 'MSFT', 'META', 'PLTR', 'C3.AI'],
}


def screen_stocks(stock_list: str = 'tech', min_confidence: float = 0.5):
    """Screen stocks from a predefined list"""

    if stock_list not in STOCK_LISTS:
        logger.error(f"Unknown stock list: {stock_list}")
        logger.info(f"Available lists: {', '.join(STOCK_LISTS.keys())}")
        return

    symbols = STOCK_LISTS[stock_list]

    logger.info(f"\n{'='*60}")
    logger.info(f"Screening {stock_list.upper()} stocks: {', '.join(symbols)}")
    logger.info(f"{'='*60}\n")

    # Initialize
    alpaca_client = AlpacaClient()
    data_manager = DataManager(alpaca_client)
    db_manager = DatabaseManager()
    strategy = AITradingStrategy(alpaca_client, data_manager, db_manager)

    # Screen stocks
    results = strategy.screen_stocks(symbols, min_confidence=min_confidence)

    # Display results
    logger.info(f"\n{'='*60}")
    logger.info("BUY CANDIDATES")
    logger.info(f"{'='*60}\n")

    if results['buy_candidates']:
        for i, analysis in enumerate(results['buy_candidates'], 1):
            logger.info(f"{i}. {analysis['symbol']}")
            logger.info(f"   Confidence: {analysis['confidence']:.1%}")
            logger.info(f"   Overall Score: {analysis['overall_score']:.2f}")
            logger.info(f"   ML: {analysis['scores'].get('ml', 0):.2f} | "
                       f"Technical: {analysis['scores'].get('technical', 0):.2f} | "
                       f"Sentiment: {analysis['scores'].get('sentiment', 0):.2f} | "
                       f"Volume: {analysis['scores'].get('volume', 0):.2f}")
            logger.info("")
    else:
        logger.info("No buy candidates found.\n")

    logger.info(f"{'='*60}")
    logger.info("SELL CANDIDATES")
    logger.info(f"{'='*60}\n")

    if results['sell_candidates']:
        for i, analysis in enumerate(results['sell_candidates'], 1):
            logger.info(f"{i}. {analysis['symbol']}")
            logger.info(f"   Confidence: {analysis['confidence']:.1%}")
            logger.info(f"   Overall Score: {analysis['overall_score']:.2f}")
            logger.info("")
    else:
        logger.info("No sell candidates found.\n")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        stock_list = sys.argv[1]
    else:
        stock_list = 'tech'

    min_conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    screen_stocks(stock_list, min_conf)
