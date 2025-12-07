#!/usr/bin/env python3
"""
AI-Powered Trading Bot - Main Entry Point
Combines ML predictions, sentiment analysis, and multi-source event monitoring
"""

import argparse
from datetime import datetime, timedelta

from src.alpaca.client import AlpacaClient
from src.data.data_manager import DataManager
from src.data.database import DatabaseManager
from src.strategy.ai_strategy import AITradingStrategy
from src.risk.risk_manager import RiskManager
from src.trading.live_trader import LiveTrader
from src.backtesting.backtest_engine import BacktestEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_models(symbols: list):
    """Train ML models for given symbols"""
    logger.info("Training ML models")

    alpaca_client = AlpacaClient()
    data_manager = DataManager(alpaca_client)
    db_manager = DatabaseManager()

    strategy = AITradingStrategy(alpaca_client, data_manager, db_manager)

    for symbol in symbols:
        logger.info(f"\nTraining models for {symbol}")
        results = strategy.train_models(symbol, days_back=365)
        logger.info(f"Training results: {results}")


def backtest(symbols: list, days_back: int = 365):
    """Run backtest for given symbols"""
    logger.info("Running backtest")

    alpaca_client = AlpacaClient()
    data_manager = DataManager(alpaca_client)

    backtest_engine = BacktestEngine(data_manager)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    for symbol in symbols:
        logger.info(f"\nBacktesting {symbol}")
        results = backtest_engine.run_backtest(symbol, start_date, end_date)

        if results:
            metrics = results['metrics']
            logger.info(f"\n{'='*60}")
            logger.info(f"Backtest Results for {symbol}")
            logger.info(f"{'='*60}")
            logger.info(f"Initial Capital: ${results['initial_capital']:,.2f}")
            logger.info(f"Final Equity: ${results['final_equity']:,.2f}")
            logger.info(f"Total Return: {metrics['total_return']:.2%}")
            logger.info(f"Annualized Return: {metrics['annualized_return']:.2%}")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
            logger.info(f"Total Trades: {metrics['total_trades']}")
            logger.info(f"{'='*60}\n")


def analyze(symbols: list):
    """Analyze symbols and show recommendations"""
    logger.info("Analyzing symbols")

    alpaca_client = AlpacaClient()
    data_manager = DataManager(alpaca_client)
    db_manager = DatabaseManager()

    strategy = AITradingStrategy(alpaca_client, data_manager, db_manager)

    results = strategy.screen_stocks(symbols, min_confidence=0.5)

    # Print buy candidates
    logger.info(f"\n{'='*60}")
    logger.info("BUY CANDIDATES")
    logger.info(f"{'='*60}")

    for analysis in results['buy_candidates']:
        logger.info(f"\n{analysis['symbol']}:")
        logger.info(f"  Confidence: {analysis['confidence']:.1%}")
        logger.info(f"  Overall Score: {analysis['overall_score']:.2f}")
        logger.info(f"  ML Score: {analysis['scores'].get('ml', 0):.2f}")
        logger.info(f"  Technical Score: {analysis['scores'].get('technical', 0):.2f}")
        logger.info(f"  Sentiment Score: {analysis['scores'].get('sentiment', 0):.2f}")
        logger.info(f"  Volume Score: {analysis['scores'].get('volume', 0):.2f}")

    # Print sell candidates
    logger.info(f"\n{'='*60}")
    logger.info("SELL CANDIDATES")
    logger.info(f"{'='*60}")

    for analysis in results['sell_candidates']:
        logger.info(f"\n{analysis['symbol']}:")
        logger.info(f"  Confidence: {analysis['confidence']:.1%}")
        logger.info(f"  Overall Score: {analysis['overall_score']:.2f}")


def live_trade(symbols: list, interval: int = 60, dry_run: bool = True):
    """Run live trading"""
    logger.info("Starting live trading")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")

    alpaca_client = AlpacaClient()
    data_manager = DataManager(alpaca_client)
    db_manager = DatabaseManager()

    strategy = AITradingStrategy(alpaca_client, data_manager, db_manager)
    risk_manager = RiskManager(alpaca_client, data_manager)

    trader = LiveTrader(
        alpaca_client,
        data_manager,
        db_manager,
        strategy,
        risk_manager
    )

    trader.dry_run = dry_run

    # Show initial status
    status = trader.get_status()
    logger.info(f"\nAccount Status:")
    logger.info(f"  Portfolio Value: ${status['account']['portfolio_value']:,.2f}")
    logger.info(f"  Cash: ${status['account']['cash']:,.2f}")
    logger.info(f"  Buying Power: ${status['account']['buying_power']:,.2f}")
    logger.info(f"  Current Positions: {len(status['positions'])}\n")

    # Run trading loop
    trader.run_trading_loop(symbols, interval_minutes=interval)


def status():
    """Show current account and positions status"""
    alpaca_client = AlpacaClient()
    data_manager = DataManager(alpaca_client)
    risk_manager = RiskManager(alpaca_client, data_manager)

    # Account info
    account = alpaca_client.get_account()
    logger.info(f"\n{'='*60}")
    logger.info("ACCOUNT STATUS")
    logger.info(f"{'='*60}")
    logger.info(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
    logger.info(f"Cash: ${account['cash']:,.2f}")
    logger.info(f"Equity: ${account['equity']:,.2f}")
    logger.info(f"Buying Power: ${account['buying_power']:,.2f}")

    # Positions
    positions = alpaca_client.get_positions()
    logger.info(f"\n{'='*60}")
    logger.info(f"POSITIONS ({len(positions)})")
    logger.info(f"{'='*60}")

    for pos in positions:
        logger.info(f"\n{pos['symbol']}:")
        logger.info(f"  Quantity: {pos['qty']}")
        logger.info(f"  Entry Price: ${pos['avg_entry_price']:.2f}")
        logger.info(f"  Current Price: ${pos['current_price']:.2f}")
        logger.info(f"  Market Value: ${pos['market_value']:,.2f}")
        logger.info(f"  P/L: ${pos['unrealized_pl']:,.2f} ({pos['unrealized_plpc']:.2%})")

    # Risk report
    risk_report = risk_manager.get_risk_report()
    logger.info(f"\n{'='*60}")
    logger.info("RISK REPORT")
    logger.info(f"{'='*60}")
    logger.info(f"Portfolio Risk: {risk_report['portfolio_risk']:.2%}")
    logger.info(f"Total Exposure: ${risk_report['total_exposure']:,.2f} ({risk_report['exposure_pct']:.1%})")
    logger.info(f"Largest Position: {risk_report['largest_position']} ({risk_report['largest_position_pct']:.1%})")


def main():
    parser = argparse.ArgumentParser(description='AI-Powered Trading Bot')

    parser.add_argument(
        'command',
        choices=['train', 'backtest', 'analyze', 'trade', 'status'],
        help='Command to execute'
    )

    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        help='Stock symbols to trade (default: AAPL MSFT GOOGL TSLA NVDA)'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Trading interval in minutes (default: 60)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days for backtesting (default: 365)'
    )

    parser.add_argument(
        '--live',
        action='store_true',
        help='Run in live mode (default: dry run)'
    )

    args = parser.parse_args()

    logger.info(f"\n{'='*60}")
    logger.info("AI-POWERED TRADING BOT")
    logger.info(f"{'='*60}\n")

    if args.command == 'train':
        train_models(args.symbols)
    elif args.command == 'backtest':
        backtest(args.symbols, args.days)
    elif args.command == 'analyze':
        analyze(args.symbols)
    elif args.command == 'trade':
        live_trade(args.symbols, args.interval, dry_run=not args.live)
    elif args.command == 'status':
        status()


if __name__ == '__main__':
    main()
