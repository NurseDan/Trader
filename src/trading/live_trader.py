"""Live trading executor with safeguards"""

from typing import Dict, List, Optional
from datetime import datetime
import time

from src.alpaca.client import AlpacaClient
from src.data.data_manager import DataManager
from src.data.database import DatabaseManager
from src.strategy.ai_strategy import AITradingStrategy
from src.risk.risk_manager import RiskManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LiveTrader:
    """Executes live trades with risk management"""

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        data_manager: DataManager,
        db_manager: DatabaseManager,
        strategy: AITradingStrategy,
        risk_manager: RiskManager
    ):
        self.alpaca_client = alpaca_client
        self.data_manager = data_manager
        self.db_manager = db_manager
        self.strategy = strategy
        self.risk_manager = risk_manager

        self.dry_run = False  # Set to True to simulate without real orders
        self.min_confidence = 0.6  # Minimum confidence to execute trade

        logger.info("LiveTrader initialized")

    def execute_trade(
        self,
        symbol: str,
        side: str,
        shares: int,
        reason: str = ""
    ) -> Optional[Dict]:
        """Execute a trade with all safety checks"""
        logger.info(f"Attempting to execute: {side.upper()} {shares} {symbol}")

        # Get current price
        current_price = self.data_manager.get_latest_price(symbol)
        if not current_price:
            logger.error(f"Could not get price for {symbol}")
            return None

        # Risk management check
        approval = self.risk_manager.check_trade_approval(symbol, side, shares, current_price)

        if not approval['approved']:
            logger.warning(f"Trade rejected by risk manager: {approval['reasons']}")
            return None

        # Execute order
        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute: {side.upper()} {shares} {symbol} @ ${current_price:.2f}")
            order_data = {
                'id': 'dry_run_' + str(int(time.time())),
                'symbol': symbol,
                'qty': shares,
                'side': side,
                'type': 'market',
                'status': 'filled',
                'created_at': datetime.now()
            }
        else:
            # Place market order
            order_data = self.alpaca_client.place_market_order(
                symbol=symbol,
                qty=shares,
                side=side,
                time_in_force="day"
            )

            if not order_data:
                logger.error(f"Failed to place order for {symbol}")
                return None

        # Save to database
        trade_data = {
            'order_id': order_data['id'],
            'symbol': symbol,
            'side': side,
            'quantity': shares,
            'price': current_price,
            'order_type': 'market',
            'status': order_data.get('status', 'pending'),
            'strategy': 'ai_multi_signal',
            'timestamp': datetime.now()
        }

        self.db_manager.save_trade(trade_data)

        logger.bind(TRADE=True).info(
            f"Trade executed: {side.upper()} {shares} {symbol} @ ${current_price:.2f} - {reason}"
        )

        return order_data

    def execute_signal(self, analysis: Dict) -> Optional[Dict]:
        """Execute a trading signal from strategy analysis"""
        symbol = analysis['symbol']
        recommendation = analysis['recommendation']
        confidence = analysis['confidence']

        if recommendation == 'hold':
            logger.info(f"{symbol}: HOLD signal")
            return None

        if confidence < self.min_confidence:
            logger.info(f"{symbol}: Confidence {confidence:.1%} below minimum {self.min_confidence:.1%}")
            return None

        # Get current price
        current_price = self.data_manager.get_latest_price(symbol)
        if not current_price:
            return None

        # Check if we have an existing position
        existing_position = self.alpaca_client.get_position(symbol)

        if recommendation == 'buy' and not existing_position:
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol, confidence, current_price
            )

            if not position_size['approved'] or position_size['shares'] == 0:
                logger.warning(f"Position size not approved for {symbol}")
                return None

            # Execute buy
            return self.execute_trade(
                symbol=symbol,
                side='buy',
                shares=position_size['shares'],
                reason=f"AI signal: {confidence:.1%} confidence, Score: {analysis['overall_score']:.2f}"
            )

        elif recommendation == 'sell' and existing_position:
            # Execute sell
            return self.execute_trade(
                symbol=symbol,
                side='sell',
                shares=int(existing_position['qty']),
                reason=f"AI signal: {confidence:.1%} confidence, Score: {analysis['overall_score']:.2f}"
            )

        return None

    def run_trading_loop(
        self,
        symbols: List[str],
        interval_minutes: int = 60
    ):
        """Run continuous trading loop"""
        logger.info(f"Starting trading loop for {len(symbols)} symbols")
        logger.info(f"Interval: {interval_minutes} minutes, Dry run: {self.dry_run}")

        iteration = 0

        try:
            while True:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"Trading loop iteration {iteration} - {datetime.now()}")
                logger.info(f"{'='*60}")

                # Check market hours
                try:
                    clock = self.alpaca_client.trading_client.get_clock()
                    if not clock.is_open:
                        logger.info("Market is closed. Waiting...")
                        time.sleep(60)
                        continue
                except:
                    pass

                # Monitor existing positions for risk management
                self._monitor_positions()

                # Analyze and trade each symbol
                for symbol in symbols:
                    try:
                        logger.info(f"\nAnalyzing {symbol}...")

                        # Run strategy analysis
                        analysis = self.strategy.analyze_symbol(symbol)

                        # Execute if signal is strong enough
                        self.execute_signal(analysis)

                    except Exception as e:
                        logger.error(f"Error trading {symbol}: {e}")

                # Update performance metrics
                self._update_performance()

                # Wait for next iteration
                logger.info(f"\nWaiting {interval_minutes} minutes until next iteration...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("\nTrading loop stopped by user")
        except Exception as e:
            logger.error(f"Trading loop error: {e}")

    def _monitor_positions(self):
        """Monitor positions for stop loss and take profit"""
        actions = self.risk_manager.monitor_positions()

        # Execute stop losses
        for stop_loss in actions['stop_losses']:
            symbol = stop_loss['symbol']
            logger.warning(f"Executing stop loss for {symbol}")

            position = self.alpaca_client.get_position(symbol)
            if position:
                self.execute_trade(
                    symbol=symbol,
                    side='sell',
                    shares=int(position['qty']),
                    reason=f"Stop loss triggered at {stop_loss['loss_pct']:.2%}"
                )

        # Execute take profits
        for take_profit in actions['take_profits']:
            symbol = take_profit['symbol']
            logger.info(f"Executing take profit for {symbol}")

            position = self.alpaca_client.get_position(symbol)
            if position:
                self.execute_trade(
                    symbol=symbol,
                    side='sell',
                    shares=int(position['qty']),
                    reason=f"Take profit triggered at {take_profit['profit_pct']:.2%}"
                )

        # Log warnings
        for warning in actions['warnings']:
            logger.warning(warning['message'])

    def _update_performance(self):
        """Update daily performance metrics"""
        try:
            account = self.alpaca_client.get_account()
            positions = self.alpaca_client.get_positions()

            perf_data = {
                'date': datetime.now(),
                'portfolio_value': account['portfolio_value'],
                'cash': account['cash'],
                'equity': account['equity'],
                'num_positions': len(positions)
            }

            self.db_manager.save_performance(perf_data)

        except Exception as e:
            logger.error(f"Error updating performance: {e}")

    def close_all_positions(self, reason: str = "Manual close"):
        """Close all positions"""
        logger.warning(f"Closing all positions: {reason}")

        positions = self.alpaca_client.get_positions()

        for position in positions:
            symbol = position['symbol']
            qty = int(position['qty'])

            self.execute_trade(
                symbol=symbol,
                side='sell',
                shares=qty,
                reason=reason
            )

    def get_status(self) -> Dict:
        """Get current trading status"""
        account = self.alpaca_client.get_account()
        positions = self.alpaca_client.get_positions()
        risk_report = self.risk_manager.get_risk_report()

        return {
            'account': account,
            'positions': positions,
            'risk_report': risk_report,
            'dry_run': self.dry_run,
            'min_confidence': self.min_confidence
        }
