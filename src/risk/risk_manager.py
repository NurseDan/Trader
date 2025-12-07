"""Risk management system for trading"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import numpy as np

from src.alpaca.client import AlpacaClient
from src.data.data_manager import DataManager
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    """Manages trading risk and position sizing"""

    def __init__(self, alpaca_client: AlpacaClient, data_manager: DataManager):
        self.alpaca_client = alpaca_client
        self.data_manager = data_manager

        # Risk parameters from config
        self.max_position_size = config.trading.max_position_size
        self.max_portfolio_risk = config.trading.max_portfolio_risk

        # Additional risk parameters
        self.max_positions = 10
        self.max_sector_exposure = 0.3
        self.max_single_position = 0.15  # 15% of portfolio
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.05  # 5% take profit

        logger.info("RiskManager initialized")

    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        current_price: float
    ) -> Dict:
        """Calculate appropriate position size based on risk parameters"""
        try:
            # Get account info
            account = self.alpaca_client.get_account()
            portfolio_value = account['portfolio_value']
            buying_power = account['buying_power']

            # Get volatility for risk adjustment
            volatility = self._get_volatility(symbol)

            # Base position size from signal strength (0-1)
            base_size = min(
                self.max_position_size,
                portfolio_value * self.max_single_position
            )

            # Adjust for signal strength
            position_value = base_size * signal_strength

            # Adjust for volatility (reduce size in high volatility)
            if volatility > 0:
                volatility_adj = max(0.5, 1 - (volatility * 2))
                position_value *= volatility_adj

            # Calculate number of shares
            shares = int(position_value / current_price)

            # Ensure we have buying power
            if shares * current_price > buying_power:
                shares = int(buying_power / current_price)

            # Calculate actual position value
            actual_value = shares * current_price

            # Calculate stop loss and take profit prices
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)

            # Calculate risk amount
            risk_amount = shares * (current_price - stop_loss)
            portfolio_risk = risk_amount / portfolio_value if portfolio_value > 0 else 0

            return {
                'shares': shares,
                'position_value': actual_value,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': risk_amount,
                'portfolio_risk_pct': portfolio_risk,
                'volatility': volatility,
                'approved': True if shares > 0 else False,
                'reason': 'Position size calculated'
            }

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'shares': 0,
                'approved': False,
                'reason': f'Error: {str(e)}'
            }

    def check_trade_approval(
        self,
        symbol: str,
        side: str,
        shares: int,
        price: float
    ) -> Dict:
        """Check if a trade should be approved based on risk rules"""
        checks = {
            'portfolio_limit': False,
            'position_limit': False,
            'buying_power': False,
            'max_positions': False,
            'risk_limit': False
        }

        reasons = []

        try:
            # Get account and positions
            account = self.alpaca_client.get_account()
            positions = self.alpaca_client.get_positions()

            portfolio_value = account['portfolio_value']
            buying_power = account['buying_power']

            # Check 1: Portfolio concentration
            position_value = shares * price
            position_pct = position_value / portfolio_value if portfolio_value > 0 else 0

            if position_pct <= self.max_single_position:
                checks['portfolio_limit'] = True
            else:
                reasons.append(f"Position would be {position_pct:.1%} of portfolio (max {self.max_single_position:.1%})")

            # Check 2: Maximum position size
            if position_value <= self.max_position_size:
                checks['position_limit'] = True
            else:
                reasons.append(f"Position value ${position_value:.2f} exceeds max ${self.max_position_size:.2f}")

            # Check 3: Buying power
            if side.lower() == 'buy':
                if position_value <= buying_power:
                    checks['buying_power'] = True
                else:
                    reasons.append(f"Insufficient buying power: ${buying_power:.2f} needed ${position_value:.2f}")
            else:
                checks['buying_power'] = True

            # Check 4: Maximum number of positions
            if len(positions) < self.max_positions:
                checks['max_positions'] = True
            else:
                # Check if we already have a position in this symbol
                existing = [p for p in positions if p['symbol'] == symbol]
                if existing:
                    checks['max_positions'] = True
                else:
                    reasons.append(f"Maximum positions reached ({self.max_positions})")

            # Check 5: Portfolio risk
            total_risk = self._calculate_portfolio_risk(positions)
            trade_risk = position_value * self.stop_loss_pct / portfolio_value if portfolio_value > 0 else 0
            total_risk_with_trade = total_risk + trade_risk

            if total_risk_with_trade <= self.max_portfolio_risk:
                checks['risk_limit'] = True
            else:
                reasons.append(f"Portfolio risk {total_risk_with_trade:.2%} would exceed max {self.max_portfolio_risk:.2%}")

            # All checks must pass
            approved = all(checks.values())

            return {
                'approved': approved,
                'checks': checks,
                'reasons': reasons if not approved else ['All risk checks passed'],
                'portfolio_risk': total_risk,
                'trade_risk': trade_risk
            }

        except Exception as e:
            logger.error(f"Error checking trade approval: {e}")
            return {
                'approved': False,
                'checks': checks,
                'reasons': [f'Error: {str(e)}']
            }

    def _get_volatility(self, symbol: str, days: int = 20) -> float:
        """Calculate recent volatility for a symbol"""
        try:
            df = self.data_manager.get_historical_data(symbol, days_back=days + 10)

            if df.empty or 'close' not in df.columns:
                return 0.02  # Default volatility

            returns = df['close'].pct_change().dropna()
            volatility = returns.tail(days).std()

            return volatility if not np.isnan(volatility) else 0.02

        except Exception as e:
            logger.warning(f"Error calculating volatility for {symbol}: {e}")
            return 0.02

    def _calculate_portfolio_risk(self, positions: list) -> float:
        """Calculate total portfolio risk from current positions"""
        if not positions:
            return 0.0

        total_risk = 0.0

        for position in positions:
            position_risk = abs(float(position['market_value'])) * self.stop_loss_pct
            total_risk += position_risk

        account = self.alpaca_client.get_account()
        portfolio_value = account['portfolio_value']

        return total_risk / portfolio_value if portfolio_value > 0 else 0.0

    def check_stop_loss(self, symbol: str, entry_price: float, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        loss_pct = (current_price - entry_price) / entry_price

        if loss_pct <= -self.stop_loss_pct:
            logger.warning(f"Stop loss triggered for {symbol}: {loss_pct:.2%}")
            return True

        return False

    def check_take_profit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        profit_pct = (current_price - entry_price) / entry_price

        if profit_pct >= self.take_profit_pct:
            logger.info(f"Take profit triggered for {symbol}: {profit_pct:.2%}")
            return True

        return False

    def get_trailing_stop_price(
        self,
        entry_price: float,
        current_price: float,
        trailing_pct: float = 0.02
    ) -> float:
        """Calculate trailing stop price"""
        # If in profit, trail the stop
        if current_price > entry_price:
            trailing_stop = current_price * (1 - trailing_pct)
            # Never lower the stop below entry
            return max(trailing_stop, entry_price * (1 - self.stop_loss_pct))
        else:
            # Use regular stop loss
            return entry_price * (1 - self.stop_loss_pct)

    def monitor_positions(self) -> Dict:
        """Monitor all positions for risk management triggers"""
        positions = self.alpaca_client.get_positions()

        actions = {
            'stop_losses': [],
            'take_profits': [],
            'warnings': []
        }

        for position in positions:
            symbol = position['symbol']
            entry_price = position['avg_entry_price']
            current_price = position['current_price']
            unrealized_plpc = position['unrealized_plpc']

            # Check stop loss
            if self.check_stop_loss(symbol, entry_price, current_price):
                actions['stop_losses'].append({
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'loss_pct': unrealized_plpc
                })

            # Check take profit
            if self.check_take_profit(symbol, entry_price, current_price):
                actions['take_profits'].append({
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'profit_pct': unrealized_plpc
                })

            # Warn on large unrealized losses
            if unrealized_plpc < -0.05:  # >5% loss
                actions['warnings'].append({
                    'symbol': symbol,
                    'loss_pct': unrealized_plpc,
                    'message': f"{symbol} down {unrealized_plpc:.1%}"
                })

        if actions['stop_losses']:
            logger.warning(f"Stop losses triggered: {len(actions['stop_losses'])}")

        if actions['take_profits']:
            logger.info(f"Take profits triggered: {len(actions['take_profits'])}")

        return actions

    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        try:
            account = self.alpaca_client.get_account()
            positions = self.alpaca_client.get_positions()

            portfolio_value = account['portfolio_value']
            cash = account['cash']

            # Calculate metrics
            total_exposure = sum(abs(float(p['market_value'])) for p in positions)
            exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0

            portfolio_risk = self._calculate_portfolio_risk(positions)

            # Largest position
            if positions:
                largest_pos = max(positions, key=lambda x: abs(float(x['market_value'])))
                largest_pos_pct = abs(float(largest_pos['market_value'])) / portfolio_value
            else:
                largest_pos = None
                largest_pos_pct = 0

            return {
                'portfolio_value': portfolio_value,
                'cash': cash,
                'cash_pct': cash / portfolio_value if portfolio_value > 0 else 0,
                'num_positions': len(positions),
                'total_exposure': total_exposure,
                'exposure_pct': exposure_pct,
                'portfolio_risk': portfolio_risk,
                'largest_position': largest_pos['symbol'] if largest_pos else None,
                'largest_position_pct': largest_pos_pct,
                'max_positions': self.max_positions,
                'max_position_size': self.max_position_size,
                'max_portfolio_risk': self.max_portfolio_risk
            }

        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {}
