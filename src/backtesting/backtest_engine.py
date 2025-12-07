"""Backtesting engine for strategy validation"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.data.data_manager import DataManager
from src.strategy.ai_strategy import AITradingStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """Backtesting engine for trading strategies"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.initial_capital = 100000
        self.commission = 0.0  # Alpaca is commission-free

        logger.info("BacktestEngine initialized")

    def run_backtest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        strategy_params: Optional[Dict] = None
    ) -> Dict:
        """Run backtest for a symbol"""
        logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")

        # Get historical data
        df = self.data_manager.get_historical_data(symbol, start_date, end_date)

        if df.empty:
            logger.error("No data available for backtesting")
            return {}

        # Add technical indicators
        df = self.data_manager.add_technical_indicators(df)

        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'shares': 0,
            'equity': self.initial_capital,
            'positions': []
        }

        # Track performance
        trades = []
        equity_curve = []
        daily_returns = []

        # Simple strategy for backtesting
        # In production, you'd integrate with AITradingStrategy
        for i in range(50, len(df)):  # Start after warmup period
            current = df.iloc[i]
            prev = df.iloc[i-1]

            price = current['close']
            timestamp = current['timestamp']

            # Generate signal (simplified)
            signal = self._generate_backtest_signal(df.iloc[:i+1])

            # Execute trades
            if signal == 'buy' and portfolio['shares'] == 0:
                # Buy
                shares_to_buy = int(portfolio['cash'] / price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    portfolio['cash'] -= cost
                    portfolio['shares'] += shares_to_buy

                    trades.append({
                        'timestamp': timestamp,
                        'type': 'buy',
                        'price': price,
                        'shares': shares_to_buy,
                        'value': cost
                    })

                    logger.debug(f"BUY: {shares_to_buy} shares @ ${price:.2f}")

            elif signal == 'sell' and portfolio['shares'] > 0:
                # Sell
                proceeds = portfolio['shares'] * price
                portfolio['cash'] += proceeds

                trades.append({
                    'timestamp': timestamp,
                    'type': 'sell',
                    'price': price,
                    'shares': portfolio['shares'],
                    'value': proceeds
                })

                logger.debug(f"SELL: {portfolio['shares']} shares @ ${price:.2f}")
                portfolio['shares'] = 0

            # Update equity
            portfolio['equity'] = portfolio['cash'] + (portfolio['shares'] * price)
            equity_curve.append({
                'timestamp': timestamp,
                'equity': portfolio['equity'],
                'cash': portfolio['cash'],
                'position_value': portfolio['shares'] * price
            })

            # Calculate daily return
            if i > 50:
                prev_equity = equity_curve[-2]['equity']
                daily_return = (portfolio['equity'] - prev_equity) / prev_equity
                daily_returns.append(daily_return)

        # Calculate performance metrics
        metrics = self._calculate_metrics(equity_curve, daily_returns, trades)

        logger.info(
            f"Backtest complete: Total Return: {metrics['total_return']:.2%}, "
            f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
            f"Max Drawdown: {metrics['max_drawdown']:.2%}"
        )

        return {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_equity': portfolio['equity'],
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve
        }

    def _generate_backtest_signal(self, df: pd.DataFrame) -> str:
        """Generate trading signal for backtesting"""
        if len(df) < 50:
            return 'hold'

        latest = df.iloc[-1]

        # Simple moving average crossover strategy
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            if latest['sma_20'] > latest['sma_50'] and latest['close'] > latest['sma_20']:
                # Check RSI not overbought
                if 'rsi' in df.columns and latest['rsi'] < 70:
                    return 'buy'

            if latest['sma_20'] < latest['sma_50'] and latest['close'] < latest['sma_20']:
                return 'sell'

        return 'hold'

    def _calculate_metrics(
        self,
        equity_curve: List[Dict],
        daily_returns: List[float],
        trades: List[Dict]
    ) -> Dict:
        """Calculate performance metrics"""
        if not equity_curve:
            return {}

        initial_equity = equity_curve[0]['equity']
        final_equity = equity_curve[-1]['equity']

        # Total return
        total_return = (final_equity - initial_equity) / initial_equity

        # Calculate drawdowns
        equity_series = pd.Series([e['equity'] for e in equity_curve])
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sharpe ratio (annualized)
        if daily_returns:
            returns_array = np.array(daily_returns)
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
        else:
            sharpe_ratio = 0

        # Win rate
        if trades:
            winning_trades = 0
            losing_trades = 0
            total_profit = 0
            total_loss = 0

            buy_price = None
            for trade in trades:
                if trade['type'] == 'buy':
                    buy_price = trade['price']
                elif trade['type'] == 'sell' and buy_price:
                    profit = (trade['price'] - buy_price) * trade['shares']
                    if profit > 0:
                        winning_trades += 1
                        total_profit += profit
                    else:
                        losing_trades += 1
                        total_loss += abs(profit)

            total_trades = winning_trades + losing_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_win = total_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            total_trades = 0

        # Annualized return
        days = len(equity_curve)
        years = days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_equity': final_equity
        }

    def compare_strategies(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        strategies: List[Dict]
    ) -> Dict:
        """Compare multiple strategy configurations"""
        results = []

        for strategy in strategies:
            result = self.run_backtest(symbol, start_date, end_date, strategy)
            result['strategy_name'] = strategy.get('name', 'unnamed')
            results.append(result)

        # Rank by Sharpe ratio
        results.sort(key=lambda x: x['metrics'].get('sharpe_ratio', 0), reverse=True)

        return {
            'symbol': symbol,
            'comparisons': results,
            'best_strategy': results[0] if results else None
        }

    def monte_carlo_simulation(
        self,
        equity_curve: List[Dict],
        num_simulations: int = 1000
    ) -> Dict:
        """Run Monte Carlo simulation on backtest results"""
        if not equity_curve:
            return {}

        # Calculate returns
        equity_series = pd.Series([e['equity'] for e in equity_curve])
        returns = equity_series.pct_change().dropna()

        # Run simulations
        final_equities = []

        for _ in range(num_simulations):
            # Randomly sample returns with replacement
            simulated_returns = np.random.choice(returns, size=len(returns), replace=True)

            # Calculate final equity
            initial = equity_series.iloc[0]
            final = initial * (1 + simulated_returns).prod()
            final_equities.append(final)

        # Calculate confidence intervals
        final_equities = np.array(final_equities)
        percentiles = np.percentile(final_equities, [5, 25, 50, 75, 95])

        return {
            'mean_final_equity': np.mean(final_equities),
            'median_final_equity': np.median(final_equities),
            'std_final_equity': np.std(final_equities),
            'percentile_5': percentiles[0],
            'percentile_25': percentiles[1],
            'percentile_50': percentiles[2],
            'percentile_75': percentiles[3],
            'percentile_95': percentiles[4],
            'probability_profit': np.mean(final_equities > equity_series.iloc[0])
        }
