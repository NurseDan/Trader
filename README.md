# AI-Powered Trading Bot

A comprehensive algorithmic trading bot using **Alpaca's official API** with machine learning predictions, sentiment analysis, and multi-source event monitoring.

## Features

- **Alpaca API Integration**: Fully legal and authorized trading via Alpaca's official API
- **Machine Learning Models**: XGBoost, LightGBM, and Random Forest for price prediction
- **Multi-Signal Analysis**: Combines technical indicators, ML predictions, sentiment, and volume
- **Sentiment Analysis**: Monitors news, Reddit, and Twitter for market sentiment
- **Risk Management**: Position sizing, stop-loss, take-profit, and portfolio risk limits
- **Backtesting**: Historical strategy validation with performance metrics
- **Live Trading**: Automated execution with safeguards and monitoring
- **Performance Tracking**: Database storage of trades, signals, and metrics

## Architecture

```
src/
├── alpaca/          # Alpaca API client
├── data/            # Historical data and database
├── ml/              # Machine learning models
├── sentiment/       # Sentiment analysis
├── news/            # News aggregation
├── social/          # Reddit and Twitter monitoring
├── strategy/        # AI trading strategy
├── risk/            # Risk management
├── backtesting/     # Backtesting engine
├── trading/         # Live trading executor
└── utils/           # Configuration and logging
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Trader.git
cd Trader
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Alpaca API - Get from https://app.alpaca.markets/
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# News API - Get from https://newsapi.org/
NEWS_API_KEY=your_news_api_key_here

# Reddit API - Create app at https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# Twitter API (Optional)
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret

# Trading Parameters
MAX_POSITION_SIZE=1000
MAX_PORTFOLIO_RISK=0.02
TRADING_MODE=paper
```

## Getting API Keys

### Alpaca (Required)

1. Go to [alpaca.markets](https://alpaca.markets)
2. Sign up for a free account
3. Navigate to "Paper Trading" section
4. Generate API keys
5. Copy API Key and Secret Key to `.env`

### NewsAPI (Optional but Recommended)

1. Go to [newsapi.org](https://newsapi.org)
2. Sign up for free tier
3. Copy API key to `.env`

### Reddit (Optional)

1. Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Create a new app (script type)
3. Copy client ID and secret to `.env`

### Twitter (Optional)

1. Go to [developer.twitter.com](https://developer.twitter.com)
2. Apply for developer access
3. Create an app and get API keys

## Usage

### 1. Train ML Models

Train machine learning models on historical data:

```bash
python main.py train --symbols AAPL MSFT GOOGL TSLA NVDA
```

### 2. Backtest Strategy

Test the strategy on historical data:

```bash
python main.py backtest --symbols AAPL MSFT --days 365
```

Example output:
```
Backtest Results for AAPL
============================================================
Initial Capital: $100,000.00
Final Equity: $125,450.00
Total Return: 25.45%
Annualized Return: 25.45%
Sharpe Ratio: 1.85
Max Drawdown: -8.32%
Win Rate: 62.50%
Total Trades: 24
============================================================
```

### 3. Analyze Stocks

Get AI-powered analysis and recommendations:

```bash
python main.py analyze --symbols AAPL MSFT GOOGL TSLA NVDA
```

### 4. Check Account Status

View your account and positions:

```bash
python main.py status
```

### 5. Run Live Trading (DRY RUN)

Test trading logic without real orders:

```bash
python main.py trade --symbols AAPL MSFT GOOGL --interval 60
```

### 6. Run Live Trading (REAL)

**⚠️ WARNING: This will execute real trades with real money!**

Start with paper trading first. When ready:

```bash
python main.py trade --symbols AAPL MSFT --interval 60 --live
```

## Configuration

### Trading Parameters

Edit `.env` to adjust:

- `MAX_POSITION_SIZE`: Maximum dollar amount per position (default: $1000)
- `MAX_PORTFOLIO_RISK`: Maximum portfolio risk percentage (default: 0.02 = 2%)
- `TRADING_MODE`: `paper` for testing, `live` for real trading

### Strategy Weights

Edit `src/strategy/ai_strategy.py`:

```python
self.weights = {
    'ml_prediction': 0.35,   # ML model predictions
    'technical': 0.25,       # Technical indicators
    'sentiment': 0.25,       # News/social sentiment
    'volume': 0.15          # Volume analysis
}
```

### Risk Management

Edit `src/risk/risk_manager.py`:

```python
self.max_positions = 10              # Maximum number of positions
self.max_single_position = 0.15      # Max 15% per position
self.stop_loss_pct = 0.02           # 2% stop loss
self.take_profit_pct = 0.05         # 5% take profit
```

## Strategy Overview

The AI trading strategy combines multiple signals:

### 1. ML Predictions (35% weight)
- XGBoost price prediction
- Direction classification
- Feature engineering with 20+ technical features

### 2. Technical Analysis (25% weight)
- Moving average crossovers
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands

### 3. Sentiment Analysis (25% weight)
- News article sentiment (NewsAPI, Yahoo Finance)
- Reddit mentions and sentiment (r/wallstreetbets, r/stocks)
- Twitter sentiment (optional)
- VADER and TextBlob for sentiment scoring

### 4. Volume Analysis (15% weight)
- Volume ratio vs. average
- Price-volume correlation

## Safety Features

- **Risk Management**: Automatic position sizing and portfolio risk limits
- **Stop Loss**: Automatic stop-loss orders at 2% loss
- **Take Profit**: Automatic profit-taking at 5% gain
- **Position Limits**: Maximum 10 positions, 15% per position
- **Dry Run Mode**: Test without real money
- **Paper Trading**: Full Alpaca paper trading support

## Database

The bot stores all activity in SQLite:

- **Trades**: All executed trades
- **Signals**: Trading signals generated
- **Sentiment**: Sentiment analysis results
- **Performance**: Daily performance metrics
- **Predictions**: ML model predictions

Access the database:

```python
from src.data.database import DatabaseManager

db = DatabaseManager()
trades = db.get_recent_trades(limit=100)
```

## Performance Monitoring

View logs:

```bash
tail -f logs/trading_bot_*.log  # All logs
tail -f logs/trades_*.log       # Trade logs only
tail -f logs/errors_*.log       # Errors only
```

## Backtesting Examples

### Single Symbol Backtest

```python
from src.backtesting.backtest_engine import BacktestEngine
from src.data.data_manager import DataManager
from src.alpaca.client import AlpacaClient
from datetime import datetime, timedelta

alpaca_client = AlpacaClient()
data_manager = DataManager(alpaca_client)
backtest = BacktestEngine(data_manager)

end = datetime.now()
start = end - timedelta(days=365)

results = backtest.run_backtest('AAPL', start, end)
```

### Monte Carlo Simulation

```python
monte_carlo = backtest.monte_carlo_simulation(
    results['equity_curve'],
    num_simulations=1000
)
print(f"Probability of profit: {monte_carlo['probability_profit']:.2%}")
```

## Development

### Run Tests

```bash
pytest tests/
```

### Code Structure

- Keep strategies in `src/strategy/`
- Add new ML models in `src/ml/models.py`
- Customize risk rules in `src/risk/risk_manager.py`

## Troubleshooting

### "No module named 'src'"

Make sure you're in the project root directory and the virtual environment is activated.

### "Authentication failed"

Check your Alpaca API keys in `.env` file.

### "Market is closed"

The bot will wait when the market is closed. US stock market hours: 9:30 AM - 4:00 PM ET.

### "Insufficient buying power"

Reduce `MAX_POSITION_SIZE` in `.env` or add more capital to your Alpaca account.

## Legal & Disclaimer

**IMPORTANT**: This software is for educational purposes only.

- ✅ Uses Alpaca's official, authorized API
- ✅ Fully legal for algorithmic trading
- ⚠️ Trading involves risk of loss
- ⚠️ Past performance does not guarantee future results
- ⚠️ Always start with paper trading
- ⚠️ Never invest more than you can afford to lose

The authors are not responsible for any financial losses incurred from using this software.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Open an issue on GitHub
- Check the logs in `logs/` directory
- Review Alpaca's documentation: [alpaca.markets/docs](https://alpaca.markets/docs)

## Roadmap

- [ ] Web dashboard for monitoring
- [ ] More ML models (LSTM, Transformer)
- [ ] Options trading support
- [ ] Multi-timeframe analysis
- [ ] Custom indicator builder
- [ ] Telegram/Discord notifications
- [ ] Cloud deployment guides

## Credits

Built with:
- [Alpaca](https://alpaca.markets) - Trading API
- [XGBoost](https://xgboost.readthedocs.io/) - ML models
- [NewsAPI](https://newsapi.org) - News data
- [PRAW](https://praw.readthedocs.io/) - Reddit API
- [Tweepy](https://www.tweepy.org/) - Twitter API

---

**Happy Trading! 📈**
