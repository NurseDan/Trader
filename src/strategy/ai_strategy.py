"""AI-powered trading strategy combining ML predictions and sentiment analysis"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.alpaca.client import AlpacaClient
from src.data.data_manager import DataManager
from src.ml.models import PricePredictionModel, DirectionClassifier
from src.sentiment.analyzer import SentimentAnalyzer
from src.news.news_aggregator import NewsAggregator
from src.social.reddit_monitor import RedditMonitor
from src.social.twitter_monitor import TwitterMonitor
from src.data.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AITradingStrategy:
    """AI-powered trading strategy with multi-signal analysis"""

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        data_manager: DataManager,
        db_manager: DatabaseManager
    ):
        self.alpaca_client = alpaca_client
        self.data_manager = data_manager
        self.db_manager = db_manager

        # ML Models
        self.price_predictor = PricePredictionModel(model_type="xgboost")
        self.direction_classifier = DirectionClassifier(model_type="xgboost")

        # Sentiment Analysis
        self.sentiment_analyzer = SentimentAnalyzer(use_transformers=False)

        # News and Social
        self.news_aggregator = NewsAggregator()
        self.reddit_monitor = RedditMonitor()
        self.twitter_monitor = TwitterMonitor()

        # Signal weights
        self.weights = {
            'ml_prediction': 0.35,
            'technical': 0.25,
            'sentiment': 0.25,
            'volume': 0.15
        }

        logger.info("AITradingStrategy initialized")

    def train_models(self, symbol: str, days_back: int = 365) -> Dict:
        """Train ML models on historical data"""
        logger.info(f"Training models for {symbol}")

        # Get historical data
        start_date = datetime.now() - timedelta(days=days_back)
        df = self.data_manager.get_historical_data(symbol, start=start_date)

        if df.empty:
            logger.error(f"No data available for {symbol}")
            return {}

        # Add technical indicators
        df = self.data_manager.add_technical_indicators(df)

        # Train models
        price_results = self.price_predictor.train(df, target_horizon=1)
        direction_results = self.direction_classifier.train(df, target_horizon=1)

        # Save models
        self.price_predictor.save(f"models/saved_models/{symbol}_price_predictor.pkl")
        self.direction_classifier.save(f"models/saved_models/{symbol}_direction_classifier.pkl")

        logger.info(f"Models trained for {symbol}")

        return {
            'price_prediction': price_results,
            'direction_classification': direction_results
        }

    def analyze_symbol(self, symbol: str) -> Dict:
        """Comprehensive analysis of a symbol"""
        logger.info(f"Analyzing {symbol}")

        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'signals': {},
            'scores': {},
            'recommendation': 'hold',
            'confidence': 0.0
        }

        # 1. ML Predictions
        ml_signal = self._get_ml_signal(symbol)
        analysis['signals']['ml'] = ml_signal
        analysis['scores']['ml'] = ml_signal.get('score', 0.0)

        # 2. Technical Analysis
        technical_signal = self._get_technical_signal(symbol)
        analysis['signals']['technical'] = technical_signal
        analysis['scores']['technical'] = technical_signal.get('score', 0.0)

        # 3. Sentiment Analysis
        sentiment_signal = self._get_sentiment_signal(symbol)
        analysis['signals']['sentiment'] = sentiment_signal
        analysis['scores']['sentiment'] = sentiment_signal.get('score', 0.0)

        # 4. Volume Analysis
        volume_signal = self._get_volume_signal(symbol)
        analysis['signals']['volume'] = volume_signal
        analysis['scores']['volume'] = volume_signal.get('score', 0.0)

        # 5. Combine signals
        combined = self._combine_signals(analysis['scores'])
        analysis['recommendation'] = combined['recommendation']
        analysis['confidence'] = combined['confidence']
        analysis['overall_score'] = combined['overall_score']

        # Save signal to database
        self._save_signal(analysis)

        logger.info(
            f"{symbol} analysis complete: {analysis['recommendation']} "
            f"(confidence: {analysis['confidence']:.2%})"
        )

        return analysis

    def _get_ml_signal(self, symbol: str) -> Dict:
        """Get ML prediction signal"""
        try:
            # Get recent data
            df = self.data_manager.get_historical_data(symbol, days_back=100)

            if df.empty:
                return {'score': 0.0, 'reason': 'No data available'}

            # Add technical indicators
            df = self.data_manager.add_technical_indicators(df)

            # Try to load existing model
            try:
                self.price_predictor.load(f"models/saved_models/{symbol}_price_predictor.pkl")
                self.direction_classifier.load(f"models/saved_models/{symbol}_direction_classifier.pkl")
            except:
                logger.warning(f"No trained model found for {symbol}, training new one")
                self.train_models(symbol)

            # Make predictions
            price_pred = self.price_predictor.predict(df)
            direction_pred = self.direction_classifier.predict(df)

            # Combine predictions
            predicted_return = price_pred['predicted_return']
            direction_confidence = direction_pred['confidence']

            # Calculate score (-1 to 1)
            if direction_pred['direction'] == 'up':
                score = min(1.0, abs(predicted_return) * 10 * direction_confidence)
            else:
                score = -min(1.0, abs(predicted_return) * 10 * direction_confidence)

            return {
                'score': score,
                'predicted_return': predicted_return,
                'direction': direction_pred['direction'],
                'confidence': direction_confidence,
                'reason': f"ML predicts {direction_pred['direction']} with {direction_confidence:.1%} confidence"
            }

        except Exception as e:
            logger.error(f"ML signal error for {symbol}: {e}")
            return {'score': 0.0, 'reason': f'ML error: {str(e)}'}

    def _get_technical_signal(self, symbol: str) -> Dict:
        """Get technical analysis signal"""
        try:
            df = self.data_manager.get_historical_data(symbol, days_back=100)

            if df.empty or len(df) < 50:
                return {'score': 0.0, 'reason': 'Insufficient data'}

            df = self.data_manager.add_technical_indicators(df)
            latest = df.iloc[-1]

            score = 0.0
            signals = []

            # Moving Average signals
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                if latest['close'] > latest['sma_20']:
                    score += 0.25
                    signals.append('Price above SMA20')
                else:
                    score -= 0.25

                if latest['sma_20'] > latest['sma_50']:
                    score += 0.25
                    signals.append('SMA20 above SMA50')
                else:
                    score -= 0.25

            # RSI
            if 'rsi' in df.columns:
                rsi = latest['rsi']
                if rsi < 30:
                    score += 0.3
                    signals.append(f'RSI oversold ({rsi:.1f})')
                elif rsi > 70:
                    score -= 0.3
                    signals.append(f'RSI overbought ({rsi:.1f})')

            # MACD
            if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
                    score += 0.2
                    signals.append('MACD bullish')
                else:
                    score -= 0.2

            # Normalize score to -1 to 1
            score = max(-1, min(1, score))

            return {
                'score': score,
                'signals': signals,
                'reason': ', '.join(signals) if signals else 'Neutral technical indicators'
            }

        except Exception as e:
            logger.error(f"Technical signal error for {symbol}: {e}")
            return {'score': 0.0, 'reason': f'Technical error: {str(e)}'}

    def _get_sentiment_signal(self, symbol: str) -> Dict:
        """Get sentiment analysis signal"""
        try:
            # Get news
            news_articles = self.news_aggregator.get_stock_news(symbol, days_back=3, max_articles=20)

            # Get Reddit mentions
            reddit_posts = self.reddit_monitor.get_stock_mentions(symbol, limit=50, time_filter="day")

            # Get Twitter mentions
            twitter_posts = self.twitter_monitor.get_stock_tweets(symbol, max_results=50, days_back=1)

            # Analyze sentiment
            overall_sentiment = self.sentiment_analyzer.get_overall_sentiment(
                news_articles=news_articles,
                social_posts=reddit_posts + twitter_posts,
                news_weight=0.6,
                social_weight=0.4
            )

            score = overall_sentiment['overall_sentiment']

            # Save to database
            self.db_manager.save_sentiment({
                'symbol': symbol,
                'source': 'combined',
                'sentiment_score': score,
                'volume': len(news_articles) + len(reddit_posts) + len(twitter_posts),
                'timestamp': datetime.now()
            })

            return {
                'score': score,
                'news_sentiment': overall_sentiment['news_sentiment'],
                'social_sentiment': overall_sentiment['social_sentiment'],
                'news_count': len(news_articles),
                'social_count': len(reddit_posts) + len(twitter_posts),
                'reason': f"{overall_sentiment['label']} sentiment ({score:.2f})"
            }

        except Exception as e:
            logger.error(f"Sentiment signal error for {symbol}: {e}")
            return {'score': 0.0, 'reason': f'Sentiment error: {str(e)}'}

    def _get_volume_signal(self, symbol: str) -> Dict:
        """Get volume analysis signal"""
        try:
            df = self.data_manager.get_historical_data(symbol, days_back=30)

            if df.empty or 'volume' not in df.columns:
                return {'score': 0.0, 'reason': 'No volume data'}

            latest = df.iloc[-1]
            avg_volume = df['volume'].tail(20).mean()

            volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 1

            # Higher volume can indicate stronger moves
            if volume_ratio > 1.5:
                score = min(1.0, (volume_ratio - 1) / 2)
                reason = f'High volume ({volume_ratio:.1f}x average)'
            elif volume_ratio < 0.5:
                score = -0.3
                reason = f'Low volume ({volume_ratio:.1f}x average)'
            else:
                score = 0.0
                reason = 'Normal volume'

            return {
                'score': score,
                'volume_ratio': volume_ratio,
                'reason': reason
            }

        except Exception as e:
            logger.error(f"Volume signal error for {symbol}: {e}")
            return {'score': 0.0, 'reason': f'Volume error: {str(e)}'}

    def _combine_signals(self, scores: Dict) -> Dict:
        """Combine all signals into final recommendation"""
        # Weighted average
        overall_score = 0.0

        for signal_type, weight in self.weights.items():
            score = scores.get(signal_type, 0.0)
            overall_score += score * weight

        # Determine recommendation
        if overall_score > 0.3:
            recommendation = 'buy'
        elif overall_score < -0.3:
            recommendation = 'sell'
        else:
            recommendation = 'hold'

        # Confidence based on agreement between signals
        signal_values = [scores.get(k, 0.0) for k in self.weights.keys()]
        signal_agreement = 1 - np.std(signal_values) if signal_values else 0
        confidence = abs(overall_score) * signal_agreement

        return {
            'overall_score': overall_score,
            'recommendation': recommendation,
            'confidence': min(1.0, confidence)
        }

    def _save_signal(self, analysis: Dict):
        """Save trading signal to database"""
        try:
            signal_data = {
                'symbol': analysis['symbol'],
                'signal_type': analysis['recommendation'],
                'strength': analysis['confidence'],
                'strategy': 'ai_multi_signal',
                'price': self.data_manager.get_latest_price(analysis['symbol']),
                'reason': f"Score: {analysis['overall_score']:.2f}, " +
                         f"ML: {analysis['scores'].get('ml', 0):.2f}, " +
                         f"Technical: {analysis['scores'].get('technical', 0):.2f}, " +
                         f"Sentiment: {analysis['scores'].get('sentiment', 0):.2f}",
                'timestamp': datetime.now(),
                'executed': False
            }

            self.db_manager.save_signal(signal_data)

        except Exception as e:
            logger.error(f"Error saving signal: {e}")

    def screen_stocks(self, symbols: List[str], min_confidence: float = 0.5) -> List[Dict]:
        """Screen multiple stocks and return ranked recommendations"""
        logger.info(f"Screening {len(symbols)} stocks")

        analyses = []

        for symbol in symbols:
            try:
                analysis = self.analyze_symbol(symbol)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        # Filter by confidence and sort
        buy_candidates = [
            a for a in analyses
            if a['recommendation'] == 'buy' and a['confidence'] >= min_confidence
        ]
        buy_candidates.sort(key=lambda x: x['confidence'], reverse=True)

        sell_candidates = [
            a for a in analyses
            if a['recommendation'] == 'sell' and a['confidence'] >= min_confidence
        ]
        sell_candidates.sort(key=lambda x: x['confidence'], reverse=True)

        logger.info(
            f"Screening complete: {len(buy_candidates)} buy signals, "
            f"{len(sell_candidates)} sell signals"
        )

        return {
            'buy_candidates': buy_candidates,
            'sell_candidates': sell_candidates,
            'all_analyses': analyses
        }
