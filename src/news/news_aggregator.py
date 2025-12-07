"""News aggregation from multiple sources"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import requests
from newsapi import NewsApiClient

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NewsAggregator:
    """Aggregates news from multiple sources"""

    def __init__(self):
        self.news_api = None
        if config.news.news_api_key:
            try:
                self.news_api = NewsApiClient(api_key=config.news.news_api_key)
                logger.info("NewsAPI initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NewsAPI: {e}")

    def get_stock_news(
        self,
        symbol: str,
        days_back: int = 7,
        max_articles: int = 50
    ) -> List[Dict]:
        """Get news articles about a stock"""
        all_articles = []

        # Get company name for better search (simplified)
        query = f"{symbol} stock OR {symbol} shares"

        # NewsAPI
        if self.news_api:
            try:
                from_date = datetime.now() - timedelta(days=days_back)
                response = self.news_api.get_everything(
                    q=query,
                    from_param=from_date.isoformat(),
                    language='en',
                    sort_by='relevancy',
                    page_size=min(max_articles, 100)
                )

                if response['status'] == 'ok':
                    for article in response['articles']:
                        all_articles.append({
                            'source': 'newsapi',
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'source_name': article.get('source', {}).get('name', ''),
                        })

                logger.info(f"Fetched {len(all_articles)} articles from NewsAPI for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching from NewsAPI: {e}")

        # Alpha Vantage News (free alternative)
        try:
            articles = self._get_alphavantage_news(symbol, max_articles)
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"Error fetching from Alpha Vantage: {e}")

        # Yahoo Finance RSS (free fallback)
        try:
            articles = self._get_yahoo_news(symbol, max_articles)
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"Error fetching from Yahoo Finance: {e}")

        return all_articles[:max_articles]

    def _get_alphavantage_news(self, symbol: str, limit: int = 20) -> List[Dict]:
        """Get news from Alpha Vantage (free, no API key for news)"""
        articles = []
        try:
            # Alpha Vantage News & Sentiment API
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'limit': limit,
                'apikey': 'demo'  # Free tier available
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'feed' in data:
                    for item in data['feed']:
                        articles.append({
                            'source': 'alphavantage',
                            'title': item.get('title', ''),
                            'description': item.get('summary', ''),
                            'content': item.get('summary', ''),
                            'url': item.get('url', ''),
                            'published_at': item.get('time_published', ''),
                            'source_name': item.get('source', ''),
                            'sentiment_score': item.get('overall_sentiment_score', 0),
                        })

            logger.info(f"Fetched {len(articles)} articles from Alpha Vantage for {symbol}")

        except Exception as e:
            logger.warning(f"Alpha Vantage news fetch failed: {e}")

        return articles

    def _get_yahoo_news(self, symbol: str, limit: int = 20) -> List[Dict]:
        """Get news from Yahoo Finance"""
        articles = []
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            news = ticker.news

            if news:
                for item in news[:limit]:
                    articles.append({
                        'source': 'yahoo',
                        'title': item.get('title', ''),
                        'description': item.get('summary', ''),
                        'content': item.get('summary', ''),
                        'url': item.get('link', ''),
                        'published_at': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat(),
                        'source_name': item.get('publisher', ''),
                    })

            logger.info(f"Fetched {len(articles)} articles from Yahoo Finance for {symbol}")

        except Exception as e:
            logger.warning(f"Yahoo Finance news fetch failed: {e}")

        return articles

    def get_market_news(self, max_articles: int = 30) -> List[Dict]:
        """Get general market news"""
        articles = []

        if self.news_api:
            try:
                response = self.news_api.get_top_headlines(
                    category='business',
                    language='en',
                    page_size=max_articles
                )

                if response['status'] == 'ok':
                    for article in response['articles']:
                        articles.append({
                            'source': 'newsapi',
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'source_name': article.get('source', {}).get('name', ''),
                        })

                logger.info(f"Fetched {len(articles)} market news articles")

            except Exception as e:
                logger.error(f"Error fetching market news: {e}")

        return articles

    def search_news(self, query: str, days_back: int = 7, max_articles: int = 20) -> List[Dict]:
        """Search for news with custom query"""
        articles = []

        if self.news_api:
            try:
                from_date = datetime.now() - timedelta(days=days_back)
                response = self.news_api.get_everything(
                    q=query,
                    from_param=from_date.isoformat(),
                    language='en',
                    sort_by='relevancy',
                    page_size=max_articles
                )

                if response['status'] == 'ok':
                    for article in response['articles']:
                        articles.append({
                            'source': 'newsapi',
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'source_name': article.get('source', {}).get('name', ''),
                        })

                logger.info(f"Fetched {len(articles)} articles for query: {query}")

            except Exception as e:
                logger.error(f"Error searching news: {e}")

        return articles

    def filter_relevant_news(self, articles: List[Dict], keywords: List[str]) -> List[Dict]:
        """Filter articles by keywords"""
        filtered = []

        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}".lower()

            if any(keyword.lower() in text for keyword in keywords):
                filtered.append(article)

        logger.info(f"Filtered {len(filtered)}/{len(articles)} articles by keywords")
        return filtered
