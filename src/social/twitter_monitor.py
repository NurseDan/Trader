"""Twitter/X monitoring using Tweepy"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import tweepy

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TwitterMonitor:
    """Monitor Twitter/X for stock mentions and sentiment"""

    def __init__(self):
        self.client = None
        self.api = None

        if all([config.twitter.api_key, config.twitter.api_secret,
                config.twitter.access_token, config.twitter.access_secret]):
            try:
                # v2 API
                self.client = tweepy.Client(
                    bearer_token=None,
                    consumer_key=config.twitter.api_key,
                    consumer_secret=config.twitter.api_secret,
                    access_token=config.twitter.access_token,
                    access_token_secret=config.twitter.access_secret
                )

                # v1.1 API for additional features
                auth = tweepy.OAuth1UserHandler(
                    config.twitter.api_key,
                    config.twitter.api_secret,
                    config.twitter.access_token,
                    config.twitter.access_secret
                )
                self.api = tweepy.API(auth)

                logger.info("Twitter API initialized")

            except Exception as e:
                logger.warning(f"Failed to initialize Twitter API: {e}")
        else:
            logger.warning("Twitter API credentials not provided")

    def search_tweets(
        self,
        query: str,
        max_results: int = 100,
        days_back: int = 7
    ) -> List[Dict]:
        """Search for tweets matching a query"""
        if not self.client:
            logger.warning("Twitter API not initialized")
            return []

        tweets = []

        try:
            # Calculate start time
            start_time = datetime.utcnow() - timedelta(days=days_back)

            # Search recent tweets
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                start_time=start_time,
                tweet_fields=['created_at', 'public_metrics', 'author_id', 'text']
            )

            if response.data:
                for tweet in response.data:
                    tweets.append({
                        'source': 'twitter',
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'author_id': tweet.author_id,
                        'retweets': tweet.public_metrics.get('retweet_count', 0),
                        'likes': tweet.public_metrics.get('like_count', 0),
                        'replies': tweet.public_metrics.get('reply_count', 0),
                        'id': tweet.id
                    })

            logger.info(f"Found {len(tweets)} tweets for query: {query}")

        except Exception as e:
            logger.error(f"Error searching tweets: {e}")

        return tweets

    def get_stock_tweets(
        self,
        symbol: str,
        max_results: int = 100,
        days_back: int = 1
    ) -> List[Dict]:
        """Get tweets about a stock symbol"""
        # Build query with common stock ticker formats
        query = f"(${symbol} OR #{symbol} OR \"{symbol} stock\") -is:retweet lang:en"

        return self.search_tweets(query, max_results, days_back)

    def get_trending_topics(self, location_id: int = 1) -> List[Dict]:
        """Get trending topics (requires v1.1 API)"""
        if not self.api:
            logger.warning("Twitter v1.1 API not initialized")
            return []

        trends = []

        try:
            trends_result = self.api.get_place_trends(location_id)

            if trends_result:
                for trend in trends_result[0]['trends']:
                    trends.append({
                        'name': trend['name'],
                        'url': trend['url'],
                        'tweet_volume': trend.get('tweet_volume', 0)
                    })

            logger.info(f"Found {len(trends)} trending topics")

        except Exception as e:
            logger.error(f"Error fetching trending topics: {e}")

        return trends

    def get_user_tweets(self, username: str, max_results: int = 50) -> List[Dict]:
        """Get recent tweets from a specific user"""
        if not self.client:
            return []

        tweets = []

        try:
            # Get user ID
            user = self.client.get_user(username=username)

            if user.data:
                user_id = user.data.id

                # Get user's tweets
                response = self.client.get_users_tweets(
                    id=user_id,
                    max_results=min(max_results, 100),
                    tweet_fields=['created_at', 'public_metrics', 'text']
                )

                if response.data:
                    for tweet in response.data:
                        tweets.append({
                            'source': 'twitter',
                            'username': username,
                            'text': tweet.text,
                            'created_at': tweet.created_at,
                            'retweets': tweet.public_metrics.get('retweet_count', 0),
                            'likes': tweet.public_metrics.get('like_count', 0),
                            'id': tweet.id
                        })

                logger.info(f"Found {len(tweets)} tweets from @{username}")

        except Exception as e:
            logger.error(f"Error fetching tweets from @{username}: {e}")

        return tweets

    def monitor_influencers(
        self,
        usernames: List[str],
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """Monitor tweets from financial influencers"""
        all_tweets = []

        for username in usernames:
            tweets = self.get_user_tweets(username, max_results=20)

            # Filter by symbol if provided
            if symbol:
                tweets = [
                    t for t in tweets
                    if symbol.upper() in t['text'].upper()
                ]

            all_tweets.extend(tweets)

        logger.info(f"Monitored {len(usernames)} influencers, found {len(all_tweets)} relevant tweets")

        return all_tweets

    def get_tweet_engagement(self, tweets: List[Dict]) -> Dict:
        """Calculate engagement metrics for a list of tweets"""
        if not tweets:
            return {
                'total_tweets': 0,
                'total_likes': 0,
                'total_retweets': 0,
                'avg_likes': 0,
                'avg_retweets': 0
            }

        total_likes = sum(t.get('likes', 0) for t in tweets)
        total_retweets = sum(t.get('retweets', 0) for t in tweets)

        return {
            'total_tweets': len(tweets),
            'total_likes': total_likes,
            'total_retweets': total_retweets,
            'avg_likes': total_likes / len(tweets),
            'avg_retweets': total_retweets / len(tweets)
        }
