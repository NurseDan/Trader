"""Reddit sentiment monitoring using PRAW"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import praw

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RedditMonitor:
    """Monitor Reddit for stock mentions and sentiment"""

    def __init__(self):
        self.reddit = None

        if config.reddit.client_id and config.reddit.client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=config.reddit.client_id,
                    client_secret=config.reddit.client_secret,
                    user_agent=config.reddit.user_agent
                )
                logger.info("Reddit API initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit API: {e}")
        else:
            logger.warning("Reddit API credentials not provided")

    def get_stock_mentions(
        self,
        symbol: str,
        subreddits: Optional[List[str]] = None,
        limit: int = 100,
        time_filter: str = "day"
    ) -> List[Dict]:
        """Get mentions of a stock symbol from Reddit"""
        if not self.reddit:
            logger.warning("Reddit API not initialized")
            return []

        if subreddits is None:
            subreddits = ["wallstreetbets", "stocks", "investing", "stockmarket", "daytrading"]

        mentions = []

        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Search for symbol
                search_query = f"${symbol} OR {symbol}"

                for submission in subreddit.search(search_query, time_filter=time_filter, limit=limit):
                    mentions.append({
                        'source': 'reddit',
                        'subreddit': subreddit_name,
                        'title': submission.title,
                        'text': submission.selftext,
                        'score': submission.score,
                        'upvote_ratio': submission.upvote_ratio,
                        'num_comments': submission.num_comments,
                        'created_utc': datetime.fromtimestamp(submission.created_utc),
                        'url': f"https://reddit.com{submission.permalink}",
                        'author': str(submission.author),
                        'type': 'submission'
                    })

                logger.info(f"Found {len(mentions)} mentions of {symbol} in r/{subreddit_name}")

            except Exception as e:
                logger.error(f"Error fetching from r/{subreddit_name}: {e}")

        return mentions

    def get_trending_stocks(
        self,
        subreddits: Optional[List[str]] = None,
        limit: int = 50,
        time_filter: str = "day"
    ) -> Dict[str, int]:
        """Get trending stock mentions from Reddit"""
        if not self.reddit:
            logger.warning("Reddit API not initialized")
            return {}

        if subreddits is None:
            subreddits = ["wallstreetbets", "stocks"]

        stock_mentions = {}

        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                for submission in subreddit.hot(limit=limit):
                    # Extract stock symbols (simple regex for $SYMBOL)
                    import re
                    text = f"{submission.title} {submission.selftext}"
                    symbols = re.findall(r'\$([A-Z]{1,5})\b', text)

                    for symbol in symbols:
                        stock_mentions[symbol] = stock_mentions.get(symbol, 0) + 1

                logger.info(f"Scanned r/{subreddit_name} for trending stocks")

            except Exception as e:
                logger.error(f"Error fetching from r/{subreddit_name}: {e}")

        # Sort by mentions
        trending = dict(sorted(stock_mentions.items(), key=lambda x: x[1], reverse=True))

        logger.info(f"Found {len(trending)} trending stocks")
        return trending

    def get_subreddit_sentiment(
        self,
        symbol: str,
        subreddit_name: str = "wallstreetbets",
        limit: int = 50
    ) -> Dict:
        """Get sentiment for a stock from a specific subreddit"""
        if not self.reddit:
            return {'mentions': 0, 'avg_score': 0, 'total_comments': 0}

        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            search_query = f"${symbol} OR {symbol}"

            total_score = 0
            total_comments = 0
            count = 0

            for submission in subreddit.search(search_query, time_filter="day", limit=limit):
                total_score += submission.score
                total_comments += submission.num_comments
                count += 1

            if count > 0:
                return {
                    'mentions': count,
                    'avg_score': total_score / count,
                    'total_comments': total_comments,
                    'avg_comments': total_comments / count
                }

        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol} in r/{subreddit_name}: {e}")

        return {'mentions': 0, 'avg_score': 0, 'total_comments': 0}

    def get_hot_posts(self, subreddit_name: str = "wallstreetbets", limit: int = 25) -> List[Dict]:
        """Get hot posts from a subreddit"""
        if not self.reddit:
            return []

        posts = []

        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            for submission in subreddit.hot(limit=limit):
                posts.append({
                    'subreddit': subreddit_name,
                    'title': submission.title,
                    'text': submission.selftext,
                    'score': submission.score,
                    'upvote_ratio': submission.upvote_ratio,
                    'num_comments': submission.num_comments,
                    'created_utc': datetime.fromtimestamp(submission.created_utc),
                    'url': f"https://reddit.com{submission.permalink}",
                    'author': str(submission.author)
                })

            logger.info(f"Fetched {len(posts)} hot posts from r/{subreddit_name}")

        except Exception as e:
            logger.error(f"Error fetching hot posts: {e}")

        return posts

    def get_comments(self, submission_id: str, limit: int = 100) -> List[Dict]:
        """Get comments from a specific submission"""
        if not self.reddit:
            return []

        comments = []

        try:
            submission = self.reddit.submission(id=submission_id)
            submission.comments.replace_more(limit=0)

            for comment in submission.comments.list()[:limit]:
                comments.append({
                    'text': comment.body,
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(comment.created_utc),
                    'author': str(comment.author)
                })

            logger.info(f"Fetched {len(comments)} comments")

        except Exception as e:
            logger.error(f"Error fetching comments: {e}")

        return comments
