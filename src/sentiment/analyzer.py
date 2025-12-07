"""Sentiment analysis using multiple NLP approaches"""

from typing import List, Dict, Optional
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """Multi-model sentiment analysis for financial text"""

    def __init__(self, use_transformers: bool = False):
        self.vader = SentimentIntensityAnalyzer()
        self.use_transformers = use_transformers
        self.transformer_pipeline = None

        if use_transformers:
            try:
                # Use FinBERT for financial sentiment
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    max_length=512,
                    truncation=True
                )
                logger.info("FinBERT sentiment model loaded")
            except Exception as e:
                logger.warning(f"Failed to load transformer model: {e}")
                self.use_transformers = False

        logger.info("SentimentAnalyzer initialized")

    def analyze_text(self, text: str) -> Dict:
        """Analyze sentiment of a single text"""
        if not text or not text.strip():
            return {
                'compound': 0.0,
                'positive': 0.0,
                'neutral': 1.0,
                'negative': 0.0,
                'label': 'neutral'
            }

        # VADER sentiment (good for social media)
        vader_scores = self.vader.polarity_scores(text)

        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity

        # Combine scores
        compound = (vader_scores['compound'] + textblob_polarity) / 2

        result = {
            'compound': compound,
            'positive': vader_scores['pos'],
            'neutral': vader_scores['neu'],
            'negative': vader_scores['neg'],
            'vader_compound': vader_scores['compound'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'label': self._get_label(compound)
        }

        # Add transformer sentiment if available
        if self.use_transformers and self.transformer_pipeline:
            try:
                transformer_result = self.transformer_pipeline(text[:512])[0]
                result['transformer_label'] = transformer_result['label']
                result['transformer_score'] = transformer_result['score']

                # Adjust compound with transformer
                if transformer_result['label'] == 'positive':
                    result['compound'] = (compound + transformer_result['score']) / 2
                elif transformer_result['label'] == 'negative':
                    result['compound'] = (compound - transformer_result['score']) / 2

            except Exception as e:
                logger.warning(f"Transformer sentiment failed: {e}")

        return result

    def _get_label(self, compound: float) -> str:
        """Convert compound score to label"""
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for multiple texts"""
        return [self.analyze_text(text) for text in texts]

    def analyze_articles(self, articles: List[Dict]) -> Dict:
        """Analyze sentiment of news articles"""
        if not articles:
            return {
                'avg_sentiment': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_count': 0
            }

        sentiments = []

        for article in articles:
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = self.analyze_text(text)
            sentiments.append(sentiment['compound'])
            article['sentiment'] = sentiment

        positive_count = sum(1 for s in sentiments if s > 0.05)
        negative_count = sum(1 for s in sentiments if s < -0.05)
        neutral_count = len(sentiments) - positive_count - negative_count

        return {
            'avg_sentiment': np.mean(sentiments) if sentiments else 0.0,
            'median_sentiment': np.median(sentiments) if sentiments else 0.0,
            'std_sentiment': np.std(sentiments) if sentiments else 0.0,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_count': len(sentiments),
            'positive_ratio': positive_count / len(sentiments) if sentiments else 0.0,
            'negative_ratio': negative_count / len(sentiments) if sentiments else 0.0,
            'articles_with_sentiment': articles
        }

    def analyze_social_posts(self, posts: List[Dict]) -> Dict:
        """Analyze sentiment of social media posts (Reddit/Twitter)"""
        if not posts:
            return {
                'avg_sentiment': 0.0,
                'weighted_sentiment': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_count': 0
            }

        sentiments = []
        weights = []

        for post in posts:
            # Get text from different sources
            if 'text' in post:
                text = post['text']
            elif 'title' in post:
                text = f"{post.get('title', '')} {post.get('text', '')}"
            else:
                continue

            sentiment = self.analyze_text(text)
            sentiments.append(sentiment['compound'])

            # Weight by engagement (likes, score, etc.)
            weight = 1.0
            if 'score' in post:
                weight = max(1, post['score'])
            elif 'likes' in post:
                weight = max(1, post['likes'])

            weights.append(weight)
            post['sentiment'] = sentiment

        if not sentiments:
            return {
                'avg_sentiment': 0.0,
                'weighted_sentiment': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_count': 0
            }

        # Calculate weighted sentiment
        weighted_sentiment = np.average(sentiments, weights=weights)

        positive_count = sum(1 for s in sentiments if s > 0.05)
        negative_count = sum(1 for s in sentiments if s < -0.05)
        neutral_count = len(sentiments) - positive_count - negative_count

        return {
            'avg_sentiment': np.mean(sentiments),
            'median_sentiment': np.median(sentiments),
            'weighted_sentiment': weighted_sentiment,
            'std_sentiment': np.std(sentiments),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_count': len(sentiments),
            'positive_ratio': positive_count / len(sentiments),
            'negative_ratio': negative_count / len(sentiments),
            'posts_with_sentiment': posts
        }

    def get_overall_sentiment(
        self,
        news_articles: Optional[List[Dict]] = None,
        social_posts: Optional[List[Dict]] = None,
        news_weight: float = 0.6,
        social_weight: float = 0.4
    ) -> Dict:
        """Combine sentiment from multiple sources"""
        news_sentiment = 0.0
        social_sentiment = 0.0

        if news_articles:
            news_result = self.analyze_articles(news_articles)
            news_sentiment = news_result['avg_sentiment']

        if social_posts:
            social_result = self.analyze_social_posts(social_posts)
            social_sentiment = social_result['weighted_sentiment']

        # Weighted average
        if news_articles and social_posts:
            overall_sentiment = (news_sentiment * news_weight) + (social_sentiment * social_weight)
        elif news_articles:
            overall_sentiment = news_sentiment
        elif social_posts:
            overall_sentiment = social_sentiment
        else:
            overall_sentiment = 0.0

        return {
            'overall_sentiment': overall_sentiment,
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'label': self._get_label(overall_sentiment),
            'confidence': abs(overall_sentiment)
        }

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords from text"""
        blob = TextBlob(text)

        # Get noun phrases
        noun_phrases = list(blob.noun_phrases)

        # Simple frequency count
        from collections import Counter
        word_freq = Counter(noun_phrases)

        return [word for word, _ in word_freq.most_common(top_n)]
