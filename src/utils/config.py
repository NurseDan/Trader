"""Configuration management using Pydantic"""

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class AlpacaConfig(BaseSettings):
    """Alpaca API configuration"""
    api_key: str = Field(..., alias="ALPACA_API_KEY")
    secret_key: str = Field(..., alias="ALPACA_SECRET_KEY")
    base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        alias="ALPACA_BASE_URL"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


class NewsConfig(BaseSettings):
    """News API configuration"""
    news_api_key: Optional[str] = Field(default=None, alias="NEWS_API_KEY")

    class Config:
        env_file = ".env"
        extra = "ignore"


class RedditConfig(BaseSettings):
    """Reddit API configuration"""
    client_id: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_ID")
    client_secret: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_SECRET")
    user_agent: str = Field(default="TradingBot/1.0", alias="REDDIT_USER_AGENT")

    class Config:
        env_file = ".env"
        extra = "ignore"


class TwitterConfig(BaseSettings):
    """Twitter API configuration"""
    api_key: Optional[str] = Field(default=None, alias="TWITTER_API_KEY")
    api_secret: Optional[str] = Field(default=None, alias="TWITTER_API_SECRET")
    access_token: Optional[str] = Field(default=None, alias="TWITTER_ACCESS_TOKEN")
    access_secret: Optional[str] = Field(default=None, alias="TWITTER_ACCESS_SECRET")

    class Config:
        env_file = ".env"
        extra = "ignore"


class TradingConfig(BaseSettings):
    """Trading parameters configuration"""
    max_position_size: float = Field(default=1000.0, alias="MAX_POSITION_SIZE")
    max_portfolio_risk: float = Field(default=0.02, alias="MAX_PORTFOLIO_RISK")
    trading_mode: str = Field(default="paper", alias="TRADING_MODE")
    database_url: str = Field(default="sqlite:///./trading_bot.db", alias="DATABASE_URL")

    class Config:
        env_file = ".env"
        extra = "ignore"


class Config:
    """Main configuration class"""
    def __init__(self):
        self.alpaca = AlpacaConfig()
        self.news = NewsConfig()
        self.reddit = RedditConfig()
        self.twitter = TwitterConfig()
        self.trading = TradingConfig()


# Global config instance
config = Config()
