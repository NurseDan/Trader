"""ML models for price prediction and classification"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from src.ml.feature_engineering import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PricePredictionModel:
    """Predicts future price movements using ensemble methods"""

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.feature_columns = []
        self.is_trained = False

        self._initialize_model()
        logger.info(f"PricePredictionModel initialized with {model_type}")

    def _initialize_model(self):
        """Initialize the ML model"""
        if self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def prepare_data(self, df: pd.DataFrame, target_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        # Create features
        df_features = self.feature_engineer.create_all_features(df, include_targets=True)

        # Get feature columns
        self.feature_columns = self.feature_engineer.get_feature_columns(df_features, exclude_targets=True)

        # Get target
        target_col = f'target_return_{target_horizon}d'
        if target_col not in df_features.columns:
            raise ValueError(f"Target column {target_col} not found")

        X = df_features[self.feature_columns]
        y = df_features[target_col]

        return X, y

    def train(self, df: pd.DataFrame, target_horizon: int = 1, test_size: float = 0.2) -> Dict:
        """Train the model"""
        logger.info("Training price prediction model")

        X, y = self.prepare_data(df, target_horizon)

        # Time series split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        # Direction accuracy
        train_direction_acc = accuracy_score(y_train > 0, train_pred > 0)
        test_direction_acc = accuracy_score(y_test > 0, test_pred > 0)

        results = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_direction_accuracy': train_direction_acc,
            'test_direction_accuracy': test_direction_acc,
            'n_features': len(self.feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }

        logger.info(f"Training complete - Test RMSE: {test_rmse:.6f}, Direction Acc: {test_direction_acc:.2%}")

        return results

    def predict(self, df: pd.DataFrame) -> Dict:
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Create features
        df_features = self.feature_engineer.create_all_features(df, include_targets=False)

        # Get latest data point
        X = df_features[self.feature_columns].iloc[[-1]]
        X_scaled = self.scaler.transform(X)

        # Predict
        prediction = self.model.predict(X_scaled)[0]

        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            top_features = []

        return {
            'predicted_return': prediction,
            'predicted_direction': 'up' if prediction > 0 else 'down',
            'confidence': abs(prediction),
            'top_features': top_features
        }

    def save(self, path: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']

        logger.info(f"Model loaded from {path}")


class DirectionClassifier:
    """Classifies whether price will go up or down"""

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.feature_columns = []
        self.is_trained = False

        self._initialize_model()
        logger.info(f"DirectionClassifier initialized with {model_type}")

    def _initialize_model(self):
        """Initialize the ML model"""
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, df: pd.DataFrame, target_horizon: int = 1, test_size: float = 0.2) -> Dict:
        """Train the classifier"""
        logger.info("Training direction classifier")

        # Create features
        df_features = self.feature_engineer.create_all_features(df, include_targets=True)

        # Get feature columns
        self.feature_columns = self.feature_engineer.get_feature_columns(df_features, exclude_targets=True)

        # Get target
        target_col = f'target_direction_{target_horizon}d'
        X = df_features[self.feature_columns]
        y = df_features[target_col]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)

        results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_features': len(self.feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }

        logger.info(f"Training complete - Test Accuracy: {test_acc:.2%}")

        return results

    def predict(self, df: pd.DataFrame) -> Dict:
        """Predict direction for new data"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Create features
        df_features = self.feature_engineer.create_all_features(df, include_targets=False)

        # Get latest data point
        X = df_features[self.feature_columns].iloc[[-1]]
        X_scaled = self.scaler.transform(X)

        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        return {
            'direction': 'up' if prediction == 1 else 'down',
            'confidence': max(probabilities),
            'probability_up': probabilities[1] if len(probabilities) > 1 else 0,
            'probability_down': probabilities[0] if len(probabilities) > 0 else 0
        }

    def save(self, path: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']

        logger.info(f"Model loaded from {path}")
