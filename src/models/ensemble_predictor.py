"""
Ensemble Model for Fantasy Football Predictions
Combines multiple models for robust predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path

# ML Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import get_logger, log_execution_time, log_model_training
from src.utils.config import get_config
from src.data.preprocessing.feature_engineering import FeatureEngineer

logger = get_logger()


class LSTMPredictor(nn.Module):
    """LSTM model for sequence prediction"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.dropout(last_out)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class EnsemblePredictor:
    """Ensemble model combining multiple algorithms"""
    
    def __init__(self):
        """Initialize ensemble predictor"""
        self.config = get_config()
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.model_weights = {}
        self.is_trained = False
        
        # Model save directory
        self.model_dir = self.config.data.models_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Ensemble Predictor initialized")
    
    def _initialize_models(self) -> Dict:
        """Initialize all models in the ensemble"""
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'neural_net': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
        
        return models
    
    @log_model_training("Ensemble Model")
    def train(self, X: pd.DataFrame, y: pd.Series, validate: bool = True) -> Dict[str, float]:
        """
        Train all models in the ensemble
        
        Args:
            X: Feature matrix
            y: Target values
            validate: Whether to perform cross-validation
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training ensemble with {len(X)} samples")
        
        # Initialize models
        self.models = self._initialize_models()
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train and evaluate each model
        model_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X, y)
                
                # Cross-validation if requested
                if validate:
                    scores = cross_val_score(
                        model, X, y, 
                        cv=tscv, 
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1
                    )
                    model_scores[name] = -scores.mean()
                    logger.info(f"{name} MAE: {model_scores[name]:.2f}")
                else:
                    # Simple train score
                    y_pred = model.predict(X)
                    model_scores[name] = mean_absolute_error(y, y_pred)
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                model_scores[name] = float('inf')
        
        # Calculate model weights based on performance
        self._calculate_model_weights(model_scores)
        
        # Save models
        self.save_models()
        
        self.is_trained = True
        
        metrics = {
            'model_scores': model_scores,
            'best_model': min(model_scores, key=model_scores.get),
            'ensemble_mae': np.average(list(model_scores.values())),
            'models_trained': len(self.models)
        }
        
        return metrics
    
    def _calculate_model_weights(self, scores: Dict[str, float]):
        """Calculate ensemble weights based on model performance"""
        # Invert scores (lower is better for MAE)
        inverted_scores = {k: 1.0 / (v + 1e-6) for k, v in scores.items()}
        
        # Normalize to sum to 1
        total = sum(inverted_scores.values())
        self.model_weights = {k: v / total for k, v in inverted_scores.items()}
        
        logger.info(f"Model weights: {self.model_weights}")
    
    def predict(self, X: pd.DataFrame, return_all: bool = False) -> np.ndarray:
        """
        Make predictions using the ensemble
        
        Args:
            X: Feature matrix
            return_all: Return all model predictions
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            self.load_models()
        
        predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")
                predictions[name] = np.zeros(len(X))
        
        if return_all:
            return predictions
        
        # Weighted average ensemble
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            weight = self.model_weights.get(name, 1.0 / len(self.models))
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with predictions and confidence bounds
        """
        # Get all model predictions
        all_predictions = self.predict(X, return_all=True)
        
        # Convert to array
        pred_array = np.array(list(all_predictions.values())).T
        
        # Calculate statistics
        mean_pred = np.mean(pred_array, axis=1)
        std_pred = np.std(pred_array, axis=1)
        
        return {
            'prediction': mean_pred,
            'lower_bound': mean_pred - 1.96 * std_pred,
            'upper_bound': mean_pred + 1.96 * std_pred,
            'std': std_pred,
            'confidence_80_lower': mean_pred - 1.28 * std_pred,
            'confidence_80_upper': mean_pred + 1.28 * std_pred
        }
    
    def save_models(self):
        """Save all trained models"""
        for name, model in self.models.items():
            model_path = self.model_dir / f"{name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")
        
        # Save weights
        weights_path = self.model_dir / "model_weights.pkl"
        joblib.dump(self.model_weights, weights_path)
    
    def load_models(self):
        """Load saved models"""
        try:
            # Load models
            for model_file in self.model_dir.glob("*_model.pkl"):
                name = model_file.stem.replace("_model", "")
                self.models[name] = joblib.load(model_file)
            
            # Load weights
            weights_path = self.model_dir / "model_weights.pkl"
            if weights_path.exists():
                self.model_weights = joblib.load(weights_path)
            else:
                # Equal weights if not found
                self.model_weights = {k: 1.0/len(self.models) for k in self.models.keys()}
            
            self.is_trained = True
            logger.info(f"Loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.is_trained = False
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True values
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        
        metrics = {
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / (y + 1e-6))) * 100,
            'directional_accuracy': np.mean((predictions > y.median()) == (y > y.median()))
        }
        
        return metrics
    
    def predict_player(self, player_id: str, week: int, season: int = 2025) -> Dict:
        """
        Predict fantasy points for a specific player
        
        Args:
            player_id: Player ID
            week: Week number
            season: Season year
            
        Returns:
            Prediction dictionary
        """
        # Create features
        features = self.feature_engineer.create_player_features(
            player_id, 
            target_week=week,
            target_season=season
        )
        
        if not features:
            logger.warning(f"Could not create features for player {player_id}")
            return {}
        
        # Prepare for prediction
        feature_df = pd.DataFrame([features])
        
        # Get prediction with confidence
        results = self.predict_with_confidence(feature_df)
        
        return {
            'player_id': player_id,
            'week': week,
            'season': season,
            'predicted_points': float(results['prediction'][0]),
            'lower_bound': float(results['lower_bound'][0]),
            'upper_bound': float(results['upper_bound'][0]),
            'confidence_80_lower': float(results['confidence_80_lower'][0]),
            'confidence_80_upper': float(results['confidence_80_upper'][0]),
            'uncertainty': float(results['std'][0])
        }


if __name__ == "__main__":
    # Test the ensemble predictor
    predictor = EnsemblePredictor()
    
    # Load or create training data
    config = get_config()
    training_data_path = config.data.processed_dir / 'training_data.parquet'
    
    if training_data_path.exists():
        logger.info("Loading existing training data...")
        df = pd.read_parquet(training_data_path)
        
        # Prepare for modeling
        X, y = predictor.feature_engineer.prepare_for_modeling(df)
        
        # Train models
        logger.info("Training ensemble models...")
        metrics = predictor.train(X, y, validate=True)
        logger.info(f"Training complete. Metrics: {metrics}")
        
        # Test prediction for a player
        test_prediction = predictor.predict_player("00-0034796", week=1, season=2025)
        logger.info(f"Test prediction: {test_prediction}")
    else:
        logger.info("No training data found. Run feature engineering first.")