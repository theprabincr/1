"""
XGBoost Machine Learning Model for Sports Betting Predictions
=============================================================
Implements real ML using XGBoost trained on historical game data.

ENHANCED VERSION - Supports ALL THREE MARKETS:
1. Moneyline (Home Win Probability)
2. Spread (Cover Probability)
3. Totals (Over/Under Probability)

Key Features:
1. Historical data collection from ESPN with outcomes for all markets
2. Comprehensive feature engineering
3. Three separate XGBoost models for each market
4. Model persistence (save/load)
5. Weekly retraining support
6. Backtesting infrastructure for all markets
"""
import logging
import os
import json
import pickle
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import asyncio

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error
import joblib

logger = logging.getLogger(__name__)

# Model storage directory
MODEL_DIR = Path(__file__).parent / "ml_models"
MODEL_DIR.mkdir(exist_ok=True)

# Feature names for consistent ordering
FEATURE_NAMES = [
    # Team strength features
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_win_pct",
    "away_win_pct",
    "win_pct_diff",
    
    # Recent form features
    "home_last10_wins",
    "away_last10_wins",
    "home_streak",
    "away_streak",
    "home_avg_margin",
    "away_avg_margin",
    "margin_diff",
    
    # Scoring features
    "home_avg_pts",
    "away_avg_pts",
    "home_avg_pts_allowed",
    "away_avg_pts_allowed",
    "home_net_rating",
    "away_net_rating",
    
    # Context features
    "home_rest_days",
    "away_rest_days",
    "rest_advantage",
    "is_back_to_back_home",
    "is_back_to_back_away",
    
    # Odds/Market features
    "home_ml_odds",
    "away_ml_odds",
    "implied_home_prob",
    "spread",
    "total_line",
    
    # Head-to-head
    "h2h_home_wins",
    "h2h_total_games",
    
    # NEW: Totals-specific features
    "combined_avg_pts",  # home_avg_pts + away_avg_pts
    "combined_pts_allowed",  # home_pts_allowed + away_pts_allowed
    "pace_factor",  # Estimated game pace
    "defensive_rating_diff",  # Defense comparison
]


class FeatureEngineering:
    """
    Comprehensive feature engineering for sports betting predictions.
    Extracts meaningful features from game and team data.
    """
    
    @staticmethod
    def extract_features(
        home_team_data: Dict,
        away_team_data: Dict,
        odds_data: Dict,
        context_data: Dict = None,
        h2h_data: Dict = None
    ) -> Dict[str, float]:
        """
        Extract all features for a game prediction.
        """
        features = {}
        
        # Default context and h2h if not provided
        context_data = context_data or {}
        h2h_data = h2h_data or {}
        
        # ===== TEAM STRENGTH FEATURES =====
        home_form = home_team_data.get("form", {})
        away_form = away_team_data.get("form", {})
        home_stats = home_team_data.get("stats", {})
        away_stats = away_team_data.get("stats", {})
        
        # ELO ratings
        features["home_elo"] = home_team_data.get("elo_rating", 1500)
        features["away_elo"] = away_team_data.get("elo_rating", 1500)
        features["elo_diff"] = features["home_elo"] - features["away_elo"]
        
        # Win percentages
        features["home_win_pct"] = home_form.get("win_pct", 0.5)
        features["away_win_pct"] = away_form.get("win_pct", 0.5)
        features["win_pct_diff"] = features["home_win_pct"] - features["away_win_pct"]
        
        # ===== RECENT FORM FEATURES =====
        features["home_last10_wins"] = home_form.get("wins", 5)
        features["away_last10_wins"] = away_form.get("wins", 5)
        features["home_streak"] = home_form.get("streak", 0)
        features["away_streak"] = away_form.get("streak", 0)
        features["home_avg_margin"] = home_form.get("avg_margin", 0)
        features["away_avg_margin"] = away_form.get("avg_margin", 0)
        features["margin_diff"] = features["home_avg_margin"] - features["away_avg_margin"]
        
        # ===== SCORING FEATURES =====
        features["home_avg_pts"] = home_stats.get("avg_points", 110)
        features["away_avg_pts"] = away_stats.get("avg_points", 110)
        features["home_avg_pts_allowed"] = home_stats.get("avg_points_allowed", 110)
        features["away_avg_pts_allowed"] = away_stats.get("avg_points_allowed", 110)
        features["home_net_rating"] = features["home_avg_pts"] - features["home_avg_pts_allowed"]
        features["away_net_rating"] = features["away_avg_pts"] - features["away_avg_pts_allowed"]
        
        # ===== CONTEXT FEATURES =====
        features["home_rest_days"] = context_data.get("home_rest_days", 2)
        features["away_rest_days"] = context_data.get("away_rest_days", 2)
        features["rest_advantage"] = features["home_rest_days"] - features["away_rest_days"]
        features["is_back_to_back_home"] = 1 if context_data.get("home_b2b", False) else 0
        features["is_back_to_back_away"] = 1 if context_data.get("away_b2b", False) else 0
        
        # ===== ODDS/MARKET FEATURES =====
        features["home_ml_odds"] = odds_data.get("home_ml_decimal", 1.91)
        features["away_ml_odds"] = odds_data.get("away_ml_decimal", 1.91)
        
        # Implied probability from odds
        if features["home_ml_odds"] > 1:
            features["implied_home_prob"] = 1 / features["home_ml_odds"]
        else:
            features["implied_home_prob"] = 0.5
            
        features["spread"] = odds_data.get("spread", 0)
        features["total_line"] = odds_data.get("total", 220)
        
        # ===== HEAD-TO-HEAD FEATURES =====
        features["h2h_home_wins"] = h2h_data.get("home_wins", 0)
        features["h2h_total_games"] = h2h_data.get("total_games", 0)
        
        # ===== TOTALS-SPECIFIC FEATURES =====
        features["combined_avg_pts"] = features["home_avg_pts"] + features["away_avg_pts"]
        features["combined_pts_allowed"] = features["home_avg_pts_allowed"] + features["away_avg_pts_allowed"]
        features["pace_factor"] = (features["combined_avg_pts"] + features["combined_pts_allowed"]) / 4
        features["defensive_rating_diff"] = features["home_avg_pts_allowed"] - features["away_avg_pts_allowed"]
        
        return features
    
    @staticmethod
    def features_to_array(features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array in consistent order."""
        return np.array([features.get(name, 0) for name in FEATURE_NAMES])


class XGBoostPredictor:
    """
    ENHANCED XGBoost-based predictor for sports betting.
    Now supports ALL THREE MARKETS:
    - Moneyline (home win probability)
    - Spread (home cover probability)
    - Totals (over probability)
    """
    
    def __init__(self, sport_key: str = "basketball_nba"):
        self.sport_key = sport_key
        
        # Three separate models for each market
        self.moneyline_model: Optional[XGBClassifier] = None
        self.spread_model: Optional[XGBClassifier] = None
        self.totals_model: Optional[XGBClassifier] = None
        
        # Also a regressor for predicting actual totals
        self.totals_regressor: Optional[XGBRegressor] = None
        
        self.scaler: Optional[StandardScaler] = None
        
        # Model paths
        self.ml_model_path = MODEL_DIR / f"xgboost_ml_{sport_key}.joblib"
        self.spread_model_path = MODEL_DIR / f"xgboost_spread_{sport_key}.joblib"
        self.totals_model_path = MODEL_DIR / f"xgboost_totals_{sport_key}.joblib"
        self.totals_reg_path = MODEL_DIR / f"xgboost_totals_reg_{sport_key}.joblib"
        self.scaler_path = MODEL_DIR / f"scaler_{sport_key}.joblib"
        self.metadata_path = MODEL_DIR / f"metadata_{sport_key}.json"
        
        self.feature_engineering = FeatureEngineering()
        self.is_loaded = False
        
        # Training metrics for each model
        self.ml_accuracy = 0.0
        self.spread_accuracy = 0.0
        self.totals_accuracy = 0.0
        self.totals_mae = 0.0  # Mean Absolute Error for totals prediction
        
        self.last_trained = None
        
        # For backward compatibility
        self.model = None
        self.training_accuracy = 0.0
        
    def load_model(self) -> bool:
        """Load trained models from disk if available."""
        try:
            models_loaded = 0
            
            # Load moneyline model
            if self.ml_model_path.exists():
                self.moneyline_model = joblib.load(self.ml_model_path)
                self.model = self.moneyline_model  # Backward compatibility
                models_loaded += 1
            
            # Load spread model
            if self.spread_model_path.exists():
                self.spread_model = joblib.load(self.spread_model_path)
                models_loaded += 1
            
            # Load totals model
            if self.totals_model_path.exists():
                self.totals_model = joblib.load(self.totals_model_path)
                models_loaded += 1
            
            # Load totals regressor
            if self.totals_reg_path.exists():
                self.totals_regressor = joblib.load(self.totals_reg_path)
                models_loaded += 1
            
            # Load scaler
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.ml_accuracy = metadata.get("ml_accuracy", metadata.get("accuracy", 0))
                    self.spread_accuracy = metadata.get("spread_accuracy", 0)
                    self.totals_accuracy = metadata.get("totals_accuracy", 0)
                    self.totals_mae = metadata.get("totals_mae", 0)
                    self.last_trained = metadata.get("last_trained")
                    self.training_accuracy = self.ml_accuracy  # Backward compat
            
            if models_loaded >= 1 and self.scaler is not None:
                self.is_loaded = True
                logger.info(f"✅ Loaded {models_loaded} XGBoost models for {self.sport_key}")
                logger.info(f"   ML: {self.ml_accuracy:.1%}, Spread: {self.spread_accuracy:.1%}, Totals: {self.totals_accuracy:.1%}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
        
        return False
    
    def save_models(self, metrics: Dict):
        """Save trained models to disk."""
        try:
            if self.moneyline_model:
                joblib.dump(self.moneyline_model, self.ml_model_path)
            if self.spread_model:
                joblib.dump(self.spread_model, self.spread_model_path)
            if self.totals_model:
                joblib.dump(self.totals_model, self.totals_model_path)
            if self.totals_regressor:
                joblib.dump(self.totals_regressor, self.totals_reg_path)
            if self.scaler:
                joblib.dump(self.scaler, self.scaler_path)
            
            # Save metadata
            metadata = {
                "sport_key": self.sport_key,
                "ml_accuracy": metrics.get("ml_accuracy", 0),
                "spread_accuracy": metrics.get("spread_accuracy", 0),
                "totals_accuracy": metrics.get("totals_accuracy", 0),
                "totals_mae": metrics.get("totals_mae", 0),
                "last_trained": datetime.now(timezone.utc).isoformat(),
                "features": FEATURE_NAMES,
                "model_type": "XGBClassifier_MultiMarket",
                # Backward compat
                "accuracy": metrics.get("ml_accuracy", 0)
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.ml_accuracy = metrics.get("ml_accuracy", 0)
            self.spread_accuracy = metrics.get("spread_accuracy", 0)
            self.totals_accuracy = metrics.get("totals_accuracy", 0)
            self.totals_mae = metrics.get("totals_mae", 0)
            self.training_accuracy = self.ml_accuracy
            self.last_trained = metadata["last_trained"]
            
            logger.info(f"✅ Saved XGBoost models for {self.sport_key}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def train(self, training_data: List[Dict]) -> Dict:
        """
        Train XGBoost models on historical game data.
        Now trains THREE models: Moneyline, Spread, Totals
        """
        if len(training_data) < 50:
            logger.warning(f"Insufficient training data: {len(training_data)} games (need 50+)")
            return {"error": "Insufficient training data", "games": len(training_data)}
        
        logger.info(f"🚀 Training XGBoost models on {len(training_data)} games...")
        
        # Extract features and labels for all three markets
        X = []
        y_ml = []  # Home win (1) or away win (0)
        y_spread = []  # Home covered spread (1) or not (0)
        y_totals = []  # Over (1) or Under (0)
        y_total_pts = []  # Actual total points (for regression)
        
        for game in training_data:
            features = game.get("features", {})
            
            # Moneyline outcome
            home_win = game.get("home_win")
            
            # Calculate spread cover
            home_score = game.get("home_score", 0)
            away_score = game.get("away_score", 0)
            spread = features.get("spread", 0)
            
            # Home covers if: home_score + spread > away_score
            # (spread is negative for favorite, positive for underdog)
            home_covered = (home_score + spread) > away_score if spread != 0 else None
            
            # Calculate over/under
            total_line = features.get("total_line", 0)
            actual_total = home_score + away_score
            went_over = actual_total > total_line if total_line > 0 else None
            
            # Only include if we have valid data
            if home_win is not None and features:
                feature_array = self.feature_engineering.features_to_array(features)
                X.append(feature_array)
                y_ml.append(1 if home_win else 0)
                
                # Spread - use moneyline as proxy if no spread data
                if home_covered is not None:
                    y_spread.append(1 if home_covered else 0)
                else:
                    y_spread.append(1 if home_win else 0)
                
                # Totals
                if went_over is not None:
                    y_totals.append(1 if went_over else 0)
                else:
                    y_totals.append(1 if actual_total > 210 else 0)  # Default
                
                y_total_pts.append(actual_total)
        
        X = np.array(X)
        y_ml = np.array(y_ml)
        y_spread = np.array(y_spread)
        y_totals = np.array(y_totals)
        y_total_pts = np.array(y_total_pts)
        
        logger.info(f"  Training samples: {len(X)}")
        logger.info(f"  ML: {sum(y_ml)} home wins, {len(y_ml) - sum(y_ml)} away wins")
        logger.info(f"  Spread: {sum(y_spread)} home covers, {len(y_spread) - sum(y_spread)} away covers")
        logger.info(f"  Totals: {sum(y_totals)} overs, {len(y_totals) - sum(y_totals)} unders")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_ml_train, y_ml_test = train_test_split(
            X_scaled, y_ml, test_size=0.2, random_state=42, stratify=y_ml
        )
        _, _, y_spread_train, y_spread_test = train_test_split(
            X_scaled, y_spread, test_size=0.2, random_state=42
        )
        _, _, y_totals_train, y_totals_test = train_test_split(
            X_scaled, y_totals, test_size=0.2, random_state=42
        )
        _, _, y_pts_train, y_pts_test = train_test_split(
            X_scaled, y_total_pts, test_size=0.2, random_state=42
        )
        
        metrics = {"success": True}
        
        # ===== TRAIN MONEYLINE MODEL =====
        logger.info("  📊 Training Moneyline model...")
        self.moneyline_model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', use_label_encoder=False
        )
        self.moneyline_model.fit(X_train, y_ml_train)
        self.model = self.moneyline_model  # Backward compat
        
        y_ml_pred = self.moneyline_model.predict(X_test)
        y_ml_prob = self.moneyline_model.predict_proba(X_test)[:, 1]
        
        ml_accuracy = accuracy_score(y_ml_test, y_ml_pred)
        ml_auc = roc_auc_score(y_ml_test, y_ml_prob)
        
        metrics["ml_accuracy"] = round(float(ml_accuracy), 4)
        metrics["ml_auc"] = round(float(ml_auc), 4)
        logger.info(f"     Accuracy: {ml_accuracy:.1%}, AUC: {ml_auc:.3f}")
        
        # ===== TRAIN SPREAD MODEL =====
        logger.info("  📊 Training Spread model...")
        self.spread_model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', use_label_encoder=False
        )
        self.spread_model.fit(X_train, y_spread_train)
        
        y_spread_pred = self.spread_model.predict(X_test)
        y_spread_prob = self.spread_model.predict_proba(X_test)[:, 1]
        
        spread_accuracy = accuracy_score(y_spread_test, y_spread_pred)
        spread_auc = roc_auc_score(y_spread_test, y_spread_prob)
        
        metrics["spread_accuracy"] = round(float(spread_accuracy), 4)
        metrics["spread_auc"] = round(float(spread_auc), 4)
        logger.info(f"     Accuracy: {spread_accuracy:.1%}, AUC: {spread_auc:.3f}")
        
        # ===== TRAIN TOTALS MODEL =====
        logger.info("  📊 Training Totals model...")
        self.totals_model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', use_label_encoder=False
        )
        self.totals_model.fit(X_train, y_totals_train)
        
        y_totals_pred = self.totals_model.predict(X_test)
        y_totals_prob = self.totals_model.predict_proba(X_test)[:, 1]
        
        totals_accuracy = accuracy_score(y_totals_test, y_totals_pred)
        totals_auc = roc_auc_score(y_totals_test, y_totals_prob)
        
        metrics["totals_accuracy"] = round(float(totals_accuracy), 4)
        metrics["totals_auc"] = round(float(totals_auc), 4)
        logger.info(f"     Accuracy: {totals_accuracy:.1%}, AUC: {totals_auc:.3f}")
        
        # ===== TRAIN TOTALS REGRESSOR (predicts actual total) =====
        logger.info("  📊 Training Totals regressor...")
        self.totals_regressor = XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        self.totals_regressor.fit(X_train, y_pts_train)
        
        y_pts_pred = self.totals_regressor.predict(X_test)
        totals_mae = mean_absolute_error(y_pts_test, y_pts_pred)
        
        metrics["totals_mae"] = round(float(totals_mae), 2)
        logger.info(f"     MAE: {totals_mae:.1f} points")
        
        # Feature importance (from moneyline model)
        feature_importance = dict(zip(FEATURE_NAMES, [float(x) for x in self.moneyline_model.feature_importances_]))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        top_features = [[name, round(float(importance), 4)] for name, importance in top_features]
        
        # Save all models
        self.save_models(metrics)
        self.is_loaded = True
        
        metrics.update({
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "top_features": top_features,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            # Backward compat
            "accuracy": metrics["ml_accuracy"],
            "auc_roc": metrics["ml_auc"]
        })
        
        logger.info(f"✅ All models trained successfully!")
        logger.info(f"   ML: {ml_accuracy:.1%}, Spread: {spread_accuracy:.1%}, Totals: {totals_accuracy:.1%}")
        
        return metrics
    
    def predict(
        self,
        home_team_data: Dict,
        away_team_data: Dict,
        odds_data: Dict,
        context_data: Dict = None,
        h2h_data: Dict = None
    ) -> Dict:
        """
        Make predictions for ALL THREE MARKETS using trained models.
        
        Returns:
            Predictions with probabilities for ML, Spread, and Totals
        """
        if not self.is_loaded:
            self.load_model()
        
        if not self.is_loaded or self.moneyline_model is None:
            logger.warning("XGBoost models not available, using fallback")
            return {
                "home_win_prob": 0.5,
                "spread_cover_prob": 0.5,
                "over_prob": 0.5,
                "confidence": 50.0,
                "model_available": False,
                "method": "fallback"
            }
        
        # Extract features
        features = self.feature_engineering.extract_features(
            home_team_data, away_team_data, odds_data, context_data, h2h_data
        )
        
        # Convert to array and scale
        X = self.feature_engineering.features_to_array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # ===== MONEYLINE PREDICTION =====
        ml_prob = self.moneyline_model.predict_proba(X_scaled)[0]
        home_win_prob = float(ml_prob[1])
        
        # ===== SPREAD PREDICTION =====
        if self.spread_model is not None:
            spread_prob = self.spread_model.predict_proba(X_scaled)[0]
            home_cover_prob = float(spread_prob[1])
        else:
            home_cover_prob = home_win_prob  # Fallback
        
        # ===== TOTALS PREDICTION =====
        if self.totals_model is not None:
            totals_prob = self.totals_model.predict_proba(X_scaled)[0]
            over_prob = float(totals_prob[1])
        else:
            over_prob = 0.5
        
        # ===== PREDICTED TOTAL POINTS =====
        if self.totals_regressor is not None:
            predicted_total = float(self.totals_regressor.predict(X_scaled)[0])
        else:
            predicted_total = features.get("combined_avg_pts", 220)
        
        # Calculate confidence for each market
        ml_confidence = abs(home_win_prob - 0.5) * 2 * 100
        ml_confidence = min(95, max(55, ml_confidence + 55))
        
        spread_confidence = abs(home_cover_prob - 0.5) * 2 * 100
        spread_confidence = min(95, max(55, spread_confidence + 55))
        
        totals_confidence = abs(over_prob - 0.5) * 2 * 100
        totals_confidence = min(95, max(55, totals_confidence + 55))
        
        # Determine best market
        best_market = "moneyline"
        best_confidence = ml_confidence
        best_prob = home_win_prob
        
        if spread_confidence > best_confidence and self.spread_model is not None:
            best_market = "spread"
            best_confidence = spread_confidence
            best_prob = home_cover_prob
        
        if totals_confidence > best_confidence and self.totals_model is not None:
            best_market = "totals"
            best_confidence = totals_confidence
            best_prob = over_prob
        
        return {
            # Moneyline
            "home_win_prob": round(float(home_win_prob), 4),
            "away_win_prob": round(float(1 - home_win_prob), 4),
            "ml_confidence": round(float(ml_confidence), 1),
            
            # Spread
            "home_cover_prob": round(float(home_cover_prob), 4),
            "away_cover_prob": round(float(1 - home_cover_prob), 4),
            "spread_confidence": round(float(spread_confidence), 1),
            
            # Totals
            "over_prob": round(float(over_prob), 4),
            "under_prob": round(float(1 - over_prob), 4),
            "totals_confidence": round(float(totals_confidence), 1),
            "predicted_total": round(float(predicted_total), 1),
            
            # Best market recommendation
            "best_market": best_market,
            "best_confidence": round(float(best_confidence), 1),
            "best_prob": round(float(best_prob), 4),
            
            # Model info
            "model_available": True,
            "ml_accuracy": float(self.ml_accuracy) if self.ml_accuracy else 0.0,
            "spread_accuracy": float(self.spread_accuracy) if self.spread_accuracy else 0.0,
            "totals_accuracy": float(self.totals_accuracy) if self.totals_accuracy else 0.0,
            "method": "xgboost_multi_market",
            "features_used": len(FEATURE_NAMES),
            
            # Backward compatibility
            "confidence": round(float(ml_confidence), 1),
            "model_accuracy": float(self.ml_accuracy) if self.ml_accuracy else 0.0
        }


class HistoricalDataCollector:
    """
    Collects historical game data from ESPN for model training.
    Enhanced to track spread and totals outcomes.
    """
    
    def __init__(self, db):
        self.db = db
        self.collection = db.historical_games
    
    async def fetch_season_data(self, sport_key: str = "basketball_nba", season: str = "2024") -> List[Dict]:
        """
        Fetch historical game data for a season from ESPN.
        Now includes spread and totals data.
        """
        import httpx
        
        logger.info(f"📊 Fetching historical data for {sport_key} season {season}...")
        
        # ESPN API endpoints
        sport_map = {
            "basketball_nba": ("basketball", "nba"),
            "americanfootball_nfl": ("football", "nfl"),
            "icehockey_nhl": ("hockey", "nhl"),
            "baseball_mlb": ("baseball", "mlb")
        }
        
        sport_type, league = sport_map.get(sport_key, ("basketball", "nba"))
        
        # Sport-specific defaults
        default_totals = {
            "basketball_nba": 220,
            "americanfootball_nfl": 45,
            "icehockey_nhl": 6,
            "baseball_mlb": 8
        }
        
        games = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Calculate date range for 1 season
            if sport_type == "basketball":
                start_date = datetime(int(season), 10, 1)
                end_date = datetime(int(season) + 1, 4, 30)
            elif sport_type == "football":
                start_date = datetime(int(season), 9, 1)
                end_date = datetime(int(season) + 1, 2, 15)
            elif sport_type == "hockey":
                start_date = datetime(int(season), 10, 1)
                end_date = datetime(int(season) + 1, 4, 30)
            else:
                start_date = datetime(int(season), 1, 1)
                end_date = datetime(int(season), 12, 31)
            
            today = datetime.now(timezone.utc).replace(tzinfo=None)
            if end_date > today:
                end_date = today - timedelta(days=1)
            
            current_date = start_date
            days_fetched = 0
            max_days = 200
            
            while current_date <= end_date and days_fetched < max_days:
                date_str = current_date.strftime("%Y%m%d")
                
                try:
                    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport_type}/{league}/scoreboard?dates={date_str}"
                    response = await client.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        events = data.get("events", [])
                        
                        for event in events:
                            game_record = await self._parse_espn_game(event, sport_key, default_totals.get(sport_key, 220))
                            if game_record and game_record.get("is_complete"):
                                games.append(game_record)
                    
                except Exception as e:
                    logger.debug(f"Error fetching {date_str}: {e}")
                
                current_date += timedelta(days=1)
                days_fetched += 1
                
                if days_fetched % 10 == 0:
                    await asyncio.sleep(0.5)
                    logger.info(f"  Fetched {days_fetched} days, {len(games)} games...")
        
        logger.info(f"✅ Collected {len(games)} historical games for {sport_key}")
        
        if games:
            await self._store_historical_games(games, sport_key, season)
        
        return games
    
    async def _parse_espn_game(self, event: Dict, sport_key: str, default_total: float) -> Optional[Dict]:
        """Parse ESPN event into game record with features for all markets."""
        try:
            status = event.get("status", {}).get("type", {}).get("name", "")
            if status != "STATUS_FINAL":
                return None
            
            competitions = event.get("competitions", [])
            if not competitions:
                return None
            
            competition = competitions[0]
            competitors = competition.get("competitors", [])
            
            if len(competitors) != 2:
                return None
            
            # Determine home/away
            home_team = None
            away_team = None
            for comp in competitors:
                if comp.get("homeAway") == "home":
                    home_team = comp
                else:
                    away_team = comp
            
            if not home_team or not away_team:
                return None
            
            home_score = int(home_team.get("score", 0))
            away_score = int(away_team.get("score", 0))
            home_win = home_score > away_score
            actual_total = home_score + away_score
            margin = home_score - away_score
            
            # Get team records
            home_record = home_team.get("records", [{}])[0] if home_team.get("records") else {}
            away_record = away_team.get("records", [{}])[0] if away_team.get("records") else {}
            
            home_wins, home_losses = self._parse_record(home_record.get("summary", "0-0"))
            away_wins, away_losses = self._parse_record(away_record.get("summary", "0-0"))
            
            home_total = home_wins + home_losses
            away_total = away_wins + away_losses
            home_win_pct = home_wins / home_total if home_total > 0 else 0.5
            away_win_pct = away_wins / away_total if away_total > 0 else 0.5
            
            # Estimate ELO from record
            home_elo = 1200 + (home_win_pct * 600)
            away_elo = 1200 + (away_win_pct * 600)
            
            # Estimate spread from ELO (negative for home favorite)
            elo_diff = home_elo - away_elo
            estimated_spread = -elo_diff / 25  # ~4 pts per 100 ELO
            
            # Estimate total line from historical scoring
            estimated_total = default_total + (home_win_pct - 0.5) * 10 + (away_win_pct - 0.5) * 10
            
            # Build features (PRE-GAME only)
            features = {
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": elo_diff,
                "home_win_pct": home_win_pct,
                "away_win_pct": away_win_pct,
                "win_pct_diff": home_win_pct - away_win_pct,
                "home_last10_wins": min(home_wins, 10),
                "away_last10_wins": min(away_wins, 10),
                "home_streak": 0,
                "away_streak": 0,
                "home_avg_margin": (home_win_pct - 0.5) * 20,
                "away_avg_margin": (away_win_pct - 0.5) * 20,
                "margin_diff": (home_win_pct - away_win_pct) * 20,
                "home_avg_pts": default_total / 2 + (home_win_pct - 0.5) * 10,
                "away_avg_pts": default_total / 2 + (away_win_pct - 0.5) * 10,
                "home_avg_pts_allowed": default_total / 2 - (home_win_pct - 0.5) * 10,
                "away_avg_pts_allowed": default_total / 2 - (away_win_pct - 0.5) * 10,
                "home_net_rating": (home_win_pct - 0.5) * 20,
                "away_net_rating": (away_win_pct - 0.5) * 20,
                "home_rest_days": 2,
                "away_rest_days": 2,
                "rest_advantage": 0,
                "is_back_to_back_home": 0,
                "is_back_to_back_away": 0,
                "home_ml_odds": 1.91,
                "away_ml_odds": 1.91,
                "implied_home_prob": 0.5,
                "spread": estimated_spread,
                "total_line": estimated_total,
                "h2h_home_wins": 0,
                "h2h_total_games": 0,
                "combined_avg_pts": default_total,
                "combined_pts_allowed": default_total,
                "pace_factor": default_total / 2,
                "defensive_rating_diff": 0,
            }
            
            # Calculate spread cover outcome
            home_covered = (home_score + estimated_spread) > away_score
            
            # Calculate over/under outcome
            went_over = actual_total > estimated_total
            
            return {
                "event_id": event.get("id"),
                "date": event.get("date"),
                "sport_key": sport_key,
                "home_team": home_team.get("team", {}).get("displayName", ""),
                "away_team": away_team.get("team", {}).get("displayName", ""),
                "home_score": home_score,
                "away_score": away_score,
                "home_win": home_win,
                "margin": margin,
                "total": actual_total,
                # Spread data
                "spread": estimated_spread,
                "home_covered": home_covered,
                # Totals data
                "total_line": estimated_total,
                "went_over": went_over,
                # Features
                "features": features,
                "is_complete": True
            }
            
        except Exception as e:
            logger.debug(f"Error parsing game: {e}")
            return None
    
    def _parse_record(self, record_str: str) -> Tuple[int, int]:
        """Parse 'W-L' record string."""
        try:
            parts = record_str.split("-")
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
        except:
            pass
        return 0, 0
    
    async def _store_historical_games(self, games: List[Dict], sport_key: str, season: str):
        """Store historical games in database."""
        try:
            await self.collection.delete_many({"sport_key": sport_key, "season": season})
            
            for game in games:
                game["season"] = season
            
            await self.collection.insert_many(games)
            logger.info(f"✅ Stored {len(games)} historical games in database")
        except Exception as e:
            logger.error(f"Error storing historical games: {e}")
    
    async def get_cached_historical_data(self, sport_key: str, season: str = None) -> List[Dict]:
        """Get historical data from database cache."""
        query = {"sport_key": sport_key}
        if season:
            query["season"] = season
        
        games = await self.collection.find(query, {"_id": 0}).to_list(5000)
        return games


class Backtester:
    """
    Backtesting infrastructure for validating model predictions.
    Now supports all three markets.
    """
    
    def __init__(self, predictor: XGBoostPredictor):
        self.predictor = predictor
    
    def backtest(self, test_data: List[Dict], threshold: float = 0.55, market: str = "all") -> Dict:
        """
        Run backtest on historical data for specified market(s).
        """
        if not self.predictor.is_loaded:
            self.predictor.load_model()
        
        if not self.predictor.is_loaded:
            return {"error": "Model not trained"}
        
        results = {
            "total_games": len(test_data),
            "moneyline": {"picks": 0, "wins": 0, "losses": 0, "profit": 0},
            "spread": {"picks": 0, "wins": 0, "losses": 0, "profit": 0},
            "totals": {"picks": 0, "wins": 0, "losses": 0, "profit": 0},
        }
        
        stake = 100
        
        for game in test_data:
            features = game.get("features", {})
            if not features:
                continue
            
            # Get predictions
            X = self.predictor.feature_engineering.features_to_array(features).reshape(1, -1)
            X_scaled = self.predictor.scaler.transform(X)
            
            # ===== MONEYLINE =====
            if market in ["all", "moneyline"] and self.predictor.moneyline_model:
                ml_prob = self.predictor.moneyline_model.predict_proba(X_scaled)[0][1]
                actual_home_win = game.get("home_win")
                
                if actual_home_win is not None:
                    if ml_prob >= threshold:
                        results["moneyline"]["picks"] += 1
                        if actual_home_win:
                            results["moneyline"]["wins"] += 1
                            results["moneyline"]["profit"] += stake * 0.91
                        else:
                            results["moneyline"]["losses"] += 1
                            results["moneyline"]["profit"] -= stake
                    elif ml_prob <= (1 - threshold):
                        results["moneyline"]["picks"] += 1
                        if not actual_home_win:
                            results["moneyline"]["wins"] += 1
                            results["moneyline"]["profit"] += stake * 0.91
                        else:
                            results["moneyline"]["losses"] += 1
                            results["moneyline"]["profit"] -= stake
            
            # ===== SPREAD =====
            if market in ["all", "spread"] and self.predictor.spread_model:
                spread_prob = self.predictor.spread_model.predict_proba(X_scaled)[0][1]
                actual_covered = game.get("home_covered")
                
                if actual_covered is not None:
                    if spread_prob >= threshold:
                        results["spread"]["picks"] += 1
                        if actual_covered:
                            results["spread"]["wins"] += 1
                            results["spread"]["profit"] += stake * 0.91
                        else:
                            results["spread"]["losses"] += 1
                            results["spread"]["profit"] -= stake
                    elif spread_prob <= (1 - threshold):
                        results["spread"]["picks"] += 1
                        if not actual_covered:
                            results["spread"]["wins"] += 1
                            results["spread"]["profit"] += stake * 0.91
                        else:
                            results["spread"]["losses"] += 1
                            results["spread"]["profit"] -= stake
            
            # ===== TOTALS =====
            if market in ["all", "totals"] and self.predictor.totals_model:
                totals_prob = self.predictor.totals_model.predict_proba(X_scaled)[0][1]
                actual_over = game.get("went_over")
                
                if actual_over is not None:
                    if totals_prob >= threshold:
                        results["totals"]["picks"] += 1
                        if actual_over:
                            results["totals"]["wins"] += 1
                            results["totals"]["profit"] += stake * 0.91
                        else:
                            results["totals"]["losses"] += 1
                            results["totals"]["profit"] -= stake
                    elif totals_prob <= (1 - threshold):
                        results["totals"]["picks"] += 1
                        if not actual_over:
                            results["totals"]["wins"] += 1
                            results["totals"]["profit"] += stake * 0.91
                        else:
                            results["totals"]["losses"] += 1
                            results["totals"]["profit"] -= stake
        
        # Calculate final metrics for each market
        for mkt in ["moneyline", "spread", "totals"]:
            m = results[mkt]
            if m["picks"] > 0:
                m["accuracy"] = round(m["wins"] / m["picks"], 4)
                m["roi"] = round(m["profit"] / (m["picks"] * stake) * 100, 2)
                m["profit"] = round(m["profit"], 2)
            else:
                m["accuracy"] = 0
                m["roi"] = 0
        
        logger.info(f"📊 Backtest Complete:")
        logger.info(f"   ML: {results['moneyline']['accuracy']:.1%} ({results['moneyline']['picks']} picks, {results['moneyline']['roi']:+.1f}% ROI)")
        logger.info(f"   Spread: {results['spread']['accuracy']:.1%} ({results['spread']['picks']} picks, {results['spread']['roi']:+.1f}% ROI)")
        logger.info(f"   Totals: {results['totals']['accuracy']:.1%} ({results['totals']['picks']} picks, {results['totals']['roi']:+.1f}% ROI)")
        
        return results


class EnhancedELOSystem:
    """
    Enhanced ELO system that properly tracks and updates from game results.
    """
    
    def __init__(self, db, sport_key: str):
        self.db = db
        self.sport_key = sport_key
        self.collection = db.elo_ratings
        self.history_collection = db.elo_history
        
        self.config = {
            "basketball_nba": {"k_factor": 20, "home_advantage": 100, "initial": 1500},
            "americanfootball_nfl": {"k_factor": 25, "home_advantage": 65, "initial": 1500},
            "icehockey_nhl": {"k_factor": 18, "home_advantage": 50, "initial": 1500},
        }.get(sport_key, {"k_factor": 20, "home_advantage": 100, "initial": 1500})
    
    async def get_team_elo(self, team_name: str) -> float:
        """Get current ELO for a team."""
        doc = await self.collection.find_one({
            "sport_key": self.sport_key,
            "team_name": team_name
        })
        
        if doc:
            return doc.get("elo", self.config["initial"])
        return self.config["initial"]
    
    async def get_all_elos(self) -> Dict[str, float]:
        """Get ELO ratings for all teams."""
        cursor = self.collection.find({"sport_key": self.sport_key})
        elos = {}
        async for doc in cursor:
            elos[doc["team_name"]] = doc.get("elo", self.config["initial"])
        return elos
    
    async def update_from_game_result(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        game_date: str
    ):
        """Update ELO ratings based on game result."""
        home_elo = await self.get_team_elo(home_team)
        away_elo = await self.get_team_elo(away_team)
        
        adjusted_home_elo = home_elo + self.config["home_advantage"]
        expected_home = 1 / (1 + 10 ** ((away_elo - adjusted_home_elo) / 400))
        
        home_win = home_score > away_score
        actual_home = 1 if home_win else 0
        actual_away = 1 - actual_home
        
        margin = abs(home_score - away_score)
        mov_mult = 1.0 + (margin / 20) * 0.5
        mov_mult = min(mov_mult, 1.5)
        
        k = self.config["k_factor"] * mov_mult
        new_home_elo = home_elo + k * (actual_home - expected_home)
        new_away_elo = away_elo + k * (actual_away - (1 - expected_home))
        
        await self.collection.update_one(
            {"sport_key": self.sport_key, "team_name": home_team},
            {"$set": {"elo": new_home_elo, "last_updated": datetime.now(timezone.utc).isoformat()}},
            upsert=True
        )
        
        await self.collection.update_one(
            {"sport_key": self.sport_key, "team_name": away_team},
            {"$set": {"elo": new_away_elo, "last_updated": datetime.now(timezone.utc).isoformat()}},
            upsert=True
        )
        
        await self.history_collection.insert_one({
            "sport_key": self.sport_key,
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "pre_home_elo": home_elo,
            "pre_away_elo": away_elo,
            "post_home_elo": new_home_elo,
            "post_away_elo": new_away_elo,
            "created_at": datetime.now(timezone.utc).isoformat()
        })
    
    async def rebuild_elos_from_history(self, games: List[Dict]):
        """Rebuild ELO ratings from historical game data."""
        logger.info(f"🔄 Rebuilding ELO ratings from {len(games)} games...")
        
        await self.collection.delete_many({"sport_key": self.sport_key})
        await self.history_collection.delete_many({"sport_key": self.sport_key})
        
        sorted_games = sorted(games, key=lambda x: x.get("date", ""))
        
        for game in sorted_games:
            await self.update_from_game_result(
                home_team=game.get("home_team", ""),
                away_team=game.get("away_team", ""),
                home_score=game.get("home_score", 0),
                away_score=game.get("away_score", 0),
                game_date=game.get("date", "")
            )
        
        logger.info(f"✅ ELO rebuild complete")


# Global instances
_predictors: Dict[str, XGBoostPredictor] = {}
_elo_systems: Dict[str, EnhancedELOSystem] = {}


def get_predictor(sport_key: str) -> XGBoostPredictor:
    """Get or create predictor for a sport."""
    if sport_key not in _predictors:
        _predictors[sport_key] = XGBoostPredictor(sport_key)
        _predictors[sport_key].load_model()
    return _predictors[sport_key]


async def get_elo_system(db, sport_key: str) -> EnhancedELOSystem:
    """Get or create ELO system for a sport."""
    if sport_key not in _elo_systems:
        _elo_systems[sport_key] = EnhancedELOSystem(db, sport_key)
    return _elo_systems[sport_key]
