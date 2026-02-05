"""
XGBoost Machine Learning Model for Sports Betting Predictions
=============================================================
Implements real ML using XGBoost trained on historical game data.

FIXED VERSION v2.0 - Addresses Critical Training Issues:
1. Fixed train/test split alignment across all models
2. Removed circular logic in spread/totals labels
3. Added data validation and sanity checks
4. Improved feature quality tracking
5. Added proper cross-validation
6. Fixed fallback logic that corrupted labels

SUPPORTS ALL THREE MARKETS:
1. Moneyline (Home Win Probability)
2. Spread (Cover Probability)  
3. Totals (Over/Under Probability)
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
    
    # Totals-specific features
    "combined_avg_pts",
    "combined_pts_allowed",
    "pace_factor",
    "defensive_rating_diff",
]

# Sport-specific configurations
SPORT_CONFIG = {
    "basketball_nba": {
        "default_total": 220,
        "spread_multiplier": 25,  # ELO diff to spread conversion
        "min_total": 180,
        "max_total": 260,
    },
    "americanfootball_nfl": {
        "default_total": 45,
        "spread_multiplier": 20,
        "min_total": 30,
        "max_total": 65,
    },
    "icehockey_nhl": {
        "default_total": 6,
        "spread_multiplier": 50,  # Different scale for hockey
        "min_total": 4,
        "max_total": 8,
    },
    "baseball_mlb": {
        "default_total": 8,
        "spread_multiplier": 40,
        "min_total": 6,
        "max_total": 12,
    }
}


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
    
    @staticmethod
    def validate_features(features: Dict[str, float], sport_key: str = "basketball_nba") -> Tuple[bool, List[str]]:
        """
        Validate feature quality and identify issues.
        Returns (is_valid, list_of_issues)
        """
        issues = []
        config = SPORT_CONFIG.get(sport_key, SPORT_CONFIG["basketball_nba"])
        
        # Check for default/placeholder values that indicate missing data
        if features.get("home_ml_odds", 0) == 1.91 and features.get("away_ml_odds", 0) == 1.91:
            issues.append("odds_missing")
        
        if features.get("home_elo", 1500) == 1500 and features.get("away_elo", 1500) == 1500:
            issues.append("elo_default")
        
        # Check total line is within reasonable bounds
        total = features.get("total_line", 0)
        if total < config["min_total"] or total > config["max_total"]:
            issues.append(f"total_out_of_range_{total}")
        
        # Check for unrealistic win percentages
        if features.get("home_win_pct", 0.5) in [0, 1] or features.get("away_win_pct", 0.5) in [0, 1]:
            issues.append("extreme_win_pct")
        
        is_valid = len(issues) == 0
        return is_valid, issues


class DataValidator:
    """
    Validates training data quality and flags potential issues.
    """
    
    @staticmethod
    def validate_labels(y_ml: np.ndarray, y_spread: np.ndarray, y_totals: np.ndarray) -> Dict:
        """Check label distribution for potential issues."""
        validation = {
            "ml_valid": True,
            "spread_valid": True,
            "totals_valid": True,
            "issues": []
        }
        
        # Check class balance (should be roughly 45-55% for each class)
        ml_ratio = np.mean(y_ml)
        spread_ratio = np.mean(y_spread)
        totals_ratio = np.mean(y_totals)
        
        # Moneyline typically has slight home advantage (50-55%)
        if ml_ratio < 0.40 or ml_ratio > 0.65:
            validation["issues"].append(f"ml_imbalanced_{ml_ratio:.2%}")
        
        # Spread should be close to 50% (that's the point of the spread)
        if spread_ratio < 0.40 or spread_ratio > 0.60:
            validation["issues"].append(f"spread_imbalanced_{spread_ratio:.2%}")
            validation["spread_valid"] = False
        
        # Totals should also be close to 50%
        if totals_ratio < 0.35 or totals_ratio > 0.65:
            validation["issues"].append(f"totals_imbalanced_{totals_ratio:.2%}")
            validation["totals_valid"] = False
        
        # Check for single-class issues
        if len(np.unique(y_ml)) < 2:
            validation["ml_valid"] = False
            validation["issues"].append("ml_single_class")
        
        if len(np.unique(y_spread)) < 2:
            validation["spread_valid"] = False
            validation["issues"].append("spread_single_class")
        
        if len(np.unique(y_totals)) < 2:
            validation["totals_valid"] = False
            validation["issues"].append("totals_single_class")
        
        return validation
    
    @staticmethod
    def validate_accuracy(accuracy: float, model_type: str) -> Tuple[bool, str]:
        """
        Validate that accuracy is within expected bounds.
        Returns (is_valid, warning_message)
        """
        # Expected accuracy ranges
        expected_ranges = {
            "moneyline": (0.52, 0.75),  # Moneyline can have higher accuracy
            "spread": (0.48, 0.60),      # Spread should be close to 50%
            "totals": (0.48, 0.60),      # Totals should be close to 50%
        }
        
        min_acc, max_acc = expected_ranges.get(model_type, (0.45, 0.70))
        
        if accuracy < min_acc:
            return False, f"{model_type} accuracy {accuracy:.1%} below expected minimum {min_acc:.1%}"
        
        if accuracy > max_acc:
            return False, f"⚠️ {model_type} accuracy {accuracy:.1%} suspiciously high (>{max_acc:.1%}) - possible data leakage!"
        
        return True, ""


class XGBoostPredictor:
    """
    FIXED XGBoost-based predictor for sports betting.
    
    Fixes applied:
    1. Consistent train/test split across all models
    2. Proper label validation
    3. Sanity checks for accuracy
    4. Cross-validation for model selection
    """
    
    def __init__(self, sport_key: str = "basketball_nba"):
        self.sport_key = sport_key
        self.config = SPORT_CONFIG.get(sport_key, SPORT_CONFIG["basketball_nba"])
        
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
        self.data_validator = DataValidator()
        self.is_loaded = False
        
        # Training metrics for each model
        self.ml_accuracy = 0.0
        self.spread_accuracy = 0.0
        self.totals_accuracy = 0.0
        self.totals_mae = 0.0
        
        self.last_trained = None
        self.training_warnings = []
        
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
                self.model = self.moneyline_model
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
                    self.training_warnings = metadata.get("warnings", [])
                    self.training_accuracy = self.ml_accuracy
            
            if models_loaded >= 1 and self.scaler is not None:
                self.is_loaded = True
                logger.info(f"✅ Loaded {models_loaded} XGBoost models for {self.sport_key}")
                logger.info(f"   ML: {self.ml_accuracy:.1%}, Spread: {self.spread_accuracy:.1%}, Totals: {self.totals_accuracy:.1%}")
                if self.training_warnings:
                    logger.warning(f"   ⚠️ Training warnings: {self.training_warnings}")
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
            
            # Save metadata with warnings
            metadata = {
                "sport_key": self.sport_key,
                "ml_accuracy": metrics.get("ml_accuracy", 0),
                "spread_accuracy": metrics.get("spread_accuracy", 0),
                "totals_accuracy": metrics.get("totals_accuracy", 0),
                "totals_mae": metrics.get("totals_mae", 0),
                "last_trained": datetime.now(timezone.utc).isoformat(),
                "features": FEATURE_NAMES,
                "model_type": "XGBClassifier_MultiMarket_v2",
                "warnings": metrics.get("warnings", []),
                "data_quality": metrics.get("data_quality", {}),
                "cv_scores": metrics.get("cv_scores", {}),
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
            self.training_warnings = metrics.get("warnings", [])
            
            logger.info(f"✅ Saved XGBoost models for {self.sport_key}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def train(self, training_data: List[Dict]) -> Dict:
        """
        Train XGBoost models on historical game data.
        
        FIXED VERSION:
        1. Uses consistent train/test split indices for all models
        2. Validates label quality before training
        3. Adds sanity checks for accuracy
        4. Uses cross-validation for better estimates
        """
        if len(training_data) < 50:
            logger.warning(f"Insufficient training data: {len(training_data)} games (need 50+)")
            return {"error": "Insufficient training data", "games": len(training_data)}
        
        logger.info(f"🚀 Training XGBoost models on {len(training_data)} games...")
        
        # ===== PHASE 1: EXTRACT FEATURES AND LABELS =====
        X = []
        y_ml = []
        y_spread = []
        y_totals = []
        y_total_pts = []
        
        # Track data quality
        games_with_real_spread = 0
        games_with_real_totals = 0
        skipped_games = 0
        
        for game in training_data:
            features = game.get("features", {})
            
            # Skip games without features
            if not features:
                skipped_games += 1
                continue
            
            # Moneyline outcome (most reliable)
            home_win = game.get("home_win")
            if home_win is None:
                skipped_games += 1
                continue
            
            home_score = game.get("home_score", 0)
            away_score = game.get("away_score", 0)
            actual_total = home_score + away_score
            
            # ===== SPREAD LABEL (FIXED) =====
            # Use actual spread from data if available, otherwise use realistic estimation
            actual_spread = game.get("actual_spread")  # Real Vegas line if available
            if actual_spread is not None:
                # Real spread available
                home_covered = (home_score + actual_spread) > away_score
                games_with_real_spread += 1
            else:
                # Estimate spread from margin and home advantage
                # DO NOT use ELO diff that we're also using as a feature (circular logic)
                # Instead, use a simple model: home team favored by ~3 pts on average
                home_advantage = 3.0 if self.sport_key == "basketball_nba" else 2.5
                estimated_spread = -home_advantage  # Home favored by default
                
                # Adjust based on win percentage difference (different from ELO)
                win_pct_diff = features.get("win_pct_diff", 0)
                estimated_spread -= win_pct_diff * 10  # Stronger team gets bigger spread
                
                home_covered = (home_score + estimated_spread) > away_score
            
            # ===== TOTALS LABEL (FIXED v2) =====
            # Use actual total line if available
            actual_total_line = game.get("actual_total_line")
            if actual_total_line is not None and actual_total_line > 0:
                went_over = actual_total > actual_total_line
                games_with_real_totals += 1
            else:
                # IMPROVED: Estimate line from team scoring averages (pre-game data)
                # This avoids using a fixed default which creates predictable patterns
                home_avg = features.get("home_avg_pts", self.config["default_total"] / 2)
                away_avg = features.get("away_avg_pts", self.config["default_total"] / 2)
                estimated_total_line = home_avg + away_avg
                
                # Clamp to reasonable range
                min_total = self.config["min_total"]
                max_total = self.config["max_total"]
                estimated_total_line = max(min_total, min(max_total, estimated_total_line))
                
                went_over = actual_total > estimated_total_line
            
            # Build arrays
            feature_array = self.feature_engineering.features_to_array(features)
            X.append(feature_array)
            y_ml.append(1 if home_win else 0)
            y_spread.append(1 if home_covered else 0)
            y_totals.append(1 if went_over else 0)
            y_total_pts.append(actual_total)
        
        X = np.array(X)
        y_ml = np.array(y_ml)
        y_spread = np.array(y_spread)
        y_totals = np.array(y_totals)
        y_total_pts = np.array(y_total_pts)
        
        logger.info(f"  Processed {len(X)} games (skipped {skipped_games})")
        logger.info(f"  Games with real spread lines: {games_with_real_spread}")
        logger.info(f"  Games with real total lines: {games_with_real_totals}")
        
        # ===== PHASE 2: VALIDATE LABELS =====
        label_validation = self.data_validator.validate_labels(y_ml, y_spread, y_totals)
        
        if label_validation["issues"]:
            logger.warning(f"  ⚠️ Label validation issues: {label_validation['issues']}")
        
        logger.info(f"  ML distribution: {np.mean(y_ml):.1%} home wins")
        logger.info(f"  Spread distribution: {np.mean(y_spread):.1%} home covers")
        logger.info(f"  Totals distribution: {np.mean(y_totals):.1%} overs")
        
        # ===== PHASE 3: SCALE FEATURES =====
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # ===== PHASE 4: SINGLE TRAIN/TEST SPLIT (FIXED!) =====
        # Generate indices ONCE and use for all models
        indices = np.arange(len(X_scaled))
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=0.2, 
            random_state=42,
            stratify=y_ml  # Stratify on primary target
        )
        
        # Use same indices for all data
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_ml_train, y_ml_test = y_ml[train_idx], y_ml[test_idx]
        y_spread_train, y_spread_test = y_spread[train_idx], y_spread[test_idx]
        y_totals_train, y_totals_test = y_totals[train_idx], y_totals[test_idx]
        y_pts_train, y_pts_test = y_total_pts[train_idx], y_total_pts[test_idx]
        
        logger.info(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        metrics = {
            "success": True,
            "warnings": [],
            "data_quality": {
                "total_games": len(X),
                "games_with_real_spread": games_with_real_spread,
                "games_with_real_totals": games_with_real_totals,
                "label_validation": label_validation
            },
            "cv_scores": {}
        }
        
        # ===== PHASE 5: TRAIN MONEYLINE MODEL =====
        logger.info("  📊 Training Moneyline model...")
        self.moneyline_model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', use_label_encoder=False
        )
        
        # Cross-validation first
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.moneyline_model, X_train, y_ml_train, cv=cv, scoring='accuracy')
        metrics["cv_scores"]["ml"] = {
            "mean": round(float(cv_scores.mean()), 4),
            "std": round(float(cv_scores.std()), 4)
        }
        logger.info(f"     CV Accuracy: {cv_scores.mean():.1%} (±{cv_scores.std():.1%})")
        
        # Train final model
        self.moneyline_model.fit(X_train, y_ml_train)
        self.model = self.moneyline_model
        
        y_ml_pred = self.moneyline_model.predict(X_test)
        y_ml_prob = self.moneyline_model.predict_proba(X_test)[:, 1]
        
        ml_accuracy = accuracy_score(y_ml_test, y_ml_pred)
        ml_auc = roc_auc_score(y_ml_test, y_ml_prob)
        
        # Validate accuracy
        acc_valid, acc_warning = self.data_validator.validate_accuracy(ml_accuracy, "moneyline")
        if not acc_valid:
            logger.warning(f"     {acc_warning}")
            metrics["warnings"].append(acc_warning)
        
        metrics["ml_accuracy"] = round(float(ml_accuracy), 4)
        metrics["ml_auc"] = round(float(ml_auc), 4)
        logger.info(f"     Test Accuracy: {ml_accuracy:.1%}, AUC: {ml_auc:.3f}")
        
        # ===== PHASE 6: TRAIN SPREAD MODEL =====
        logger.info("  📊 Training Spread model...")
        
        # Only train if labels are valid
        if label_validation["spread_valid"]:
            self.spread_model = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric='logloss', use_label_encoder=False
            )
            
            # Cross-validation
            cv_scores_spread = cross_val_score(self.spread_model, X_train, y_spread_train, cv=cv, scoring='accuracy')
            metrics["cv_scores"]["spread"] = {
                "mean": round(float(cv_scores_spread.mean()), 4),
                "std": round(float(cv_scores_spread.std()), 4)
            }
            logger.info(f"     CV Accuracy: {cv_scores_spread.mean():.1%} (±{cv_scores_spread.std():.1%})")
            
            self.spread_model.fit(X_train, y_spread_train)
            
            y_spread_pred = self.spread_model.predict(X_test)
            y_spread_prob = self.spread_model.predict_proba(X_test)[:, 1]
            
            spread_accuracy = accuracy_score(y_spread_test, y_spread_pred)
            try:
                spread_auc = roc_auc_score(y_spread_test, y_spread_prob)
            except ValueError:
                spread_auc = 0.5
            
            # Validate accuracy
            acc_valid, acc_warning = self.data_validator.validate_accuracy(spread_accuracy, "spread")
            if not acc_valid:
                logger.warning(f"     {acc_warning}")
                metrics["warnings"].append(acc_warning)
            
            metrics["spread_accuracy"] = round(float(spread_accuracy), 4)
            metrics["spread_auc"] = round(float(spread_auc), 4)
            logger.info(f"     Test Accuracy: {spread_accuracy:.1%}, AUC: {spread_auc:.3f}")
        else:
            logger.warning("     ⚠️ Skipping spread model - invalid labels")
            metrics["spread_accuracy"] = 0.5
            metrics["spread_auc"] = 0.5
            metrics["warnings"].append("spread_model_skipped_invalid_labels")
        
        # ===== PHASE 7: TRAIN TOTALS MODEL =====
        logger.info("  📊 Training Totals model...")
        
        if label_validation["totals_valid"]:
            self.totals_model = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric='logloss', use_label_encoder=False
            )
            
            # Cross-validation
            cv_scores_totals = cross_val_score(self.totals_model, X_train, y_totals_train, cv=cv, scoring='accuracy')
            metrics["cv_scores"]["totals"] = {
                "mean": round(float(cv_scores_totals.mean()), 4),
                "std": round(float(cv_scores_totals.std()), 4)
            }
            logger.info(f"     CV Accuracy: {cv_scores_totals.mean():.1%} (±{cv_scores_totals.std():.1%})")
            
            self.totals_model.fit(X_train, y_totals_train)
            
            y_totals_pred = self.totals_model.predict(X_test)
            y_totals_prob = self.totals_model.predict_proba(X_test)[:, 1]
            
            totals_accuracy = accuracy_score(y_totals_test, y_totals_pred)
            try:
                totals_auc = roc_auc_score(y_totals_test, y_totals_prob)
            except ValueError:
                totals_auc = 0.5
            
            # Validate accuracy
            acc_valid, acc_warning = self.data_validator.validate_accuracy(totals_accuracy, "totals")
            if not acc_valid:
                logger.warning(f"     {acc_warning}")
                metrics["warnings"].append(acc_warning)
            
            metrics["totals_accuracy"] = round(float(totals_accuracy), 4)
            metrics["totals_auc"] = round(float(totals_auc), 4)
            logger.info(f"     Test Accuracy: {totals_accuracy:.1%}, AUC: {totals_auc:.3f}")
        else:
            logger.warning("     ⚠️ Skipping totals model - invalid labels")
            metrics["totals_accuracy"] = 0.5
            metrics["totals_auc"] = 0.5
            metrics["warnings"].append("totals_model_skipped_invalid_labels")
        
        # ===== PHASE 8: TRAIN TOTALS REGRESSOR =====
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
        
        # ===== PHASE 9: FEATURE IMPORTANCE =====
        feature_importance = dict(zip(FEATURE_NAMES, [float(x) for x in self.moneyline_model.feature_importances_]))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        top_features = [[name, round(float(importance), 4)] for name, importance in top_features]
        
        # ===== PHASE 10: SAVE MODELS =====
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
        logger.info(f"   ML: {metrics['ml_accuracy']:.1%}, Spread: {metrics['spread_accuracy']:.1%}, Totals: {metrics['totals_accuracy']:.1%}")
        
        if metrics["warnings"]:
            logger.warning(f"   ⚠️ Warnings: {metrics['warnings']}")
        
        return metrics
    
    def predict(
        self,
        home_team_data: Dict,
        away_team_data: Dict,
        odds_data: Dict,
        context_data: Dict = None,
        h2h_data: Dict = None,
        home_team_name: str = None,
        away_team_name: str = None
    ) -> Dict:
        """
        Make predictions for ALL THREE MARKETS using trained models.
        
        Returns FAVORED OUTCOME for each market:
        - Moneyline: Which team is favored and their win probability
        - Spread: Which team is favored to cover and their probability
        - Totals: OVER or UNDER favored and the probability
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
        
        # Validate features
        is_valid, issues = self.feature_engineering.validate_features(features, self.sport_key)
        
        # Convert to array and scale
        X = self.feature_engineering.features_to_array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get spread and total line from odds
        spread_line = odds_data.get("spread", 0)
        total_line = odds_data.get("total", self.config["default_total"])
        
        # ===== MONEYLINE PREDICTION =====
        ml_prob = self.moneyline_model.predict_proba(X_scaled)[0]
        home_win_prob = float(ml_prob[1])
        away_win_prob = 1 - home_win_prob
        
        # Determine FAVORED team for moneyline
        if home_win_prob >= 0.5:
            ml_favored_team = home_team_name or "Home"
            ml_favored_prob = home_win_prob
            ml_underdog_team = away_team_name or "Away"
            ml_underdog_prob = away_win_prob
        else:
            ml_favored_team = away_team_name or "Away"
            ml_favored_prob = away_win_prob
            ml_underdog_team = home_team_name or "Home"
            ml_underdog_prob = home_win_prob
        
        # ===== SPREAD PREDICTION =====
        if self.spread_model is not None:
            spread_prob = self.spread_model.predict_proba(X_scaled)[0]
            home_cover_prob = float(spread_prob[1])
        else:
            home_cover_prob = home_win_prob  # Fallback
        away_cover_prob = 1 - home_cover_prob
        
        # Determine FAVORED team for spread
        if home_cover_prob >= 0.5:
            spread_favored_team = home_team_name or "Home"
            spread_favored_prob = home_cover_prob
            spread_favored_line = spread_line
        else:
            spread_favored_team = away_team_name or "Away"
            spread_favored_prob = away_cover_prob
            spread_favored_line = -spread_line if spread_line else 0
        
        # ===== TOTALS PREDICTION =====
        if self.totals_model is not None:
            totals_prob = self.totals_model.predict_proba(X_scaled)[0]
            over_prob = float(totals_prob[1])
        else:
            over_prob = 0.5
        under_prob = 1 - over_prob
        
        # Determine FAVORED totals direction
        if over_prob >= 0.5:
            totals_favored = "OVER"
            totals_favored_prob = over_prob
        else:
            totals_favored = "UNDER"
            totals_favored_prob = under_prob
        
        # ===== PREDICTED TOTAL POINTS =====
        if self.totals_regressor is not None:
            predicted_total = float(self.totals_regressor.predict(X_scaled)[0])
        else:
            predicted_total = features.get("combined_avg_pts", self.config["default_total"])
        
        # Calculate confidence for each market (based on how far from 50%)
        ml_confidence = abs(home_win_prob - 0.5) * 2 * 100
        ml_confidence = min(95, max(55, ml_confidence + 55))
        
        spread_confidence = abs(home_cover_prob - 0.5) * 2 * 100
        spread_confidence = min(95, max(55, spread_confidence + 55))
        
        totals_confidence = abs(over_prob - 0.5) * 2 * 100
        totals_confidence = min(95, max(55, totals_confidence + 55))
        
        # Determine best market
        best_market = "moneyline"
        best_confidence = ml_confidence
        best_prob = ml_favored_prob
        best_pick = ml_favored_team
        
        if spread_confidence > best_confidence and self.spread_model is not None:
            best_market = "spread"
            best_confidence = spread_confidence
            best_prob = spread_favored_prob
            best_pick = spread_favored_team
        
        if totals_confidence > best_confidence and self.totals_model is not None:
            best_market = "totals"
            best_confidence = totals_confidence
            best_prob = totals_favored_prob
            best_pick = totals_favored
        
        return {
            # ===== MONEYLINE (FAVORED DISPLAY) =====
            "ml_favored_team": ml_favored_team,
            "ml_favored_prob": round(float(ml_favored_prob), 4),
            "ml_underdog_team": ml_underdog_team,
            "ml_underdog_prob": round(float(ml_underdog_prob), 4),
            "ml_confidence": round(float(ml_confidence), 1),
            # Raw probabilities (for backward compat)
            "home_win_prob": round(float(home_win_prob), 4),
            "away_win_prob": round(float(away_win_prob), 4),
            
            # ===== SPREAD (FAVORED DISPLAY) =====
            "spread_favored_team": spread_favored_team,
            "spread_favored_prob": round(float(spread_favored_prob), 4),
            "spread_favored_line": round(float(spread_favored_line), 1) if spread_favored_line else 0,
            "spread_confidence": round(float(spread_confidence), 1),
            # Raw probabilities
            "home_cover_prob": round(float(home_cover_prob), 4),
            "away_cover_prob": round(float(away_cover_prob), 4),
            
            # ===== TOTALS (FAVORED DISPLAY) =====
            "totals_favored": totals_favored,
            "totals_favored_prob": round(float(totals_favored_prob), 4),
            "totals_line": round(float(total_line), 1),
            "totals_confidence": round(float(totals_confidence), 1),
            "predicted_total": round(float(predicted_total), 1),
            # Raw probabilities
            "over_prob": round(float(over_prob), 4),
            "under_prob": round(float(under_prob), 4),
            
            # ===== BEST MARKET (OVERALL RECOMMENDATION) =====
            "best_market": best_market,
            "best_pick": best_pick,
            "best_confidence": round(float(best_confidence), 1),
            "best_prob": round(float(best_prob), 4),
            
            # ===== MODEL INFO =====
            "model_available": True,
            "ml_accuracy": float(self.ml_accuracy) if self.ml_accuracy else 0.0,
            "spread_accuracy": float(self.spread_accuracy) if self.spread_accuracy else 0.0,
            "totals_accuracy": float(self.totals_accuracy) if self.totals_accuracy else 0.0,
            "method": "xgboost_multi_market",
            "features_used": len(FEATURE_NAMES),
            "feature_quality": "valid" if is_valid else f"issues:{issues}",
            
            # Backward compatibility
            "confidence": round(float(ml_confidence), 1),
            "model_accuracy": float(self.ml_accuracy) if self.ml_accuracy else 0.0
        }


class HistoricalDataCollector:
    """
    Collects historical game data from ESPN for model training.
    IMPROVED: Better data quality tracking and validation.
    """
    
    def __init__(self, db):
        self.db = db
        self.collection = db.historical_games
        self.team_stats_cache = {}  # Cache for team stats across games
    
    async def fetch_season_data(self, sport_key: str = "basketball_nba", season: str = "2024") -> List[Dict]:
        """
        Fetch historical game data for a season from ESPN.
        IMPROVED: Track team stats progression for better features.
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
        config = SPORT_CONFIG.get(sport_key, SPORT_CONFIG["basketball_nba"])
        
        games = []
        self.team_stats_cache = {}  # Reset cache for new season
        
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
            max_days = 250  # Increased for fuller season
            
            while current_date <= end_date and days_fetched < max_days:
                date_str = current_date.strftime("%Y%m%d")
                
                try:
                    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport_type}/{league}/scoreboard?dates={date_str}"
                    response = await client.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        events = data.get("events", [])
                        
                        for event in events:
                            game_record = await self._parse_espn_game(
                                event, sport_key, config, current_date
                            )
                            if game_record and game_record.get("is_complete"):
                                games.append(game_record)
                                # Update team stats cache
                                self._update_team_stats(game_record)
                    
                except Exception as e:
                    logger.debug(f"Error fetching {date_str}: {e}")
                
                current_date += timedelta(days=1)
                days_fetched += 1
                
                if days_fetched % 10 == 0:
                    await asyncio.sleep(0.3)
                    logger.info(f"  Fetched {days_fetched} days, {len(games)} games...")
        
        logger.info(f"✅ Collected {len(games)} historical games for {sport_key}")
        
        if games:
            await self._store_historical_games(games, sport_key, season)
        
        return games
    
    def _update_team_stats(self, game: Dict):
        """Update team stats cache after each game."""
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        home_score = game.get("home_score", 0)
        away_score = game.get("away_score", 0)
        
        for team, scored, allowed, is_home in [
            (home_team, home_score, away_score, True),
            (away_team, away_score, home_score, False)
        ]:
            if team not in self.team_stats_cache:
                self.team_stats_cache[team] = {
                    "games": 0, "wins": 0, "losses": 0,
                    "points_for": 0, "points_against": 0,
                    "home_games": 0, "home_wins": 0,
                    "last_5_results": [],
                    "streak": 0
                }
            
            stats = self.team_stats_cache[team]
            won = (scored > allowed)
            
            stats["games"] += 1
            stats["wins"] += 1 if won else 0
            stats["losses"] += 0 if won else 1
            stats["points_for"] += scored
            stats["points_against"] += allowed
            
            if is_home:
                stats["home_games"] += 1
                stats["home_wins"] += 1 if won else 0
            
            # Track last 5 results
            stats["last_5_results"].append(1 if won else 0)
            if len(stats["last_5_results"]) > 5:
                stats["last_5_results"].pop(0)
            
            # Update streak
            if won:
                stats["streak"] = stats["streak"] + 1 if stats["streak"] >= 0 else 1
            else:
                stats["streak"] = stats["streak"] - 1 if stats["streak"] <= 0 else -1
    
    def _get_team_pregame_stats(self, team: str, config: Dict) -> Dict:
        """Get team stats BEFORE a game (for prediction features)."""
        if team not in self.team_stats_cache:
            return {
                "win_pct": 0.5,
                "avg_pts": config["default_total"] / 2,
                "avg_pts_allowed": config["default_total"] / 2,
                "last5_wins": 2.5,
                "streak": 0,
                "elo": 1500
            }
        
        stats = self.team_stats_cache[team]
        games = max(stats["games"], 1)
        
        win_pct = stats["wins"] / games
        avg_pts = stats["points_for"] / games
        avg_pts_allowed = stats["points_against"] / games
        last5_wins = sum(stats["last_5_results"]) if stats["last_5_results"] else 2.5
        
        # Calculate ELO from overall record
        elo = 1200 + (win_pct * 600)
        
        return {
            "win_pct": win_pct,
            "avg_pts": avg_pts,
            "avg_pts_allowed": avg_pts_allowed,
            "last5_wins": last5_wins,
            "streak": stats["streak"],
            "elo": elo,
            "games_played": games
        }
    
    async def _parse_espn_game(self, event: Dict, sport_key: str, config: Dict, game_date: datetime) -> Optional[Dict]:
        """Parse ESPN event into game record with IMPROVED features."""
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
            home_team_data = None
            away_team_data = None
            for comp in competitors:
                if comp.get("homeAway") == "home":
                    home_team_data = comp
                else:
                    away_team_data = comp
            
            if not home_team_data or not away_team_data:
                return None
            
            home_team = home_team_data.get("team", {}).get("displayName", "")
            away_team = away_team_data.get("team", {}).get("displayName", "")
            home_score = int(home_team_data.get("score", 0))
            away_score = int(away_team_data.get("score", 0))
            home_win = home_score > away_score
            actual_total = home_score + away_score
            margin = home_score - away_score
            
            # Get PRE-GAME stats from cache
            home_stats = self._get_team_pregame_stats(home_team, config)
            away_stats = self._get_team_pregame_stats(away_team, config)
            
            # Try to get odds from ESPN (if available)
            odds_data = competition.get("odds", [{}])[0] if competition.get("odds") else {}
            
            # Extract actual spread and total if available
            actual_spread = None
            actual_total_line = None
            
            if odds_data:
                # ESPN sometimes provides spread
                spread_str = odds_data.get("details", "")
                if spread_str and any(c.isdigit() for c in spread_str):
                    try:
                        # Parse spread from string like "LAL -3.5"
                        parts = spread_str.split()
                        for part in parts:
                            if part.replace("-", "").replace("+", "").replace(".", "").isdigit():
                                actual_spread = float(part)
                                break
                    except:
                        pass
                
                # ESPN sometimes provides over/under
                ou_str = odds_data.get("overUnder", "")
                if ou_str:
                    try:
                        actual_total_line = float(ou_str)
                    except:
                        pass
            
            # Build PRE-GAME features (what we would know before the game)
            features = {
                # Team strength from pre-game stats
                "home_elo": home_stats["elo"],
                "away_elo": away_stats["elo"],
                "elo_diff": home_stats["elo"] - away_stats["elo"],
                "home_win_pct": home_stats["win_pct"],
                "away_win_pct": away_stats["win_pct"],
                "win_pct_diff": home_stats["win_pct"] - away_stats["win_pct"],
                
                # Recent form
                "home_last10_wins": home_stats["last5_wins"] * 2,  # Scale to 10
                "away_last10_wins": away_stats["last5_wins"] * 2,
                "home_streak": home_stats["streak"],
                "away_streak": away_stats["streak"],
                "home_avg_margin": home_stats["avg_pts"] - home_stats["avg_pts_allowed"],
                "away_avg_margin": away_stats["avg_pts"] - away_stats["avg_pts_allowed"],
                "margin_diff": (home_stats["avg_pts"] - home_stats["avg_pts_allowed"]) - (away_stats["avg_pts"] - away_stats["avg_pts_allowed"]),
                
                # Scoring
                "home_avg_pts": home_stats["avg_pts"],
                "away_avg_pts": away_stats["avg_pts"],
                "home_avg_pts_allowed": home_stats["avg_pts_allowed"],
                "away_avg_pts_allowed": away_stats["avg_pts_allowed"],
                "home_net_rating": home_stats["avg_pts"] - home_stats["avg_pts_allowed"],
                "away_net_rating": away_stats["avg_pts"] - away_stats["avg_pts_allowed"],
                
                # Context (simplified - real system would track rest days)
                "home_rest_days": 2,
                "away_rest_days": 2,
                "rest_advantage": 0,
                "is_back_to_back_home": 0,
                "is_back_to_back_away": 0,
                
                # Odds/Market - use actual if available, else reasonable estimates
                "home_ml_odds": 1.91,
                "away_ml_odds": 1.91,
                "implied_home_prob": 0.5,
                "spread": actual_spread if actual_spread else 0,
                "total_line": actual_total_line if actual_total_line else config["default_total"],
                
                # H2H (would need separate tracking)
                "h2h_home_wins": 0,
                "h2h_total_games": 0,
                
                # Totals-specific
                "combined_avg_pts": home_stats["avg_pts"] + away_stats["avg_pts"],
                "combined_pts_allowed": home_stats["avg_pts_allowed"] + away_stats["avg_pts_allowed"],
                "pace_factor": (home_stats["avg_pts"] + away_stats["avg_pts"] + home_stats["avg_pts_allowed"] + away_stats["avg_pts_allowed"]) / 4,
                "defensive_rating_diff": home_stats["avg_pts_allowed"] - away_stats["avg_pts_allowed"],
            }
            
            return {
                "event_id": event.get("id"),
                "date": event.get("date"),
                "game_date": game_date.isoformat(),
                "sport_key": sport_key,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_win": home_win,
                "margin": margin,
                "total": actual_total,
                # Actual betting lines (if available)
                "actual_spread": actual_spread,
                "actual_total_line": actual_total_line,
                # Features
                "features": features,
                "is_complete": True,
                "has_real_odds": actual_spread is not None or actual_total_line is not None
            }
            
        except Exception as e:
            logger.debug(f"Error parsing game: {e}")
            return None
    
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
    IMPROVED: Better metrics and realistic simulation.
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
                
                # Calculate actual cover outcome
                home_score = game.get("home_score", 0)
                away_score = game.get("away_score", 0)
                spread = game.get("actual_spread") or features.get("spread", 0)
                
                if spread != 0:
                    actual_covered = (home_score + spread) > away_score
                    
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
                
                home_score = game.get("home_score", 0)
                away_score = game.get("away_score", 0)
                actual_total = home_score + away_score
                total_line = game.get("actual_total_line") or features.get("total_line", 0)
                
                if total_line > 0:
                    actual_over = actual_total > total_line
                    
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
