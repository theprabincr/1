"""
XGBoost Machine Learning Model for Sports Betting Predictions
=============================================================
Implements real ML using XGBoost trained on historical game data.

Key Features:
1. Historical data collection from ESPN
2. Comprehensive feature engineering
3. XGBoost model training and prediction
4. Model persistence (save/load)
5. Weekly retraining support
6. Backtesting infrastructure

This replaces the fake "models" with real trained ML.
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
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
        
        Args:
            home_team_data: Home team stats and form
            away_team_data: Away team stats and form
            odds_data: Current betting odds
            context_data: Rest days, travel, etc.
            h2h_data: Head-to-head history
            
        Returns:
            Dict of feature_name -> value
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
        
        return features
    
    @staticmethod
    def features_to_array(features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array in consistent order."""
        return np.array([features.get(name, 0) for name in FEATURE_NAMES])


class XGBoostPredictor:
    """
    XGBoost-based predictor for sports betting.
    Handles training, prediction, and model persistence.
    """
    
    def __init__(self, sport_key: str = "basketball_nba"):
        self.sport_key = sport_key
        self.model: Optional[XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.model_path = MODEL_DIR / f"xgboost_{sport_key}.joblib"
        self.scaler_path = MODEL_DIR / f"scaler_{sport_key}.joblib"
        self.metadata_path = MODEL_DIR / f"metadata_{sport_key}.json"
        self.feature_engineering = FeatureEngineering()
        self.is_loaded = False
        self.training_accuracy = 0.0
        self.last_trained = None
        
    def load_model(self) -> bool:
        """Load trained model from disk if available."""
        try:
            if self.model_path.exists() and self.scaler_path.exists():
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                
                # Load metadata
                if self.metadata_path.exists():
                    with open(self.metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.training_accuracy = metadata.get("accuracy", 0)
                        self.last_trained = metadata.get("last_trained")
                
                self.is_loaded = True
                logger.info(f"✅ Loaded XGBoost model for {self.sport_key} (accuracy: {self.training_accuracy:.1%})")
                return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
        
        return False
    
    def save_model(self, accuracy: float):
        """Save trained model to disk."""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            # Save metadata
            metadata = {
                "sport_key": self.sport_key,
                "accuracy": accuracy,
                "last_trained": datetime.now(timezone.utc).isoformat(),
                "features": FEATURE_NAMES,
                "model_type": "XGBClassifier"
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.training_accuracy = accuracy
            self.last_trained = metadata["last_trained"]
            logger.info(f"✅ Saved XGBoost model for {self.sport_key}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def train(self, training_data: List[Dict]) -> Dict:
        """
        Train XGBoost model on historical game data.
        
        Args:
            training_data: List of game records with features and outcomes
            
        Returns:
            Training metrics
        """
        if len(training_data) < 50:
            logger.warning(f"Insufficient training data: {len(training_data)} games (need 50+)")
            return {"error": "Insufficient training data", "games": len(training_data)}
        
        logger.info(f"🚀 Training XGBoost model on {len(training_data)} games...")
        
        # Extract features and labels
        X = []
        y = []
        
        for game in training_data:
            features = game.get("features", {})
            outcome = game.get("home_win", None)
            
            if outcome is not None and features:
                feature_array = self.feature_engineering.features_to_array(features)
                X.append(feature_array)
                y.append(1 if outcome else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"  Training samples: {len(X)}, Home wins: {sum(y)}, Away wins: {len(y) - sum(y)}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train XGBoost
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='accuracy')
        
        # Feature importance
        feature_importance = dict(zip(FEATURE_NAMES, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Save model
        self.save_model(accuracy)
        self.is_loaded = True
        
        metrics = {
            "success": True,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc, 4),
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "top_features": top_features,
            "trained_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"✅ Training complete - Accuracy: {accuracy:.1%}, AUC: {auc:.3f}")
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
        Make prediction for a game using trained model.
        
        Returns:
            Prediction with probability and confidence
        """
        if not self.is_loaded:
            self.load_model()
        
        if not self.is_loaded or self.model is None:
            logger.warning("XGBoost model not available, using fallback")
            return {
                "home_win_prob": 0.5,
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
        
        # Predict
        prob = self.model.predict_proba(X_scaled)[0]
        home_win_prob = prob[1]  # Probability of home win
        
        # Calculate confidence (distance from 50%)
        confidence = abs(home_win_prob - 0.5) * 2 * 100  # 0-100 scale
        confidence = min(95, max(55, confidence + 55))  # Reasonable range
        
        return {
            "home_win_prob": round(home_win_prob, 4),
            "away_win_prob": round(1 - home_win_prob, 4),
            "confidence": round(confidence, 1),
            "model_available": True,
            "model_accuracy": self.training_accuracy,
            "method": "xgboost",
            "features_used": len(FEATURE_NAMES)
        }


class HistoricalDataCollector:
    """
    Collects historical game data from ESPN for model training.
    Supports fetching 1 season of data.
    """
    
    def __init__(self, db):
        self.db = db
        self.collection = db.historical_games
    
    async def fetch_season_data(self, sport_key: str = "basketball_nba", season: str = "2024") -> List[Dict]:
        """
        Fetch historical game data for a season from ESPN.
        
        Args:
            sport_key: Sport identifier
            season: Season year (e.g., "2024" for 2024-25 season)
            
        Returns:
            List of game records with features and outcomes
        """
        import httpx
        
        logger.info(f"📊 Fetching historical data for {sport_key} season {season}...")
        
        # ESPN API endpoints for historical scores
        sport_map = {
            "basketball_nba": ("basketball", "nba"),
            "americanfootball_nfl": ("football", "nfl"),
            "icehockey_nhl": ("hockey", "nhl"),
            "baseball_mlb": ("baseball", "mlb")
        }
        
        sport_type, league = sport_map.get(sport_key, ("basketball", "nba"))
        
        games = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch completed games from ESPN scoreboard history
            # ESPN provides scores for past dates
            
            # Calculate date range for 1 season
            if sport_type == "basketball":
                # NBA season: October to April
                start_date = datetime(int(season), 10, 1)
                end_date = datetime(int(season) + 1, 4, 30)
            elif sport_type == "football":
                # NFL season: September to February
                start_date = datetime(int(season), 9, 1)
                end_date = datetime(int(season) + 1, 2, 15)
            elif sport_type == "hockey":
                # NHL season: October to April
                start_date = datetime(int(season), 10, 1)
                end_date = datetime(int(season) + 1, 4, 30)
            else:
                start_date = datetime(int(season), 1, 1)
                end_date = datetime(int(season), 12, 31)
            
            # Limit to today if end_date is in the future
            today = datetime.now(timezone.utc).replace(tzinfo=None)
            if end_date > today:
                end_date = today - timedelta(days=1)
            
            # Fetch data day by day (ESPN API)
            current_date = start_date
            days_fetched = 0
            max_days = 200  # Limit to prevent too many requests
            
            while current_date <= end_date and days_fetched < max_days:
                date_str = current_date.strftime("%Y%m%d")
                
                try:
                    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport_type}/{league}/scoreboard?dates={date_str}"
                    response = await client.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        events = data.get("events", [])
                        
                        for event in events:
                            game_record = await self._parse_espn_game(event, sport_key)
                            if game_record and game_record.get("is_complete"):
                                games.append(game_record)
                    
                except Exception as e:
                    logger.debug(f"Error fetching {date_str}: {e}")
                
                current_date += timedelta(days=1)
                days_fetched += 1
                
                # Small delay to avoid rate limiting
                if days_fetched % 10 == 0:
                    await asyncio.sleep(0.5)
                    logger.info(f"  Fetched {days_fetched} days, {len(games)} games...")
        
        logger.info(f"✅ Collected {len(games)} historical games for {sport_key}")
        
        # Store in database for caching
        if games:
            await self._store_historical_games(games, sport_key, season)
        
        return games
    
    async def _parse_espn_game(self, event: Dict, sport_key: str) -> Optional[Dict]:
        """Parse ESPN event into game record with features."""
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
            
            # Extract team stats from the game
            home_stats = home_team.get("statistics", [])
            away_stats = away_team.get("statistics", [])
            
            # Get team records at time of game
            home_record = home_team.get("records", [{}])[0] if home_team.get("records") else {}
            away_record = away_team.get("records", [{}])[0] if away_team.get("records") else {}
            
            # Parse record string "W-L"
            home_wins, home_losses = self._parse_record(home_record.get("summary", "0-0"))
            away_wins, away_losses = self._parse_record(away_record.get("summary", "0-0"))
            
            # Calculate win percentages
            home_total = home_wins + home_losses
            away_total = away_wins + away_losses
            home_win_pct = home_wins / home_total if home_total > 0 else 0.5
            away_win_pct = away_wins / away_total if away_total > 0 else 0.5
            
            # Estimate ELO from record
            home_elo = 1200 + (home_win_pct * 600)
            away_elo = 1200 + (away_win_pct * 600)
            
            # Build features
            features = {
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": home_elo - away_elo,
                "home_win_pct": home_win_pct,
                "away_win_pct": away_win_pct,
                "win_pct_diff": home_win_pct - away_win_pct,
                "home_last10_wins": min(home_wins, 10),
                "away_last10_wins": min(away_wins, 10),
                "home_streak": 0,  # Not available from this data
                "away_streak": 0,
                "home_avg_margin": (home_score - away_score) if home_win else (away_score - home_score) * -1,
                "away_avg_margin": (away_score - home_score) if not home_win else (home_score - away_score) * -1,
                "margin_diff": home_score - away_score,
                "home_avg_pts": home_score,
                "away_avg_pts": away_score,
                "home_avg_pts_allowed": away_score,
                "away_avg_pts_allowed": home_score,
                "home_net_rating": home_score - away_score,
                "away_net_rating": away_score - home_score,
                "home_rest_days": 2,  # Default
                "away_rest_days": 2,
                "rest_advantage": 0,
                "is_back_to_back_home": 0,
                "is_back_to_back_away": 0,
                "home_ml_odds": 1.91,  # Default
                "away_ml_odds": 1.91,
                "implied_home_prob": 0.5,
                "spread": 0,
                "total_line": home_score + away_score,
                "h2h_home_wins": 0,
                "h2h_total_games": 0,
            }
            
            return {
                "event_id": event.get("id"),
                "date": event.get("date"),
                "sport_key": sport_key,
                "home_team": home_team.get("team", {}).get("displayName", ""),
                "away_team": away_team.get("team", {}).get("displayName", ""),
                "home_score": home_score,
                "away_score": away_score,
                "home_win": home_win,
                "margin": abs(home_score - away_score),
                "total": home_score + away_score,
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
        """Store historical games in database for caching."""
        try:
            # Clear old data for this sport/season
            await self.collection.delete_many({"sport_key": sport_key, "season": season})
            
            # Insert new data
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
    """
    
    def __init__(self, predictor: XGBoostPredictor):
        self.predictor = predictor
    
    def backtest(self, test_data: List[Dict], threshold: float = 0.55) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            test_data: Historical games with features and outcomes
            threshold: Probability threshold for making picks
            
        Returns:
            Backtest results with ROI and accuracy
        """
        if not self.predictor.is_loaded:
            self.predictor.load_model()
        
        if not self.predictor.is_loaded:
            return {"error": "Model not trained"}
        
        results = {
            "total_games": len(test_data),
            "picks_made": 0,
            "correct_picks": 0,
            "incorrect_picks": 0,
            "no_pick": 0,
            "profit_loss": 0.0,
            "roi": 0.0,
            "accuracy": 0.0,
            "details": []
        }
        
        stake = 100  # $100 per bet
        
        for game in test_data:
            features = game.get("features", {})
            actual_home_win = game.get("home_win")
            
            if actual_home_win is None or not features:
                continue
            
            # Get prediction
            X = self.predictor.feature_engineering.features_to_array(features).reshape(1, -1)
            X_scaled = self.predictor.scaler.transform(X)
            prob = self.predictor.model.predict_proba(X_scaled)[0]
            home_win_prob = prob[1]
            
            # Determine pick
            pick = None
            if home_win_prob >= threshold:
                pick = "home"
            elif home_win_prob <= (1 - threshold):
                pick = "away"
            
            if pick is None:
                results["no_pick"] += 1
                continue
            
            results["picks_made"] += 1
            
            # Check if correct
            is_correct = (pick == "home" and actual_home_win) or (pick == "away" and not actual_home_win)
            
            if is_correct:
                results["correct_picks"] += 1
                results["profit_loss"] += stake * 0.91  # Assuming -110 odds
            else:
                results["incorrect_picks"] += 1
                results["profit_loss"] -= stake
            
            results["details"].append({
                "game": f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}",
                "date": game.get("date"),
                "pick": pick,
                "prob": round(home_win_prob, 3),
                "actual_winner": "home" if actual_home_win else "away",
                "correct": is_correct
            })
        
        # Calculate final metrics
        if results["picks_made"] > 0:
            results["accuracy"] = round(results["correct_picks"] / results["picks_made"], 4)
            results["roi"] = round(results["profit_loss"] / (results["picks_made"] * stake) * 100, 2)
        
        results["total_wagered"] = results["picks_made"] * stake
        results["profit_loss"] = round(results["profit_loss"], 2)
        
        # Remove details from main results (too verbose)
        del results["details"]
        
        logger.info(f"📊 Backtest: {results['accuracy']:.1%} accuracy, {results['roi']:+.1f}% ROI on {results['picks_made']} picks")
        
        return results


class EnhancedELOSystem:
    """
    Enhanced ELO system that properly tracks and updates from game results.
    Stores ELO history in database for accurate historical reconstruction.
    """
    
    def __init__(self, db, sport_key: str):
        self.db = db
        self.sport_key = sport_key
        self.collection = db.elo_ratings
        self.history_collection = db.elo_history
        
        # Sport-specific configuration
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
        """
        Update ELO ratings based on game result.
        """
        home_elo = await self.get_team_elo(home_team)
        away_elo = await self.get_team_elo(away_team)
        
        # Calculate expected results
        adjusted_home_elo = home_elo + self.config["home_advantage"]
        expected_home = 1 / (1 + 10 ** ((away_elo - adjusted_home_elo) / 400))
        expected_away = 1 - expected_home
        
        # Actual results
        home_win = home_score > away_score
        actual_home = 1 if home_win else 0
        actual_away = 1 - actual_home
        
        # Margin of victory multiplier
        margin = abs(home_score - away_score)
        mov_mult = 1.0 + (margin / 20) * 0.5  # Up to 1.5x for big margins
        mov_mult = min(mov_mult, 1.5)
        
        # Update ELOs
        k = self.config["k_factor"] * mov_mult
        new_home_elo = home_elo + k * (actual_home - expected_home)
        new_away_elo = away_elo + k * (actual_away - expected_away)
        
        # Store new ratings
        await self.collection.update_one(
            {"sport_key": self.sport_key, "team_name": home_team},
            {
                "$set": {
                    "elo": new_home_elo,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                },
                "$setOnInsert": {"created_at": datetime.now(timezone.utc).isoformat()}
            },
            upsert=True
        )
        
        await self.collection.update_one(
            {"sport_key": self.sport_key, "team_name": away_team},
            {
                "$set": {
                    "elo": new_away_elo,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                },
                "$setOnInsert": {"created_at": datetime.now(timezone.utc).isoformat()}
            },
            upsert=True
        )
        
        # Store history
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
            "elo_change_home": new_home_elo - home_elo,
            "elo_change_away": new_away_elo - away_elo,
            "created_at": datetime.now(timezone.utc).isoformat()
        })
        
        logger.debug(f"ELO Update: {home_team} {home_elo:.0f}→{new_home_elo:.0f}, {away_team} {away_elo:.0f}→{new_away_elo:.0f}")
    
    async def rebuild_elos_from_history(self, games: List[Dict]):
        """
        Rebuild ELO ratings from historical game data.
        """
        logger.info(f"🔄 Rebuilding ELO ratings from {len(games)} games...")
        
        # Reset all ELOs
        await self.collection.delete_many({"sport_key": self.sport_key})
        await self.history_collection.delete_many({"sport_key": self.sport_key})
        
        # Sort games by date
        sorted_games = sorted(games, key=lambda x: x.get("date", ""))
        
        for game in sorted_games:
            await self.update_from_game_result(
                home_team=game.get("home_team", ""),
                away_team=game.get("away_team", ""),
                home_score=game.get("home_score", 0),
                away_score=game.get("away_score", 0),
                game_date=game.get("date", "")
            )
        
        logger.info("✅ ELO rebuild complete")


# Global instances (initialized on startup)
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
