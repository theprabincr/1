"""
Adaptive Learning System for BetPredictor
==========================================
Implements feedback loops that allow the ML models to learn from past results.

Key Features:
1. Persistent model performance tracking in MongoDB
2. Individual model prediction storage with each pick
3. Dynamic ensemble weight adjustment based on rolling accuracy
4. Online learning for logistic regression weights
5. Confidence calibration tracking (Brier score)
6. Sport-specific performance tracking

This transforms the static algorithm into a self-improving system.
"""
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from motor.motor_asyncio import AsyncIOMotorDatabase
import statistics

logger = logging.getLogger(__name__)

# Default model weights (used when no historical data)
DEFAULT_MODEL_WEIGHTS = {
    "elo_model": 0.20,
    "context_model": 0.15,
    "line_movement_model": 0.25,
    "statistical_model": 0.20,
    "psychology_model": 0.20
}

# Sport-specific default logistic regression weights
DEFAULT_LR_WEIGHTS = {
    "basketball_nba": {
        "elo_diff": 0.006,
        "form_diff": 0.30,
        "margin_diff": 0.04,
        "rest_advantage": 0.15,
        "home_advantage": 0.055,
        "injury_impact": -0.50,
        "line_movement": 0.20,
        "four_factors": 0.015,
        "intercept": 0.50
    },
    "americanfootball_nfl": {
        "elo_diff": 0.005,
        "form_diff": 0.25,
        "margin_diff": 0.06,
        "rest_advantage": 0.12,
        "home_advantage": 0.050,
        "injury_impact": -0.60,
        "line_movement": 0.25,
        "efficiency_diff": 0.012,
        "intercept": 0.50
    },
    "icehockey_nhl": {
        "elo_diff": 0.005,
        "form_diff": 0.22,
        "margin_diff": 0.08,
        "rest_advantage": 0.14,
        "home_advantage": 0.048,
        "injury_impact": -0.45,
        "line_movement": 0.22,
        "possession_diff": 0.010,
        "intercept": 0.50
    }
}


class AdaptiveLearningSystem:
    """
    Main class for managing adaptive learning across all models.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db.model_performance
        self.predictions_collection = db.predictions
        self.lr_weights_collection = db.lr_weights
        self.calibration_collection = db.calibration_data
    
    async def initialize(self):
        """Initialize collections and indexes"""
        # Create indexes for efficient queries
        await self.collection.create_index("model_name")
        await self.collection.create_index("sport_key")
        await self.collection.create_index([("model_name", 1), ("sport_key", 1)])
        await self.calibration_collection.create_index("sport_key")
        await self.lr_weights_collection.create_index("sport_key")
        
        logger.info("✅ Adaptive Learning System initialized")
    
    # ==================== MODEL PERFORMANCE TRACKING ====================
    
    async def record_individual_model_predictions(
        self,
        prediction_id: str,
        event_id: str,
        sport_key: str,
        model_predictions: Dict[str, Dict],
        final_pick: str,
        home_team: str,
        away_team: str
    ):
        """
        Store individual model predictions with the main prediction.
        This allows us to track which models were correct after the result comes in.
        
        Args:
            prediction_id: UUID of the main prediction
            event_id: Event being predicted
            sport_key: Sport (basketball_nba, etc.)
            model_predictions: Dict of model_name -> {probability, confidence, pick}
            final_pick: The actual pick made
            home_team: Home team name
            away_team: Away team name
        """
        model_record = {
            "prediction_id": prediction_id,
            "event_id": event_id,
            "sport_key": sport_key,
            "home_team": home_team,
            "away_team": away_team,
            "final_pick": final_pick,
            "model_predictions": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "result": "pending",  # Will be updated when game ends
            "result_updated_at": None,
            "models_correct": {},  # Will be filled when result comes in
        }
        
        # Store each model's prediction
        for model_name, pred in model_predictions.items():
            model_record["model_predictions"][model_name] = {
                "pick": pred.get("pick"),
                "probability": pred.get("probability", 0.5),
                "confidence": pred.get("confidence", 50),
                "was_correct": None  # Will be updated when result comes in
            }
        
        # Store in predictions collection as embedded document
        await self.predictions_collection.update_one(
            {"id": prediction_id},
            {"$set": {"individual_model_predictions": model_record["model_predictions"]}},
            upsert=False
        )
        
        logger.info(f"📊 Stored individual model predictions for {prediction_id}")
    
    async def update_model_performance_from_result(
        self,
        prediction_id: str,
        actual_winner: str,
        sport_key: str
    ):
        """
        Called when a prediction result comes in.
        Updates performance stats for each individual model.
        
        Args:
            prediction_id: UUID of the prediction
            actual_winner: The team that won (or 'over'/'under' for totals)
            sport_key: Sport key
        """
        # Get the prediction with individual model data
        prediction = await self.predictions_collection.find_one({"id": prediction_id})
        
        if not prediction:
            logger.warning(f"Prediction {prediction_id} not found for model update")
            return
        
        model_predictions = prediction.get("individual_model_predictions", {})
        if not model_predictions:
            logger.warning(f"No individual model predictions for {prediction_id}")
            return
        
        # Determine which models were correct
        models_correct = {}
        for model_name, model_pred in model_predictions.items():
            pick = model_pred.get("pick")
            if pick:
                was_correct = (pick == actual_winner)
                models_correct[model_name] = was_correct
                
                # Update the individual model's stats
                await self._update_single_model_stats(
                    model_name=model_name,
                    sport_key=sport_key,
                    was_correct=was_correct,
                    confidence=model_pred.get("confidence", 50),
                    probability=model_pred.get("probability", 0.5)
                )
        
        # Update the prediction document with correctness info
        await self.predictions_collection.update_one(
            {"id": prediction_id},
            {
                "$set": {
                    "models_correct": models_correct,
                    "individual_model_predictions.$[].result_checked": True
                }
            }
        )
        
        # Update calibration data
        final_confidence = prediction.get("confidence", 0.5)
        final_result = prediction.get("result")
        if final_result in ["win", "loss"]:
            await self._update_calibration_data(
                sport_key=sport_key,
                confidence=final_confidence,
                was_correct=(final_result == "win")
            )
        
        logger.info(f"✅ Updated model performance from result: {models_correct}")
    
    async def _update_single_model_stats(
        self,
        model_name: str,
        sport_key: str,
        was_correct: bool,
        confidence: float,
        probability: float
    ):
        """
        Update a single model's performance statistics.
        Uses upsert to create if doesn't exist.
        """
        now = datetime.now(timezone.utc).isoformat()
        
        # Update overall stats
        await self.collection.update_one(
            {"model_name": model_name, "sport_key": sport_key},
            {
                "$inc": {
                    "total_predictions": 1,
                    "correct_predictions": 1 if was_correct else 0,
                },
                "$push": {
                    "recent_results": {
                        "$each": [{
                            "was_correct": was_correct,
                            "confidence": confidence,
                            "probability": probability,
                            "timestamp": now
                        }],
                        "$slice": -100  # Keep last 100 results
                    }
                },
                "$set": {
                    "last_updated": now
                },
                "$setOnInsert": {
                    "model_name": model_name,
                    "sport_key": sport_key,
                    "created_at": now
                }
            },
            upsert=True
        )
        
        # Recalculate accuracy and update weight recommendation
        await self._recalculate_model_accuracy(model_name, sport_key)
    
    async def _recalculate_model_accuracy(self, model_name: str, sport_key: str):
        """
        Recalculate model accuracy and suggested weight based on recent performance.
        Uses exponential decay to weight recent results more heavily.
        """
        doc = await self.collection.find_one(
            {"model_name": model_name, "sport_key": sport_key}
        )
        
        if not doc:
            return
        
        recent_results = doc.get("recent_results", [])
        
        if len(recent_results) < 5:
            # Not enough data, use default weight
            suggested_weight = DEFAULT_MODEL_WEIGHTS.get(model_name, 0.20)
            accuracy = 0.5
        else:
            # Calculate weighted accuracy with exponential decay
            # More recent results have higher weight
            total_weight = 0
            weighted_correct = 0
            
            for i, result in enumerate(recent_results):
                # Exponential decay: most recent = highest weight
                decay_weight = math.exp(i / len(recent_results) * 2)  # More aggressive decay
                total_weight += decay_weight
                if result.get("was_correct"):
                    weighted_correct += decay_weight
            
            accuracy = weighted_correct / total_weight if total_weight > 0 else 0.5
            
            # Calculate suggested weight based on accuracy
            # Accuracy > 55% = boost weight, < 50% = reduce weight
            base_weight = DEFAULT_MODEL_WEIGHTS.get(model_name, 0.20)
            
            if accuracy > 0.55:
                # Boost up to 50% extra weight for high performers
                boost = (accuracy - 0.55) * 2.5  # 60% accuracy = 12.5% boost
                suggested_weight = base_weight * (1 + min(boost, 0.5))
            elif accuracy < 0.50:
                # Reduce weight for underperformers
                reduction = (0.50 - accuracy) * 2  # 45% accuracy = 10% reduction
                suggested_weight = base_weight * max(0.5, 1 - reduction)
            else:
                suggested_weight = base_weight
        
        # Also calculate simple accuracy
        total = doc.get("total_predictions", 0)
        correct = doc.get("correct_predictions", 0)
        simple_accuracy = correct / total if total > 0 else 0.5
        
        # Update the document
        await self.collection.update_one(
            {"model_name": model_name, "sport_key": sport_key},
            {
                "$set": {
                    "weighted_accuracy": round(accuracy, 4),
                    "simple_accuracy": round(simple_accuracy, 4),
                    "suggested_weight": round(suggested_weight, 4),
                    "accuracy_updated_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
    
    # ==================== DYNAMIC WEIGHT RETRIEVAL ====================
    
    async def get_current_model_weights(self, sport_key: str) -> Dict[str, float]:
        """
        Get the current recommended weights for all models based on performance.
        Returns normalized weights that sum to 1.0
        
        Args:
            sport_key: Sport to get weights for
            
        Returns:
            Dict of model_name -> weight
        """
        weights = {}
        
        # Get performance data for all models
        cursor = self.collection.find({"sport_key": sport_key})
        
        async for doc in cursor:
            model_name = doc.get("model_name")
            if model_name:
                # Use suggested_weight if we have enough data, otherwise default
                total = doc.get("total_predictions", 0)
                if total >= 10:
                    weights[model_name] = doc.get("suggested_weight", DEFAULT_MODEL_WEIGHTS.get(model_name, 0.20))
                else:
                    weights[model_name] = DEFAULT_MODEL_WEIGHTS.get(model_name, 0.20)
        
        # Fill in any missing models with defaults
        for model_name in DEFAULT_MODEL_WEIGHTS:
            if model_name not in weights:
                weights[model_name] = DEFAULT_MODEL_WEIGHTS[model_name]
        
        # Normalize to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    async def get_model_performance_summary(self, sport_key: Optional[str] = None) -> Dict:
        """
        Get a summary of all model performance for display.
        """
        query = {"sport_key": sport_key} if sport_key else {}
        
        cursor = self.collection.find(query)
        
        summary = {
            "models": {},
            "best_performer": None,
            "worst_performer": None,
            "total_predictions_tracked": 0
        }
        
        best_accuracy = 0
        worst_accuracy = 1.0
        
        async for doc in cursor:
            model_name = doc.get("model_name")
            accuracy = doc.get("weighted_accuracy", 0.5)
            total = doc.get("total_predictions", 0)
            
            summary["models"][model_name] = {
                "total_predictions": total,
                "correct_predictions": doc.get("correct_predictions", 0),
                "simple_accuracy": doc.get("simple_accuracy", 0),
                "weighted_accuracy": accuracy,
                "suggested_weight": doc.get("suggested_weight", DEFAULT_MODEL_WEIGHTS.get(model_name, 0.20)),
                "default_weight": DEFAULT_MODEL_WEIGHTS.get(model_name, 0.20),
                "last_updated": doc.get("last_updated")
            }
            
            summary["total_predictions_tracked"] += total
            
            if total >= 10:  # Only consider models with enough data
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    summary["best_performer"] = model_name
                if accuracy < worst_accuracy:
                    worst_accuracy = accuracy
                    summary["worst_performer"] = model_name
        
        return summary
    
    # ==================== LOGISTIC REGRESSION ONLINE LEARNING ====================
    
    async def update_lr_weights_from_result(
        self,
        sport_key: str,
        features: Dict,
        was_correct: bool,
        learning_rate: float = 0.01
    ):
        """
        Update logistic regression weights using stochastic gradient descent.
        This implements online learning - weights are adjusted after each result.
        
        Args:
            sport_key: Sport key
            features: Feature dict used for prediction
            was_correct: Whether the prediction was correct
            learning_rate: Step size for gradient descent
        """
        # Get current weights
        weights_doc = await self.lr_weights_collection.find_one({"sport_key": sport_key})
        
        if weights_doc:
            weights = weights_doc.get("weights", {})
        else:
            weights = DEFAULT_LR_WEIGHTS.get(sport_key, DEFAULT_LR_WEIGHTS["basketball_nba"]).copy()
        
        # Calculate current prediction probability
        z = weights.get("intercept", 0.5)
        for feature_name, feature_value in features.items():
            if feature_name in weights:
                z += feature_value * weights[feature_name]
        
        predicted_prob = 1 / (1 + math.exp(-z))
        
        # Target: 1 if correct (home won when we predicted home), 0 otherwise
        target = 1.0 if was_correct else 0.0
        
        # Gradient descent update
        # For logistic regression: gradient = (predicted - target) * feature
        error = predicted_prob - target
        
        # Update each weight
        for feature_name, feature_value in features.items():
            if feature_name in weights:
                gradient = error * feature_value
                weights[feature_name] -= learning_rate * gradient
        
        # Update intercept
        weights["intercept"] -= learning_rate * error
        
        # Clip weights to reasonable ranges to prevent divergence
        for key in weights:
            if key != "intercept":
                weights[key] = max(-2.0, min(2.0, weights[key]))
        
        # Store updated weights
        await self.lr_weights_collection.update_one(
            {"sport_key": sport_key},
            {
                "$set": {
                    "weights": weights,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "updates_count": weights_doc.get("updates_count", 0) + 1 if weights_doc else 1
                },
                "$setOnInsert": {
                    "sport_key": sport_key,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            },
            upsert=True
        )
        
        logger.debug(f"Updated LR weights for {sport_key}: error={error:.4f}")
    
    async def get_lr_weights(self, sport_key: str) -> Dict:
        """
        Get the current learned LR weights for a sport.
        Falls back to defaults if no learned weights exist.
        """
        doc = await self.lr_weights_collection.find_one({"sport_key": sport_key})
        
        if doc and doc.get("updates_count", 0) >= 10:
            return doc.get("weights", DEFAULT_LR_WEIGHTS.get(sport_key, DEFAULT_LR_WEIGHTS["basketball_nba"]))
        
        return DEFAULT_LR_WEIGHTS.get(sport_key, DEFAULT_LR_WEIGHTS["basketball_nba"])
    
    # ==================== CALIBRATION TRACKING ====================
    
    async def _update_calibration_data(
        self,
        sport_key: str,
        confidence: float,
        was_correct: bool
    ):
        """
        Track calibration data for confidence scores.
        Groups predictions by confidence bucket to check if 80% confident picks win 80% of the time.
        """
        # Bucket confidence into 5% ranges
        bucket = int(confidence * 100 // 5) * 5  # e.g., 0.82 -> 80
        
        await self.calibration_collection.update_one(
            {"sport_key": sport_key, "confidence_bucket": bucket},
            {
                "$inc": {
                    "total": 1,
                    "correct": 1 if was_correct else 0
                },
                "$set": {
                    "last_updated": datetime.now(timezone.utc).isoformat()
                },
                "$setOnInsert": {
                    "sport_key": sport_key,
                    "confidence_bucket": bucket,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            },
            upsert=True
        )
    
    async def get_calibration_report(self, sport_key: Optional[str] = None) -> Dict:
        """
        Get calibration report showing predicted vs actual win rates by confidence bucket.
        """
        query = {"sport_key": sport_key} if sport_key else {}
        cursor = self.calibration_collection.find(query).sort("confidence_bucket", 1)
        
        buckets = []
        total_squared_error = 0
        total_samples = 0
        
        async for doc in cursor:
            bucket = doc.get("confidence_bucket", 50)
            total = doc.get("total", 0)
            correct = doc.get("correct", 0)
            
            if total > 0:
                actual_win_rate = correct / total
                expected_win_rate = bucket / 100
                
                buckets.append({
                    "confidence_bucket": f"{bucket}-{bucket+5}%",
                    "expected_win_rate": expected_win_rate,
                    "actual_win_rate": round(actual_win_rate, 3),
                    "total_predictions": total,
                    "calibration_error": round(actual_win_rate - expected_win_rate, 3)
                })
                
                # For Brier score calculation
                total_squared_error += (expected_win_rate - actual_win_rate) ** 2 * total
                total_samples += total
        
        brier_score = total_squared_error / total_samples if total_samples > 0 else 0
        
        return {
            "buckets": buckets,
            "brier_score": round(brier_score, 4),
            "interpretation": self._interpret_brier_score(brier_score),
            "total_samples": total_samples
        }
    
    def _interpret_brier_score(self, score: float) -> str:
        """Interpret Brier score (lower is better)"""
        if score < 0.1:
            return "Excellent calibration"
        elif score < 0.2:
            return "Good calibration"
        elif score < 0.3:
            return "Moderate calibration - consider recalibrating"
        else:
            return "Poor calibration - confidence scores unreliable"
    
    # ==================== ROLLING WINDOW ANALYSIS ====================
    
    async def get_rolling_performance(
        self,
        model_name: str,
        sport_key: str,
        days: int = 30
    ) -> Dict:
        """
        Get rolling performance for a model over the last N days.
        """
        doc = await self.collection.find_one(
            {"model_name": model_name, "sport_key": sport_key}
        )
        
        if not doc:
            return {"error": "No data found"}
        
        recent_results = doc.get("recent_results", [])
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Filter to results within window
        in_window = []
        for result in recent_results:
            try:
                ts = datetime.fromisoformat(result.get("timestamp", "").replace('Z', '+00:00'))
                if ts >= cutoff:
                    in_window.append(result)
            except (ValueError, TypeError):
                continue
        
        if not in_window:
            return {
                "model_name": model_name,
                "sport_key": sport_key,
                "window_days": days,
                "total_in_window": 0,
                "accuracy": 0,
                "message": "No results in window"
            }
        
        correct = sum(1 for r in in_window if r.get("was_correct"))
        
        return {
            "model_name": model_name,
            "sport_key": sport_key,
            "window_days": days,
            "total_in_window": len(in_window),
            "correct_in_window": correct,
            "accuracy": round(correct / len(in_window), 4) if in_window else 0,
            "avg_confidence": round(sum(r.get("confidence", 50) for r in in_window) / len(in_window), 1)
        }


# ==================== ADAPTIVE LOGISTIC REGRESSION ====================

class AdaptiveLogisticRegression:
    """
    Logistic regression model that uses learned weights from the database.
    """
    
    def __init__(self, learning_system: AdaptiveLearningSystem, sport_key: str):
        self.learning_system = learning_system
        self.sport_key = sport_key
        self._weights = None
    
    async def load_weights(self):
        """Load the current weights from database"""
        self._weights = await self.learning_system.get_lr_weights(self.sport_key)
    
    def predict_probability(self, features: Dict) -> float:
        """
        Predict win probability using current weights.
        """
        if self._weights is None:
            # Use defaults if not loaded
            self._weights = DEFAULT_LR_WEIGHTS.get(self.sport_key, DEFAULT_LR_WEIGHTS["basketball_nba"])
        
        z = self._weights.get("intercept", 0.5)
        
        # Add weighted features
        for feature_name, feature_value in features.items():
            if feature_name in self._weights:
                z += feature_value * self._weights[feature_name]
        
        # Logistic function
        probability = 1 / (1 + math.exp(-z))
        
        # Clamp to reasonable range
        return max(0.15, min(0.85, probability))


# ==================== ADAPTIVE ENSEMBLE ====================

class AdaptiveEnsemble:
    """
    Ensemble model that uses dynamically adjusted weights from the database.
    """
    
    def __init__(self, learning_system: AdaptiveLearningSystem, sport_key: str):
        self.learning_system = learning_system
        self.sport_key = sport_key
        self._weights = None
    
    async def load_weights(self):
        """Load current weights from database"""
        self._weights = await self.learning_system.get_current_model_weights(self.sport_key)
    
    def combine_predictions(self, model_predictions: Dict) -> Dict:
        """
        Combine predictions using learned weights.
        """
        if self._weights is None:
            self._weights = DEFAULT_MODEL_WEIGHTS.copy()
        
        if not model_predictions:
            return {
                "ensemble_probability": 0.50,
                "ensemble_confidence": 0.0,
                "consensus_pick": None,
                "model_agreement": 0.0,
                "weights_used": self._weights
            }
        
        # Calculate weighted average probability
        weighted_prob = 0.0
        total_weight = 0.0
        
        model_probs = []
        model_picks = []
        
        for model_name, prediction in model_predictions.items():
            weight = self._weights.get(model_name, 0.20)
            prob = prediction.get("probability", 0.50)
            pick = prediction.get("pick")
            
            weighted_prob += prob * weight
            total_weight += weight
            
            model_probs.append(prob)
            if pick:
                model_picks.append(pick)
        
        ensemble_prob = weighted_prob / total_weight if total_weight > 0 else 0.50
        
        # Calculate model agreement
        prob_std = statistics.stdev(model_probs) if len(model_probs) > 1 else 0.0
        agreement = 1.0 - min(1.0, prob_std / 0.25)
        
        # Calculate consensus
        pick_counts = {}
        for pick in model_picks:
            pick_counts[pick] = pick_counts.get(pick, 0) + 1
        
        consensus_pick = max(pick_counts.items(), key=lambda x: x[1])[0] if pick_counts else None
        consensus_strength = max(pick_counts.values()) / len(model_picks) if model_picks else 0
        
        # Ensemble confidence
        ensemble_confidence = (agreement * 0.6 + consensus_strength * 0.4) * 100
        
        return {
            "ensemble_probability": round(ensemble_prob, 3),
            "ensemble_confidence": round(ensemble_confidence, 1),
            "consensus_pick": consensus_pick,
            "model_agreement": round(agreement, 3),
            "consensus_strength": round(consensus_strength, 3),
            "num_models": len(model_predictions),
            "weights_used": self._weights,
            "individual_predictions": model_predictions
        }


# ==================== HELPER FUNCTIONS ====================

async def create_adaptive_learning_system(db: AsyncIOMotorDatabase) -> AdaptiveLearningSystem:
    """
    Factory function to create and initialize the adaptive learning system.
    """
    system = AdaptiveLearningSystem(db)
    await system.initialize()
    return system
