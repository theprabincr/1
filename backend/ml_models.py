"""
Machine Learning Models
Logistic regression, ensemble methods, and historical tracking
"""
import logging
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import statistics

logger = logging.getLogger(__name__)

# In-memory storage for model performance (would use DB in production)
MODEL_PREDICTIONS = []
MODEL_PERFORMANCE = {
    "elo_model": {"correct": 0, "total": 0, "roi": 0.0, "weight": 0.20},
    "context_model": {"correct": 0, "total": 0, "roi": 0.0, "weight": 0.15},
    "line_movement_model": {"correct": 0, "total": 0, "roi": 0.0, "weight": 0.25},
    "statistical_model": {"correct": 0, "total": 0, "roi": 0.0, "weight": 0.20},
    "psychology_model": {"correct": 0, "total": 0, "roi": 0.0, "weight": 0.20}
}


class LogisticRegressionModel:
    """
    Simple logistic regression for win probability.
    Uses weighted features to calculate probability.
    """
    
    def __init__(self, sport_key: str):
        self.sport_key = sport_key
        
        # Feature weights (learned from historical data - simplified)
        self.weights = {
            "basketball_nba": {
                "elo_diff": 0.006,  # Per ELO point
                "form_diff": 0.30,  # Win % difference
                "margin_diff": 0.04,  # Avg margin difference
                "rest_advantage": 0.15,  # Rest days
                "home_advantage": 0.055,  # Home court
                "injury_impact": -0.50,  # Injury disadvantage
                "line_movement": 0.20,  # Line movement signal
                "four_factors": 0.015,  # Per four factors point
                "intercept": 0.50  # Base probability
            },
            "americanfootball_nfl": {
                "elo_diff": 0.005,
                "form_diff": 0.25,
                "margin_diff": 0.06,
                "rest_advantage": 0.12,
                "home_advantage": 0.050,
                "injury_impact": -0.60,  # QB injury huge
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
        
        self.sport_weights = self.weights.get(sport_key, self.weights["basketball_nba"])
    
    def predict_probability(self, features: Dict) -> float:
        """
        Predict win probability using logistic regression.
        
        P(win) = 1 / (1 + e^(-z))
        where z = w1*x1 + w2*x2 + ... + intercept
        """
        z = self.sport_weights["intercept"]
        
        # Add weighted features
        if "elo_diff" in features:
            z += features["elo_diff"] * self.sport_weights.get("elo_diff", 0.005)
        
        if "form_diff" in features:
            z += features["form_diff"] * self.sport_weights.get("form_diff", 0.25)
        
        if "margin_diff" in features:
            z += features["margin_diff"] * self.sport_weights.get("margin_diff", 0.05)
        
        if "rest_advantage" in features:
            z += features["rest_advantage"] * self.sport_weights.get("rest_advantage", 0.15)
        
        if "home_advantage" in features:
            z += features["home_advantage"] * self.sport_weights.get("home_advantage", 0.055)
        
        if "injury_impact" in features:
            z += features["injury_impact"] * self.sport_weights.get("injury_impact", -0.50)
        
        if "line_movement" in features:
            z += features["line_movement"] * self.sport_weights.get("line_movement", 0.20)
        
        # Sport-specific features
        if "four_factors_diff" in features and "four_factors" in self.sport_weights:
            z += features["four_factors_diff"] * self.sport_weights["four_factors"]
        
        if "efficiency_diff" in features and "efficiency_diff" in self.sport_weights:
            z += features["efficiency_diff"] * self.sport_weights["efficiency_diff"]
        
        if "possession_diff" in features and "possession_diff" in self.sport_weights:
            z += features["possession_diff"] * self.sport_weights["possession_diff"]
        
        # Logistic function
        probability = 1 / (1 + math.exp(-z))
        
        # Clamp to reasonable range
        return max(0.15, min(0.85, probability))


class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple models.
    Uses weighted voting based on historical performance.
    """
    
    def __init__(self, sport_key: str):
        self.sport_key = sport_key
    
    def combine_predictions(self, model_predictions: Dict) -> Dict:
        """
        Combine predictions from multiple models using weighted voting.
        
        Args:
            model_predictions: Dict with model_name -> {probability, confidence, pick}
        
        Returns:
            Combined prediction with ensemble probability and confidence
        """
        if not model_predictions:
            return {
                "ensemble_probability": 0.50,
                "ensemble_confidence": 0.0,
                "consensus_pick": None,
                "model_agreement": 0.0
            }
        
        # Get current model weights (updated based on performance)
        weights = self._get_current_weights()
        
        # Calculate weighted average probability
        weighted_prob = 0.0
        total_weight = 0.0
        
        model_probs = []
        model_picks = []
        
        for model_name, prediction in model_predictions.items():
            weight = weights.get(model_name, 0.20)
            prob = prediction.get("probability", 0.50)
            pick = prediction.get("pick")
            
            weighted_prob += prob * weight
            total_weight += weight
            
            model_probs.append(prob)
            if pick:
                model_picks.append(pick)
        
        ensemble_prob = weighted_prob / total_weight if total_weight > 0 else 0.50
        
        # Calculate model agreement (standard deviation of probabilities)
        prob_std = statistics.stdev(model_probs) if len(model_probs) > 1 else 0.0
        agreement = 1.0 - min(1.0, prob_std / 0.25)  # Lower std = higher agreement (adjusted from 0.15 to 0.25)
        
        # Calculate consensus
        pick_counts = {}
        for pick in model_picks:
            pick_counts[pick] = pick_counts.get(pick, 0) + 1
        
        consensus_pick = max(pick_counts.items(), key=lambda x: x[1])[0] if pick_counts else None
        consensus_strength = max(pick_counts.values()) / len(model_picks) if model_picks else 0
        
        # Ensemble confidence (based on agreement and consensus)
        ensemble_confidence = (agreement * 0.6 + consensus_strength * 0.4) * 100
        
        return {
            "ensemble_probability": round(ensemble_prob, 3),
            "ensemble_confidence": round(ensemble_confidence, 1),
            "consensus_pick": consensus_pick,
            "model_agreement": round(agreement, 3),
            "consensus_strength": round(consensus_strength, 3),
            "num_models": len(model_predictions),
            "individual_predictions": model_predictions
        }
    
    def _get_current_weights(self) -> Dict:
        """
        Get current model weights based on performance.
        Better performing models get higher weights.
        """
        weights = {}
        
        for model_name, perf in MODEL_PERFORMANCE.items():
            if perf["total"] < 10:
                # Not enough data, use default weight
                weights[model_name] = perf["weight"]
            else:
                # Adjust weight based on accuracy
                accuracy = perf["correct"] / perf["total"]
                
                # Boost weight if accuracy > 55%, reduce if < 50%
                if accuracy > 0.55:
                    weights[model_name] = perf["weight"] * (1 + (accuracy - 0.55) * 2)
                elif accuracy < 0.50:
                    weights[model_name] = perf["weight"] * (0.5 + accuracy)
                else:
                    weights[model_name] = perf["weight"]
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def calculate_kelly_criterion(self, probability: float, odds: float) -> float:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Kelly % = (p * (odds - 1) - (1 - p)) / (odds - 1)
        
        Args:
            probability: Estimated win probability (0-1)
            odds: Decimal odds
        
        Returns:
            Recommended bet size as % of bankroll
        """
        if odds <= 1.0:
            return 0.0
        
        # Calculate edge
        implied_prob = 1 / odds
        edge = probability - implied_prob
        
        if edge <= 0:
            return 0.0
        
        # Kelly formula
        kelly_pct = (probability * (odds - 1) - (1 - probability)) / (odds - 1)
        
        # Use fractional Kelly (1/4 Kelly for safety)
        fractional_kelly = kelly_pct * 0.25
        
        # Cap at 5% of bankroll
        return max(0.0, min(0.05, fractional_kelly))


class ModelPerformanceTracker:
    """
    Tracks model performance over time and calculates accuracy metrics.
    """
    
    @staticmethod
    def record_prediction(model_name: str, prediction: Dict, actual_result: Dict):
        """
        Record a prediction and its outcome.
        """
        MODEL_PREDICTIONS.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "prediction": prediction,
            "actual_result": actual_result
        })
        
        # Update model performance
        if model_name in MODEL_PERFORMANCE:
            was_correct = prediction.get("pick") == actual_result.get("winner")
            
            MODEL_PERFORMANCE[model_name]["total"] += 1
            if was_correct:
                MODEL_PERFORMANCE[model_name]["correct"] += 1
    
    @staticmethod
    def get_model_stats() -> Dict:
        """
        Get current statistics for all models.
        """
        stats = {}
        
        for model_name, perf in MODEL_PERFORMANCE.items():
            if perf["total"] > 0:
                accuracy = perf["correct"] / perf["total"]
            else:
                accuracy = 0.0
            
            stats[model_name] = {
                "accuracy": round(accuracy, 3),
                "correct": perf["correct"],
                "total": perf["total"],
                "roi": round(perf["roi"], 2),
                "current_weight": round(perf["weight"], 3)
            }
        
        return stats
    
    @staticmethod
    def calculate_brier_score(predictions: List[Dict]) -> float:
        """
        Calculate Brier score (lower is better, 0 = perfect).
        Brier = (1/N) * Σ(predicted_prob - actual_outcome)^2
        """
        if not predictions:
            return 0.0
        
        squared_errors = []
        
        for pred in predictions:
            predicted_prob = pred.get("prediction", {}).get("probability", 0.5)
            actual_outcome = 1 if pred.get("actual_result", {}).get("correct", False) else 0
            
            squared_errors.append((predicted_prob - actual_outcome) ** 2)
        
        return sum(squared_errors) / len(squared_errors)


def create_feature_vector(analysis_data: Dict) -> Dict:
    """
    Create feature vector for ML model from analysis data.
    """
    features = {}
    
    # ELO difference
    home_elo = analysis_data.get("home_metrics", {}).get("elo_rating", 1500)
    away_elo = analysis_data.get("away_metrics", {}).get("elo_rating", 1500)
    features["elo_diff"] = home_elo - away_elo
    
    # Form difference
    home_form = analysis_data.get("home_metrics", {}).get("basic_stats", {}).get("win_pct", 0.5)
    away_form = analysis_data.get("away_metrics", {}).get("basic_stats", {}).get("win_pct", 0.5)
    features["form_diff"] = home_form - away_form
    
    # Margin difference
    home_margin = analysis_data.get("home_metrics", {}).get("basic_stats", {}).get("avg_margin", 0)
    away_margin = analysis_data.get("away_metrics", {}).get("basic_stats", {}).get("avg_margin", 0)
    features["margin_diff"] = home_margin - away_margin
    
    # Context
    context = analysis_data.get("context", {})
    features["rest_advantage"] = context.get("net_context_advantage", 0)
    features["home_advantage"] = 1.0  # Home team always gets this
    
    # Injury
    injury = analysis_data.get("injury", {})
    features["injury_impact"] = injury.get("net_advantage", 0)
    
    # Line movement
    line_movement = analysis_data.get("line_movement", {})
    features["line_movement"] = line_movement.get("confidence_adjustment", 0)
    
    # Sport-specific
    sport_key = analysis_data.get("sport_key", "")
    
    if sport_key == "basketball_nba":
        home_ff = analysis_data.get("home_metrics", {}).get("four_factors", {}).get("four_factors_score", 50)
        away_ff = analysis_data.get("away_metrics", {}).get("four_factors", {}).get("four_factors_score", 50)
        features["four_factors_diff"] = home_ff - away_ff
    
    return features
