"""
Simulation Engine
Monte Carlo simulations and Poisson modeling for score prediction
"""
import logging
import random
import math
from typing import Dict, List, Tuple
import statistics

logger = logging.getLogger(__name__)

# Sport-specific scoring configurations
SCORING_CONFIG = {
    "basketball_nba": {
        "avg_total": 225,
        "std_dev": 15,
        "min_score": 80,
        "max_score": 160,
        "scoring_type": "continuous"
    },
    "americanfootball_nfl": {
        "avg_total": 45,
        "std_dev": 12,
        "min_score": 0,
        "max_score": 60,
        "scoring_type": "discrete"
    },
    "icehockey_nhl": {
        "avg_total": 6,
        "std_dev": 2,
        "min_score": 0,
        "max_score": 10,
        "scoring_type": "discrete"
    },
    "soccer_epl": {
        "avg_total": 2.5,
        "std_dev": 1.5,
        "min_score": 0,
        "max_score": 8,
        "scoring_type": "discrete"
    }
}


class SimulationEngine:
    """
    Monte Carlo and Poisson simulation engine for game outcomes.
    """
    
    def __init__(self, sport_key: str):
        self.sport_key = sport_key
        self.config = SCORING_CONFIG.get(sport_key, SCORING_CONFIG["basketball_nba"])
    
    def run_monte_carlo(self, home_prob: float, spread: float, total: float, num_simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulation of game outcomes.
        
        Args:
            home_prob: Probability of home team winning (0-1)
            spread: Point spread
            total: Over/under total
            num_simulations: Number of simulations to run
        
        Returns:
            Dict with simulation results and statistics
        """
        home_wins = 0
        spread_covers = 0
        over_hits = 0
        
        home_scores = []
        away_scores = []
        margins = []
        totals = []
        
        for _ in range(num_simulations):
            # Simulate single game
            home_score, away_score = self._simulate_game(home_prob, total)
            
            home_scores.append(home_score)
            away_scores.append(away_score)
            
            margin = home_score - away_score
            margins.append(margin)
            
            game_total = home_score + away_score
            totals.append(game_total)
            
            # Check outcomes
            if home_score > away_score:
                home_wins += 1
            
            # Spread (home team covers if they win by more than spread)
            if margin > -spread:
                spread_covers += 1
            
            # Total
            if game_total > total:
                over_hits += 1
        
        # Calculate statistics
        home_win_pct = home_wins / num_simulations
        spread_cover_pct = spread_covers / num_simulations
        over_pct = over_hits / num_simulations
        
        avg_home_score = statistics.mean(home_scores)
        avg_away_score = statistics.mean(away_scores)
        avg_margin = statistics.mean(margins)
        avg_total = statistics.mean(totals)
        
        median_margin = statistics.median(margins)
        std_margin = statistics.stdev(margins) if len(margins) > 1 else 0
        
        return {
            "num_simulations": num_simulations,
            "outcomes": {
                "home_win_pct": round(home_win_pct, 3),
                "away_win_pct": round(1 - home_win_pct, 3),
                "spread_cover_pct": round(spread_cover_pct, 3),
                "over_pct": round(over_pct, 3),
                "under_pct": round(1 - over_pct, 3)
            },
            "score_projections": {
                "avg_home_score": round(avg_home_score, 1),
                "avg_away_score": round(avg_away_score, 1),
                "avg_margin": round(avg_margin, 1),
                "avg_total": round(avg_total, 1),
                "median_margin": round(median_margin, 1),
                "margin_std_dev": round(std_margin, 1)
            },
            "confidence_intervals": self._calculate_confidence_intervals(margins, totals)
        }
    
    def _simulate_game(self, home_prob: float, expected_total: float) -> Tuple[float, float]:
        """
        Simulate a single game outcome.
        """
        # Expected scores based on probability and total
        # Home team gets bonus based on win probability
        home_share = 0.5 + (home_prob - 0.5) * 0.5  # 50-65% range typically
        
        expected_home = expected_total * home_share
        expected_away = expected_total * (1 - home_share)
        
        # Add randomness based on sport
        std_dev = self.config["std_dev"]
        
        if self.config["scoring_type"] == "continuous":
            # Basketball-style (high scoring, more variance)
            home_score = random.gauss(expected_home, std_dev * 0.6)
            away_score = random.gauss(expected_away, std_dev * 0.6)
        else:
            # Football/Hockey/Soccer (lower scoring, Poisson-like)
            home_score = self._poisson_sample(expected_home)
            away_score = self._poisson_sample(expected_away)
        
        # Clamp to realistic ranges
        home_score = max(self.config["min_score"], min(self.config["max_score"], home_score))
        away_score = max(self.config["min_score"], min(self.config["max_score"], away_score))
        
        return home_score, away_score
    
    def _poisson_sample(self, lambda_param: float) -> float:
        """
        Sample from Poisson distribution (for low-scoring sports).
        """
        # Poisson sampling using inverse transform
        L = math.exp(-lambda_param)
        k = 0
        p = 1.0
        
        while p > L:
            k += 1
            p *= random.random()
        
        return float(k - 1)
    
    def _calculate_confidence_intervals(self, margins: List[float], totals: List[float]) -> Dict:
        """
        Calculate 95% confidence intervals.
        """
        sorted_margins = sorted(margins)
        sorted_totals = sorted(totals)
        
        n = len(sorted_margins)
        
        # 95% CI (2.5th to 97.5th percentile)
        margin_lower = sorted_margins[int(n * 0.025)]
        margin_upper = sorted_margins[int(n * 0.975)]
        
        total_lower = sorted_totals[int(n * 0.025)]
        total_upper = sorted_totals[int(n * 0.975)]
        
        return {
            "margin_95ci": [round(margin_lower, 1), round(margin_upper, 1)],
            "total_95ci": [round(total_lower, 1), round(total_upper, 1)]
        }
    
    def calculate_poisson_probabilities(self, home_expected: float, away_expected: float) -> Dict:
        """
        Calculate probabilities using Poisson distribution.
        Best for low-scoring sports (NFL, NHL, Soccer).
        """
        # Calculate probability mass function for range of scores
        max_score = 15 if self.sport_key == "americanfootball_nfl" else 10
        
        home_probs = [self._poisson_pmf(k, home_expected) for k in range(max_score + 1)]
        away_probs = [self._poisson_pmf(k, away_expected) for k in range(max_score + 1)]
        
        # Calculate outcome probabilities
        home_win_prob = 0.0
        away_win_prob = 0.0
        tie_prob = 0.0
        
        for h_score in range(max_score + 1):
            for a_score in range(max_score + 1):
                prob = home_probs[h_score] * away_probs[a_score]
                
                if h_score > a_score:
                    home_win_prob += prob
                elif a_score > h_score:
                    away_win_prob += prob
                else:
                    tie_prob += prob
        
        return {
            "home_win_prob": round(home_win_prob, 3),
            "away_win_prob": round(away_win_prob, 3),
            "tie_prob": round(tie_prob, 3),
            "most_likely_scores": self._find_most_likely_scores(home_probs, away_probs, max_score)
        }
    
    def _poisson_pmf(self, k: int, lambda_param: float) -> float:
        """
        Poisson probability mass function.
        P(X = k) = (λ^k * e^(-λ)) / k!
        """
        if lambda_param <= 0:
            return 0.0
        
        return (math.pow(lambda_param, k) * math.exp(-lambda_param)) / math.factorial(k)
    
    def _find_most_likely_scores(self, home_probs: List[float], away_probs: List[float], max_score: int) -> List[Dict]:
        """
        Find most likely final scores.
        """
        score_probs = []
        
        for h in range(max_score + 1):
            for a in range(max_score + 1):
                prob = home_probs[h] * away_probs[a]
                score_probs.append({
                    "home_score": h,
                    "away_score": a,
                    "probability": prob
                })
        
        # Sort by probability and return top 5
        score_probs.sort(key=lambda x: x["probability"], reverse=True)
        
        return [
            {
                "score": f"{s['home_score']}-{s['away_score']}",
                "probability": round(s["probability"], 3)
            }
            for s in score_probs[:5]
        ]


def run_game_simulation(sport_key: str, home_prob: float, spread: float, total: float) -> Dict:
    """
    Main entry point for game simulation.
    """
    engine = SimulationEngine(sport_key)
    
    # Run Monte Carlo
    mc_results = engine.run_monte_carlo(home_prob, spread, total, num_simulations=1000)
    
    # For low-scoring sports, also run Poisson
    poisson_results = None
    if sport_key in ["americanfootball_nfl", "icehockey_nhl", "soccer_epl"]:
        home_expected = total * home_prob
        away_expected = total * (1 - home_prob)
        poisson_results = engine.calculate_poisson_probabilities(home_expected, away_expected)
    
    return {
        "sport_key": sport_key,
        "monte_carlo": mc_results,
        "poisson": poisson_results
    }
