"""
Advanced Metrics Calculator
Computes ELO ratings, Four Factors (NBA), efficiency metrics, and sport-specific analytics
"""
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import statistics

logger = logging.getLogger(__name__)

# ELO System Configuration
ELO_CONFIG = {
    "basketball_nba": {
        "initial_elo": 1500,
        "k_factor": 20,
        "home_advantage": 100,
        "blowout_multiplier": 1.5,  # Extra weight for big wins
        "blowout_threshold": 15
    },
    "americanfootball_nfl": {
        "initial_elo": 1500,
        "k_factor": 25,
        "home_advantage": 65,
        "blowout_multiplier": 1.4,
        "blowout_threshold": 14
    },
    "icehockey_nhl": {
        "initial_elo": 1500,
        "k_factor": 18,
        "home_advantage": 50,
        "blowout_multiplier": 1.3,
        "blowout_threshold": 3
    },
    "soccer_epl": {
        "initial_elo": 1500,
        "k_factor": 30,
        "home_advantage": 80,
        "blowout_multiplier": 1.2,
        "blowout_threshold": 2
    }
}

# In-memory ELO storage (fallback only - prefer DB)
ELO_RATINGS = {}

# Cached DB ELO ratings (populated from MongoDB on startup)
DB_ELO_CACHE = {}


class ELORatingSystem:
    """
    ELO rating system for sports teams.
    Dynamically calculates team strength and predicts outcomes.
    Uses trained ELO from database when available.
    """
    
    def __init__(self, sport_key: str):
        self.sport_key = sport_key
        self.config = ELO_CONFIG.get(sport_key, ELO_CONFIG["basketball_nba"])
    
    def get_team_elo(self, team_name: str) -> float:
        """Get current ELO rating for a team. Prefers DB-trained ELO over in-memory."""
        # First try DB cache (trained ELO from XGBoost training)
        db_key = f"{self.sport_key}:{team_name}"
        if db_key in DB_ELO_CACHE:
            return DB_ELO_CACHE[db_key]
        
        # Fallback to in-memory
        key = f"{self.sport_key}:{team_name}"
        return ELO_RATINGS.get(key, self.config["initial_elo"])
    
    def set_team_elo(self, team_name: str, elo: float):
        """Set ELO rating for a team."""
        key = f"{self.sport_key}:{team_name}"
        ELO_RATINGS[key] = elo
    
    def calculate_win_probability(self, home_elo: float, away_elo: float, is_neutral: bool = False) -> Tuple[float, float]:
        """Calculate win probability based on ELO ratings."""
        # Add home advantage if not neutral site
        adjusted_home_elo = home_elo + (0 if is_neutral else self.config["home_advantage"])
        
        # ELO formula: 1 / (1 + 10^((rating_diff)/400))
        elo_diff = adjusted_home_elo - away_elo
        home_prob = 1 / (1 + math.pow(10, -elo_diff / 400))
        away_prob = 1 - home_prob
        
        return home_prob, away_prob
    
    def update_elos_from_result(self, home_team: str, away_team: str, home_score: int, away_score: int, is_neutral: bool = False):
        """Update ELO ratings after a game."""
        home_elo = self.get_team_elo(home_team)
        away_elo = self.get_team_elo(away_team)
        
        # Calculate expected result
        home_prob, away_prob = self.calculate_win_probability(home_elo, away_elo, is_neutral)
        
        # Actual result (1 = win, 0 = loss)
        home_result = 1 if home_score > away_score else 0
        away_result = 1 - home_result
        
        # Calculate margin of victory multiplier
        margin = abs(home_score - away_score)
        mov_multiplier = 1.0
        if margin >= self.config["blowout_threshold"]:
            mov_multiplier = self.config["blowout_multiplier"]
        
        # Update ELOs
        k = self.config["k_factor"] * mov_multiplier
        new_home_elo = home_elo + k * (home_result - home_prob)
        new_away_elo = away_elo + k * (away_result - away_prob)
        
        self.set_team_elo(home_team, new_home_elo)
        self.set_team_elo(away_team, new_away_elo)
        
        logger.debug(f"ELO Update: {home_team} {home_elo:.0f} -> {new_home_elo:.0f}, {away_team} {away_elo:.0f} -> {new_away_elo:.0f}")
    
    def initialize_elo_from_record(self, team_name: str, wins: int, losses: int):
        """Initialize ELO based on current record."""
        if wins + losses == 0:
            return
        
        win_pct = wins / (wins + losses)
        # Map win percentage to ELO range (1200-1800)
        elo = 1200 + (win_pct * 600)
        self.set_team_elo(team_name, elo)


class NBAAdvancedMetrics:
    """
    NBA-specific advanced metrics calculator.
    Implements Four Factors, Pace, Net Rating, etc.
    """
    
    @staticmethod
    def calculate_four_factors(recent_games: List[Dict]) -> Dict:
        """
        Calculate Dean Oliver's Four Factors:
        1. eFG% (Effective Field Goal %)
        2. TOV% (Turnover Rate)
        3. ORB% (Offensive Rebound Rate)  
        4. FT Rate (Free Throws per FGA)
        
        Estimated from game scores and averages.
        """
        if not recent_games or len(recent_games) < 3:
            return {
                "efg_pct": 0.500,
                "tov_pct": 0.140,
                "orb_pct": 0.250,
                "ft_rate": 0.200,
                "four_factors_score": 50.0
            }
        
        total_points = sum(g.get("our_score", 0) for g in recent_games)
        avg_points = total_points / len(recent_games)
        total_opp_points = sum(g.get("opponent_score", 0) for g in recent_games)
        avg_opp_points = total_opp_points / len(recent_games)
        
        # Estimate eFG% from scoring (league avg ~53%)
        # Higher scoring teams tend to have better eFG%
        efg_pct = 0.500 + ((avg_points - 110) / 100) * 0.15
        efg_pct = max(0.40, min(0.60, efg_pct))
        
        # Estimate TOV% from variance in scoring (more consistent = fewer TOs)
        if len(recent_games) >= 5:
            score_stdev = statistics.stdev([g.get("our_score", 0) for g in recent_games])
            tov_pct = 0.120 + (score_stdev / 100) * 0.05
            tov_pct = max(0.10, min(0.18, tov_pct))
        else:
            tov_pct = 0.140
        
        # Estimate ORB% from margin (teams that dominate boards win bigger)
        avg_margin = sum(g.get("margin", 0) for g in recent_games) / len(recent_games)
        orb_pct = 0.250 + (avg_margin / 50) * 0.10
        orb_pct = max(0.15, min(0.35, orb_pct))
        
        # Estimate FT Rate (league avg ~20%)
        ft_rate = 0.200 + ((avg_points - 110) / 200) * 0.05
        ft_rate = max(0.15, min(0.30, ft_rate))
        
        # Calculate composite Four Factors score (weighted)
        # Weights: eFG% 40%, TOV% 25%, ORB% 20%, FT Rate 15%
        ff_score = (
            (efg_pct - 0.50) * 100 * 0.40 +
            (0.14 - tov_pct) * 100 * 0.25 +  # Lower TOV% is better
            (orb_pct - 0.25) * 100 * 0.20 +
            (ft_rate - 0.20) * 100 * 0.15
        ) + 50  # Baseline 50
        
        return {
            "efg_pct": round(efg_pct, 3),
            "tov_pct": round(tov_pct, 3),
            "orb_pct": round(orb_pct, 3),
            "ft_rate": round(ft_rate, 3),
            "four_factors_score": round(ff_score, 1),
            "avg_points": round(avg_points, 1),
            "avg_opp_points": round(avg_opp_points, 1)
        }
    
    @staticmethod
    def calculate_pace(recent_games: List[Dict]) -> float:
        """Estimate pace (possessions per game)."""
        if not recent_games:
            return 100.0  # League average
        
        total_points = sum(g.get("our_score", 0) + g.get("opponent_score", 0) for g in recent_games)
        avg_total = total_points / len(recent_games) if recent_games else 220
        
        # Estimate pace from total scoring (higher total = faster pace)
        pace = 95.0 + ((avg_total - 220) / 20)
        return max(90.0, min(110.0, pace))
    
    @staticmethod
    def calculate_net_rating(recent_games: List[Dict]) -> float:
        """Calculate net rating (point differential per 100 possessions)."""
        if not recent_games:
            return 0.0
        
        total_margin = sum(g.get("margin", 0) for g in recent_games)
        avg_margin = total_margin / len(recent_games)
        
        # Scale to per 100 possessions (roughly margin * 10)
        net_rating = avg_margin * 0.9
        return round(net_rating, 1)


class NFLAdvancedMetrics:
    """
    NFL-specific advanced metrics.
    Estimates EPA-style efficiency from game results.
    """
    
    @staticmethod
    def calculate_efficiency_rating(recent_games: List[Dict]) -> Dict:
        """
        Calculate offensive and defensive efficiency ratings.
        Based on scoring patterns and variance.
        """
        if not recent_games or len(recent_games) < 3:
            return {
                "offensive_efficiency": 50.0,
                "defensive_efficiency": 50.0,
                "overall_efficiency": 50.0
            }
        
        avg_points = sum(g.get("our_score", 0) for g in recent_games) / len(recent_games)
        avg_opp_points = sum(g.get("opponent_score", 0) for g in recent_games) / len(recent_games)
        
        # Offensive efficiency (league avg ~24 pts)
        off_eff = 40 + ((avg_points - 24) / 20) * 30
        off_eff = max(20, min(80, off_eff))
        
        # Defensive efficiency (lower points allowed = better)
        def_eff = 40 + ((24 - avg_opp_points) / 20) * 30
        def_eff = max(20, min(80, def_eff))
        
        overall_eff = (off_eff + def_eff) / 2
        
        return {
            "offensive_efficiency": round(off_eff, 1),
            "defensive_efficiency": round(def_eff, 1),
            "overall_efficiency": round(overall_eff, 1),
            "avg_points": round(avg_points, 1),
            "avg_opp_points": round(avg_opp_points, 1)
        }


class NHLAdvancedMetrics:
    """
    NHL-specific advanced metrics.
    Estimates possession metrics and PDO.
    """
    
    @staticmethod
    def calculate_possession_metrics(recent_games: List[Dict]) -> Dict:
        """
        Estimate Corsi-like possession metrics from results.
        """
        if not recent_games or len(recent_games) < 3:
            return {
                "possession_rating": 50.0,
                "pdo": 1.000,
                "goal_diff_rating": 50.0
            }
        
        wins = sum(1 for g in recent_games if g.get("won", False))
        win_pct = wins / len(recent_games)
        
        avg_margin = sum(g.get("margin", 0) for g in recent_games) / len(recent_games)
        
        # Estimate possession from win rate and margin
        possession = 45 + (win_pct * 15) + (avg_margin * 3)
        possession = max(35, min(65, possession))
        
        # Estimate PDO (shooting % + save %, league avg = 1.000)
        # Higher win% with small margins = good PDO (luck)
        if abs(avg_margin) < 1.0 and win_pct > 0.55:
            pdo = 1.020
        elif abs(avg_margin) > 2.0:
            pdo = 0.990
        else:
            pdo = 1.000
        
        goal_diff = 50 + (avg_margin * 8)
        goal_diff = max(20, min(80, goal_diff))
        
        return {
            "possession_rating": round(possession, 1),
            "pdo": round(pdo, 3),
            "goal_diff_rating": round(goal_diff, 1)
        }


def calculate_advanced_metrics(sport_key: str, team_data: Dict) -> Dict:
    """
    Main entry point for calculating sport-specific advanced metrics.
    """
    recent_games = team_data.get("recent_games", [])
    form = team_data.get("form", {})
    
    metrics = {
        "sport_key": sport_key,
        "team_name": team_data.get("name", ""),
        "basic_stats": {
            "wins": form.get("wins", 0),
            "losses": form.get("losses", 0),
            "win_pct": form.get("win_pct", 0.5),
            "avg_margin": form.get("avg_margin", 0.0)
        }
    }
    
    # Calculate sport-specific metrics
    if sport_key == "basketball_nba":
        metrics["four_factors"] = NBAAdvancedMetrics.calculate_four_factors(recent_games)
        metrics["pace"] = NBAAdvancedMetrics.calculate_pace(recent_games)
        metrics["net_rating"] = NBAAdvancedMetrics.calculate_net_rating(recent_games)
    
    elif sport_key == "americanfootball_nfl":
        metrics["efficiency"] = NFLAdvancedMetrics.calculate_efficiency_rating(recent_games)
    
    elif sport_key == "icehockey_nhl":
        metrics["possession"] = NHLAdvancedMetrics.calculate_possession_metrics(recent_games)
    
    # ELO rating
    elo_system = ELORatingSystem(sport_key)
    team_elo = elo_system.get_team_elo(team_data.get("name", ""))
    
    # Initialize from record if ELO is at default
    if team_elo == elo_system.config["initial_elo"]:
        # Try to get season record from stats first, fallback to form (last 10)
        stats = team_data.get("stats", {})
        record_str = stats.get("record", "")
        
        if record_str and "-" in record_str:
            # Parse "32-18" format
            try:
                parts = record_str.split("-")
                wins = int(parts[0])
                losses = int(parts[1])
            except (ValueError, IndexError):
                wins = form.get("wins", 0)
                losses = form.get("losses", 0)
        else:
            wins = form.get("wins", 0)
            losses = form.get("losses", 0)
        
        if wins + losses > 0:
            elo_system.initialize_elo_from_record(team_data.get("name", ""), wins, losses)
            team_elo = elo_system.get_team_elo(team_data.get("name", ""))
    
    metrics["elo_rating"] = round(team_elo, 0)
    
    return metrics


def calculate_matchup_metrics(sport_key: str, home_metrics: Dict, away_metrics: Dict) -> Dict:
    """
    Calculate matchup-specific metrics comparing two teams.
    """
    home_elo = home_metrics.get("elo_rating", 1500)
    away_elo = away_metrics.get("elo_rating", 1500)
    
    matchup = {
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_advantage": home_elo - away_elo,
        "win_pct_diff": home_metrics["basic_stats"]["win_pct"] - away_metrics["basic_stats"]["win_pct"],
        "margin_diff": home_metrics["basic_stats"]["avg_margin"] - away_metrics["basic_stats"]["avg_margin"]
    }
    
    # Sport-specific comparisons
    if sport_key == "basketball_nba":
        home_ff = home_metrics.get("four_factors", {}).get("four_factors_score", 50)
        away_ff = away_metrics.get("four_factors", {}).get("four_factors_score", 50)
        matchup["four_factors_diff"] = home_ff - away_ff
        matchup["net_rating_diff"] = home_metrics.get("net_rating", 0) - away_metrics.get("net_rating", 0)
    
    elif sport_key == "americanfootball_nfl":
        home_eff = home_metrics.get("efficiency", {}).get("overall_efficiency", 50)
        away_eff = away_metrics.get("efficiency", {}).get("overall_efficiency", 50)
        matchup["efficiency_diff"] = home_eff - away_eff
    
    elif sport_key == "icehockey_nhl":
        home_poss = home_metrics.get("possession", {}).get("possession_rating", 50)
        away_poss = away_metrics.get("possession", {}).get("possession_rating", 50)
        matchup["possession_diff"] = home_poss - away_poss
    
    return matchup



async def load_elo_cache_from_db(db):
    """
    Load trained ELO ratings from MongoDB into memory cache.
    Call this on server startup to use trained ELO values.
    """
    global DB_ELO_CACHE
    
    try:
        # Load from elo_ratings collection (populated by XGBoost training)
        cursor = db.elo_ratings.find({}, {"_id": 0, "sport_key": 1, "team_name": 1, "elo": 1})
        ratings = await cursor.to_list(1000)
        
        for rating in ratings:
            sport_key = rating.get("sport_key", "")
            team_name = rating.get("team_name", "")
            elo = rating.get("elo", 1500)
            
            if sport_key and team_name:
                cache_key = f"{sport_key}:{team_name}"
                DB_ELO_CACHE[cache_key] = elo
        
        logger.info(f"✅ Loaded {len(DB_ELO_CACHE)} ELO ratings from database")
        return len(DB_ELO_CACHE)
        
    except Exception as e:
        logger.error(f"Failed to load ELO cache from DB: {e}")
        return 0


def get_cached_elo(sport_key: str, team_name: str) -> float:
    """Get ELO from cache, with fallback to default."""
    cache_key = f"{sport_key}:{team_name}"
    return DB_ELO_CACHE.get(cache_key, ELO_CONFIG.get(sport_key, {}).get("initial_elo", 1500))
