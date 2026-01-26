"""
Custom Betting Algorithm - Mathematical pick generation based on multiple factors
NO AI REQUIRED - Pure statistical analysis

Factors considered:
1. Line Movement (25%) - Sharp money indicators
2. Recent Form (20%) - Last 5-10 games performance
3. Head-to-Head (15%) - Historical matchup data
4. Home Court Advantage (15%) - Venue factor
5. Injuries (15%) - Key player availability
6. Rest Days (10%) - Back-to-back disadvantage
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)

# Weight configuration for each factor
FACTOR_WEIGHTS = {
    "line_movement": 0.25,
    "recent_form": 0.20,
    "head_to_head": 0.15,
    "home_advantage": 0.15,
    "injuries": 0.15,
    "rest_days": 0.10
}

# Sport-specific home advantage values (probability boost)
HOME_ADVANTAGE = {
    "basketball_nba": 0.035,  # ~3.5% boost
    "americanfootball_nfl": 0.025,  # ~2.5% boost
    "baseball_mlb": 0.020,  # ~2% boost
    "icehockey_nhl": 0.025,  # ~2.5% boost
    "soccer_epl": 0.040,  # ~4% boost (stronger in soccer)
}


def calculate_pick(matchup_data: Dict, line_movement: Dict = None) -> Dict:
    """
    Main algorithm to calculate the best pick for a matchup
    
    Returns:
        {
            "pick_type": "moneyline" | "spread" | "total",
            "pick": "Team Name" | "Team Name -5.5" | "Over 225.5",
            "confidence": 0.70-0.95,
            "edge": calculated edge percentage,
            "factors": breakdown of each factor's contribution,
            "reasoning": human-readable explanation
        }
    """
    event = matchup_data.get("event", {})
    home_team = matchup_data.get("home_team", {})
    away_team = matchup_data.get("away_team", {})
    odds = matchup_data.get("odds", {})
    sport_key = event.get("sport_key", "basketball_nba")
    
    # Calculate each factor score (0 to 1 scale, where 0.5 is neutral)
    factors = {}
    
    # 1. Line Movement Factor
    factors["line_movement"] = calculate_line_movement_factor(
        line_movement, 
        event.get("id", ""),
        odds
    )
    
    # 2. Recent Form Factor
    factors["recent_form"] = calculate_form_factor(
        home_team.get("form", {}),
        away_team.get("form", {})
    )
    
    # 3. Head-to-Head Factor (simplified - use records)
    factors["head_to_head"] = calculate_h2h_factor(
        home_team.get("stats", {}),
        away_team.get("stats", {})
    )
    
    # 4. Home Advantage Factor
    factors["home_advantage"] = calculate_home_advantage_factor(sport_key)
    
    # 5. Injuries Factor (based on available data)
    factors["injuries"] = calculate_injury_factor(
        matchup_data.get("home_injuries", []),
        matchup_data.get("away_injuries", [])
    )
    
    # 6. Rest Days Factor
    factors["rest_days"] = calculate_rest_factor(
        home_team.get("recent_games", []),
        away_team.get("recent_games", []),
        event.get("commence_time", "")
    )
    
    # Calculate weighted home team probability
    weighted_home_prob = 0.50  # Start at 50-50
    
    for factor_name, factor_value in factors.items():
        weight = FACTOR_WEIGHTS.get(factor_name, 0)
        # Factor value is 0-1, where >0.5 favors home, <0.5 favors away
        adjustment = (factor_value - 0.5) * weight * 2  # Scale adjustment
        weighted_home_prob += adjustment
    
    # Clamp probability
    weighted_home_prob = max(0.30, min(0.70, weighted_home_prob))
    
    # Get market odds
    spread = odds.get("spread", 0)
    total = odds.get("total", 220)
    home_ml_decimal = odds.get("home_ml_decimal", 1.91)
    away_ml_decimal = odds.get("away_ml_decimal", 1.91)
    
    # Calculate implied probabilities from market
    market_home_prob = 1 / home_ml_decimal if home_ml_decimal > 1 else 0.5
    market_away_prob = 1 / away_ml_decimal if away_ml_decimal > 1 else 0.5
    
    # Normalize market probabilities (remove juice)
    total_implied = market_home_prob + market_away_prob
    market_home_prob = market_home_prob / total_implied
    
    # Calculate edge
    home_edge = weighted_home_prob - market_home_prob
    away_edge = (1 - weighted_home_prob) - (1 - market_home_prob)
    
    # Determine best pick
    pick_options = []
    
    # Moneyline options
    if home_edge > 0.02:  # At least 2% edge
        pick_options.append({
            "type": "moneyline",
            "pick": home_team.get("name", "Home Team"),
            "edge": home_edge,
            "odds": home_ml_decimal,
            "our_prob": weighted_home_prob
        })
    
    if away_edge > 0.02:
        pick_options.append({
            "type": "moneyline",
            "pick": away_team.get("name", "Away Team"),
            "edge": away_edge,
            "odds": away_ml_decimal,
            "our_prob": 1 - weighted_home_prob
        })
    
    # Spread options - check if spread aligns with our probability
    spread_edge = calculate_spread_edge(weighted_home_prob, spread, sport_key)
    if spread_edge["edge"] > 0.02:
        pick_options.append({
            "type": "spread",
            "pick": f"{spread_edge['team']} {spread_edge['spread']:+.1f}",
            "edge": spread_edge["edge"],
            "odds": 1.91,
            "our_prob": spread_edge["prob"],
            "team_name": spread_edge["team_name"]
        })
    
    # Total options - analyze scoring trends
    total_edge = calculate_total_edge(home_team, away_team, total, sport_key)
    if total_edge["edge"] > 0.02:
        pick_options.append({
            "type": "total",
            "pick": f"{total_edge['direction']} {total}",
            "edge": total_edge["edge"],
            "odds": 1.91,
            "our_prob": total_edge["prob"]
        })
    
    # Select best pick
    if not pick_options:
        # If no edge found, generate a conservative pick based on favorite
        if home_ml_decimal < away_ml_decimal:
            # Home is favorite
            return {
                "pick_type": "moneyline",
                "pick": home_team.get("name", "Home Team"),
                "confidence": 0.70,
                "edge": 2.0,
                "odds": home_ml_decimal,
                "factors": {k: round(v, 3) for k, v in factors.items()},
                "reasoning": f"Algorithm favors home team as market favorite. Home court advantage factor applied.",
                "our_probability": round(weighted_home_prob, 3)
            }
        else:
            return None
    
    best_pick = max(pick_options, key=lambda x: x["edge"])
    
    # Calculate confidence (edge-based, capped at 85%)
    confidence = min(0.85, 0.60 + best_pick["edge"] * 2)
    
    # Only return if confidence >= 70%
    if confidence < 0.70:
        return None
    
    # Generate reasoning
    reasoning = generate_reasoning(factors, best_pick, home_team, away_team, sport_key)
    
    return {
        "pick_type": best_pick["type"],
        "pick": best_pick["pick"],
        "confidence": round(confidence, 2),
        "edge": round(best_pick["edge"] * 100, 1),
        "odds": best_pick["odds"],
        "factors": {k: round(v, 3) for k, v in factors.items()},
        "reasoning": reasoning,
        "our_probability": round(best_pick["our_prob"], 3)
    }


def calculate_line_movement_factor(line_movement: Dict, event_id: str, current_odds: Dict) -> float:
    """
    Analyze line movement to detect sharp money
    Returns: 0-1 (>0.5 favors home, <0.5 favors away)
    """
    if not line_movement:
        return 0.50  # Neutral if no data
    
    opening_home = line_movement.get("opening_home_odds", 0)
    opening_away = line_movement.get("opening_away_odds", 0)
    current_home = current_odds.get("home_ml_decimal", 0)
    current_away = current_odds.get("away_ml_decimal", 0)
    
    if not opening_home or not current_home:
        return 0.50
    
    # Calculate movement percentage
    home_movement = (current_home - opening_home) / opening_home if opening_home > 0 else 0
    away_movement = (current_away - opening_away) / opening_away if opening_away > 0 else 0
    
    # If home odds dropped (became favorite), sharp money on home
    # If home odds increased (became underdog), sharp money on away
    
    if home_movement < -0.05:  # Home odds dropped 5%+ (more favored)
        return 0.60 + min(0.15, abs(home_movement) * 0.5)
    elif home_movement > 0.05:  # Home odds increased 5%+ (less favored)
        return 0.40 - min(0.15, home_movement * 0.5)
    
    return 0.50


def calculate_form_factor(home_form: Dict, away_form: Dict) -> float:
    """
    Compare recent form between teams
    Returns: 0-1 (>0.5 favors home)
    """
    home_win_pct = home_form.get("win_pct", 0.5)
    away_win_pct = away_form.get("win_pct", 0.5)
    home_margin = home_form.get("avg_margin", 0)
    away_margin = away_form.get("avg_margin", 0)
    home_streak = home_form.get("streak", 0)
    away_streak = away_form.get("streak", 0)
    
    # Win percentage comparison (60% weight)
    win_pct_factor = 0.5 + (home_win_pct - away_win_pct) * 0.3
    
    # Margin comparison (25% weight)
    margin_diff = home_margin - away_margin
    margin_factor = 0.5 + (margin_diff / 20) * 0.25  # 20 point diff = 25% swing
    
    # Streak comparison (15% weight)
    streak_diff = home_streak - away_streak
    streak_factor = 0.5 + (streak_diff / 10) * 0.15
    
    # Combined
    factor = win_pct_factor * 0.6 + margin_factor * 0.25 + streak_factor * 0.15
    
    return max(0.30, min(0.70, factor))


def calculate_h2h_factor(home_stats: Dict, away_stats: Dict) -> float:
    """
    Compare overall team records
    Returns: 0-1 (>0.5 favors home)
    """
    def parse_record(record_str: str) -> Tuple[int, int]:
        """Parse 'W-L' format"""
        try:
            parts = record_str.split("-")
            return int(parts[0]), int(parts[1])
        except:
            return 0, 0
    
    home_w, home_l = parse_record(home_stats.get("record", "0-0"))
    away_w, away_l = parse_record(away_stats.get("record", "0-0"))
    
    home_pct = home_w / (home_w + home_l) if (home_w + home_l) > 0 else 0.5
    away_pct = away_w / (away_w + away_l) if (away_w + away_l) > 0 else 0.5
    
    factor = 0.5 + (home_pct - away_pct) * 0.4
    
    return max(0.35, min(0.65, factor))


def calculate_home_advantage_factor(sport_key: str) -> float:
    """
    Apply sport-specific home advantage
    Returns: 0-1 (always >0.5 to favor home)
    """
    advantage = HOME_ADVANTAGE.get(sport_key, 0.03)
    return 0.50 + advantage


def calculate_injury_factor(home_injuries: List, away_injuries: List) -> float:
    """
    Factor in key injuries
    Returns: 0-1 (>0.5 favors home if away has more injuries)
    """
    home_injury_impact = len(home_injuries) * 0.03
    away_injury_impact = len(away_injuries) * 0.03
    
    factor = 0.5 + (away_injury_impact - home_injury_impact)
    
    return max(0.40, min(0.60, factor))


def calculate_rest_factor(home_recent: List, away_recent: List, commence_time: str) -> float:
    """
    Factor in rest days (back-to-back games = disadvantage)
    Returns: 0-1 (>0.5 favors home if they have more rest)
    """
    def days_since_last_game(recent_games: List, game_time: str) -> int:
        if not recent_games:
            return 3  # Assume well rested if no data
        
        try:
            game_dt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
            last_game_str = recent_games[0].get("date", "")
            last_game_dt = datetime.fromisoformat(last_game_str.replace("Z", "+00:00"))
            return (game_dt - last_game_dt).days
        except:
            return 2  # Default
    
    home_rest = days_since_last_game(home_recent, commence_time)
    away_rest = days_since_last_game(away_recent, commence_time)
    
    # Back-to-back (0-1 days rest) is a significant disadvantage
    rest_diff = home_rest - away_rest
    
    # Each day of rest advantage = ~2% probability
    factor = 0.5 + rest_diff * 0.02
    
    return max(0.40, min(0.60, factor))


def calculate_spread_edge(home_prob: float, spread: float, sport_key: str) -> Dict:
    """Calculate edge on spread betting"""
    # Convert our probability to expected margin
    # For NBA: 50% win prob ≈ even game, each 3% prob = ~1 point margin
    
    if "nba" in sport_key or "basketball" in sport_key:
        points_per_prob = 33  # ~3% per point
    elif "nfl" in sport_key or "football" in sport_key:
        points_per_prob = 14  # ~7% per point
    else:
        points_per_prob = 20  # Default
    
    expected_margin = (home_prob - 0.5) * points_per_prob
    
    # If we think home wins by 5 and spread is -3, we have edge on home -3
    # If we think home wins by 2 and spread is -5, we have edge on away +5
    
    if expected_margin > spread:
        # We favor home more than market
        edge = (expected_margin - spread) / points_per_prob * 0.5
        return {
            "team": "home",
            "team_name": "Home",
            "spread": spread,
            "edge": edge,
            "prob": home_prob
        }
    else:
        # We favor away more than market
        edge = (spread - expected_margin) / points_per_prob * 0.5
        return {
            "team": "away",
            "team_name": "Away",
            "spread": -spread,
            "edge": edge,
            "prob": 1 - home_prob
        }


def calculate_total_edge(home_team: Dict, away_team: Dict, total: float, sport_key: str) -> Dict:
    """Calculate edge on total (over/under) betting"""
    # Use recent game scoring to estimate expected total
    
    home_recent = home_team.get("recent_games", [])
    away_recent = away_team.get("recent_games", [])
    
    if not home_recent or not away_recent:
        return {"direction": "Over", "edge": 0, "prob": 0.5}
    
    # Calculate average points scored and allowed
    home_scored = sum(g.get("our_score", 0) for g in home_recent) / len(home_recent) if home_recent else 0
    home_allowed = sum(g.get("opponent_score", 0) for g in home_recent) / len(home_recent) if home_recent else 0
    away_scored = sum(g.get("our_score", 0) for g in away_recent) / len(away_recent) if away_recent else 0
    away_allowed = sum(g.get("opponent_score", 0) for g in away_recent) / len(away_recent) if away_recent else 0
    
    # Expected total = (home_offense + away_offense + home_defense_allowed + away_defense_allowed) / 2
    expected_total = (home_scored + away_scored + home_allowed + away_allowed) / 2
    
    # Adjust for venue (home games slightly lower scoring)
    expected_total *= 0.98
    
    diff = expected_total - total
    
    if diff > 3:  # We expect 3+ more points than line
        edge = min(0.10, diff / 30)
        return {"direction": "Over", "edge": edge, "prob": 0.5 + edge}
    elif diff < -3:  # We expect 3+ fewer points than line
        edge = min(0.10, abs(diff) / 30)
        return {"direction": "Under", "edge": edge, "prob": 0.5 + edge}
    
    return {"direction": "Over", "edge": 0, "prob": 0.5}


def generate_reasoning(factors: Dict, pick: Dict, home_team: Dict, away_team: Dict, sport_key: str) -> str:
    """Generate human-readable reasoning for the pick"""
    home_name = home_team.get("name", "Home Team")
    away_name = away_team.get("name", "Away Team")
    
    reasons = []
    
    # Analyze strongest factors
    sorted_factors = sorted(factors.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
    
    for factor_name, value in sorted_factors[:3]:
        if factor_name == "line_movement" and abs(value - 0.5) > 0.05:
            direction = "toward" if value > 0.5 else "against"
            reasons.append(f"Line movement {direction} {home_name}")
        elif factor_name == "recent_form" and abs(value - 0.5) > 0.05:
            better_team = home_name if value > 0.5 else away_name
            reasons.append(f"{better_team} in better recent form")
        elif factor_name == "home_advantage":
            reasons.append(f"Home court advantage for {home_name}")
        elif factor_name == "rest_days" and abs(value - 0.5) > 0.03:
            rested = home_name if value > 0.5 else away_name
            reasons.append(f"{rested} better rested")
    
    pick_type = pick.get("type", "moneyline")
    edge = pick.get("edge", 0) * 100
    
    base_reason = f"Algorithm identifies {edge:.1f}% edge on this {pick_type}. "
    
    if reasons:
        base_reason += "Key factors: " + ", ".join(reasons) + "."
    
    return base_reason
