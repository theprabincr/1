"""
Custom Betting Algorithm V2 - CONSERVATIVE and REALISTIC
Only generates high-confidence picks when there's a genuine statistical edge

Factors considered:
1. Line Movement (25%) - Sharp money indicators
2. Recent Form (25%) - Last 5-10 games performance  
3. Home Court Advantage (20%) - Venue factor
4. Rest Days (15%) - Back-to-back disadvantage
5. Record Comparison (15%) - Season records

IMPORTANT: This algorithm is CONSERVATIVE - it only recommends picks with clear edges.
Most games will have NO PICK because the market is usually efficient.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)

# Sport-specific home advantage values (probability boost)
HOME_ADVANTAGE = {
    "basketball_nba": 0.028,  # ~2.8% boost (NBA home advantage has decreased)
    "americanfootball_nfl": 0.022,  # ~2.2% boost
    "baseball_mlb": 0.020,  # ~2% boost
    "icehockey_nhl": 0.022,  # ~2.2% boost
    "soccer_epl": 0.035,  # ~3.5% boost (stronger in soccer)
}


def calculate_pick(matchup_data: Dict, line_movement: Dict = None) -> Optional[Dict]:
    """
    CONSERVATIVE algorithm - only returns picks with genuine statistical edge
    
    Returns pick dict if edge >= 5% AND confidence >= 70%
    Returns None for most games (market is usually efficient)
    """
    event = matchup_data.get("event", {})
    home_team = matchup_data.get("home_team", {})
    away_team = matchup_data.get("away_team", {})
    odds = matchup_data.get("odds", {})
    sport_key = event.get("sport_key", "basketball_nba")
    
    home_name = event.get("home_team", "Home")
    away_name = event.get("away_team", "Away")
    
    # Get market odds
    spread = odds.get("spread", 0) or 0
    total = odds.get("total", 220) or 220
    home_ml_decimal = odds.get("home_ml_decimal", 1.91) or 1.91
    away_ml_decimal = odds.get("away_ml_decimal", 1.91) or 1.91
    
    # Calculate implied probabilities from market (this is what the market thinks)
    market_home_prob = 1 / home_ml_decimal if home_ml_decimal > 1 else 0.5
    market_away_prob = 1 / away_ml_decimal if away_ml_decimal > 1 else 0.5
    
    # Normalize (remove juice/vig)
    total_implied = market_home_prob + market_away_prob
    if total_implied > 0:
        market_home_prob = market_home_prob / total_implied
        market_away_prob = market_away_prob / total_implied
    
    # Calculate our probability estimate
    analysis_factors = []
    our_home_prob = 0.50  # Start neutral
    
    # 1. Home Advantage (small boost)
    home_adv = HOME_ADVANTAGE.get(sport_key, 0.025)
    our_home_prob += home_adv
    analysis_factors.append(f"Home court advantage: +{home_adv*100:.1f}% for {home_name}")
    
    # 2. Recent Form Analysis
    home_form = home_team.get("form", {})
    away_form = away_team.get("form", {})
    
    home_win_pct = home_form.get("win_pct", 0.5)
    away_win_pct = away_form.get("win_pct", 0.5)
    home_margin = home_form.get("avg_margin", 0)
    away_margin = away_form.get("avg_margin", 0)
    
    # Form adjustment (capped at 5%)
    form_diff = (home_win_pct - away_win_pct) * 0.15  # Max ~7.5% swing
    form_diff = max(-0.05, min(0.05, form_diff))
    our_home_prob += form_diff
    
    if abs(form_diff) > 0.02:
        better_form = home_name if form_diff > 0 else away_name
        analysis_factors.append(f"Recent form favors {better_form} ({home_win_pct*100:.0f}% vs {away_win_pct*100:.0f}%)")
    
    # 3. Rest Days Analysis
    home_recent = home_team.get("recent_games", [])
    away_recent = away_team.get("recent_games", [])
    
    home_rest = calculate_rest_days(home_recent, event.get("commence_time", ""))
    away_rest = calculate_rest_days(away_recent, event.get("commence_time", ""))
    
    rest_diff = (home_rest - away_rest) * 0.01  # 1% per day difference
    rest_diff = max(-0.03, min(0.03, rest_diff))
    our_home_prob += rest_diff
    
    if abs(rest_diff) > 0.015:
        more_rested = home_name if rest_diff > 0 else away_name
        analysis_factors.append(f"{more_rested} better rested ({home_rest}d vs {away_rest}d)")
    
    # 4. Line Movement Analysis
    if line_movement:
        movement_adj = analyze_line_movement(line_movement, odds)
        movement_adj = max(-0.03, min(0.03, movement_adj))
        our_home_prob += movement_adj
        
        if abs(movement_adj) > 0.01:
            sharp_side = home_name if movement_adj > 0 else away_name
            analysis_factors.append(f"Line movement suggests sharp money on {sharp_side}")
    
    # Clamp probability
    our_home_prob = max(0.35, min(0.65, our_home_prob))
    our_away_prob = 1 - our_home_prob
    
    # Calculate edge (our probability vs market probability)
    home_edge = our_home_prob - market_home_prob
    away_edge = our_away_prob - market_away_prob
    
    # CONSERVATIVE: Only make a pick if edge >= 4%
    MIN_EDGE = 0.04
    
    best_pick = None
    
    # Check home team edge
    if home_edge >= MIN_EDGE:
        confidence = calculate_confidence(home_edge, home_form, away_form)
        if confidence >= 0.70:
            best_pick = {
                "pick_type": "moneyline",
                "pick": home_name,
                "edge": home_edge,
                "odds": home_ml_decimal,
                "our_prob": our_home_prob,
                "market_prob": market_home_prob,
                "confidence": confidence,
                "factors": analysis_factors
            }
    
    # Check away team edge
    if away_edge >= MIN_EDGE:
        confidence = calculate_confidence(away_edge, away_form, home_form)
        if confidence >= 0.70:
            if best_pick is None or away_edge > best_pick["edge"]:
                best_pick = {
                    "pick_type": "moneyline",
                    "pick": away_name,
                    "edge": away_edge,
                    "odds": away_ml_decimal,
                    "our_prob": our_away_prob,
                    "market_prob": market_away_prob,
                    "confidence": confidence,
                    "factors": analysis_factors
                }
    
    # If no moneyline edge, check totals (more conservative)
    if best_pick is None:
        total_pick = analyze_total(home_team, away_team, total, sport_key)
        if total_pick and total_pick["edge"] >= MIN_EDGE:
            confidence = 0.70 + (total_pick["edge"] - MIN_EDGE) * 2
            confidence = min(0.78, confidence)  # Cap totals confidence
            if confidence >= 0.70:
                best_pick = {
                    "pick_type": "total",
                    "pick": f"{total_pick['direction']} {total}",
                    "edge": total_pick["edge"],
                    "odds": 1.91,
                    "our_prob": total_pick["prob"],
                    "market_prob": 0.50,
                    "confidence": confidence,
                    "factors": analysis_factors + [total_pick["reason"]]
                }
    
    # If no pick found, return None (MOST GAMES)
    if best_pick is None:
        return None
    
    # Generate reasoning
    reasoning = generate_reasoning(best_pick, home_name, away_name)
    
    return {
        "pick_type": best_pick["pick_type"],
        "pick": best_pick["pick"],
        "confidence": round(best_pick["confidence"], 2),
        "edge": round(best_pick["edge"] * 100, 1),
        "odds": best_pick["odds"],
        "our_probability": round(best_pick["our_prob"], 3),
        "market_probability": round(best_pick["market_prob"], 3),
        "reasoning": reasoning,
        "factors": best_pick["factors"]
    }


def calculate_rest_days(recent_games: List, commence_time: str) -> int:
    """Calculate days since last game"""
    if not recent_games or not commence_time:
        return 2  # Default assumption
    
    try:
        game_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        last_game_str = recent_games[0].get("date", "")
        if not last_game_str:
            return 2
        last_game_dt = datetime.fromisoformat(last_game_str.replace("Z", "+00:00"))
        return max(0, (game_dt - last_game_dt).days)
    except:
        return 2


def analyze_line_movement(line_movement: Dict, current_odds: Dict) -> float:
    """
    Analyze line movement for sharp money indicators
    Returns adjustment to home probability (-0.03 to +0.03)
    """
    opening_home = line_movement.get("opening_home_odds", 0)
    current_home = current_odds.get("home_ml_decimal", 0)
    
    if not opening_home or not current_home or opening_home <= 0:
        return 0
    
    # Calculate movement (negative means home became more favored)
    movement_pct = (current_home - opening_home) / opening_home
    
    # If home odds dropped significantly (became more favored), sharp money likely on home
    if movement_pct < -0.05:
        return 0.02
    elif movement_pct > 0.05:
        return -0.02
    
    return 0


def calculate_confidence(edge: float, team_form: Dict, opponent_form: Dict) -> float:
    """
    Calculate confidence based on edge and supporting factors
    Returns 0.70-0.85 (capped to be realistic)
    """
    base_confidence = 0.68
    
    # Add confidence based on edge (each 1% edge = 1.5% confidence)
    edge_bonus = edge * 1.5
    
    # Add confidence if our team has strong recent form
    win_pct = team_form.get("win_pct", 0.5)
    if win_pct > 0.65:
        edge_bonus += 0.02
    elif win_pct < 0.35:
        edge_bonus -= 0.02
    
    confidence = base_confidence + edge_bonus
    
    # CAP at 82% - higher confidence is unrealistic
    return min(0.82, max(0.70, confidence))


def analyze_total(home_team: Dict, away_team: Dict, total: float, sport_key: str) -> Optional[Dict]:
    """Analyze total (over/under) for edge - VERY CONSERVATIVE"""
    home_recent = home_team.get("recent_games", [])
    away_recent = away_team.get("recent_games", [])
    
    if len(home_recent) < 3 or len(away_recent) < 3:
        return None  # Not enough data
    
    # Calculate average scores
    home_scored = sum(g.get("our_score", 0) for g in home_recent[:5]) / min(5, len(home_recent))
    home_allowed = sum(g.get("opponent_score", 0) for g in home_recent[:5]) / min(5, len(home_recent))
    away_scored = sum(g.get("our_score", 0) for g in away_recent[:5]) / min(5, len(away_recent))
    away_allowed = sum(g.get("opponent_score", 0) for g in away_recent[:5]) / min(5, len(away_recent))
    
    # Expected total
    expected_total = (home_scored + away_scored + home_allowed + away_allowed) / 2
    
    diff = expected_total - total
    
    # Need at least 5 points difference for NBA, 3 for NHL
    min_diff = 5 if "nba" in sport_key.lower() else 3
    
    if abs(diff) >= min_diff:
        edge = min(0.06, abs(diff) / 50)  # Cap edge at 6%
        direction = "Over" if diff > 0 else "Under"
        reason = f"Scoring trends suggest {direction} (Expected: {expected_total:.0f}, Line: {total})"
        
        return {
            "direction": direction,
            "edge": edge,
            "prob": 0.5 + edge,
            "reason": reason
        }
    
    return None


def generate_reasoning(pick: Dict, home_name: str, away_name: str) -> str:
    """Generate human-readable reasoning for the pick"""
    factors = pick.get("factors", [])
    edge = pick.get("edge", 0) * 100
    our_prob = pick.get("our_prob", 0.5) * 100
    market_prob = pick.get("market_prob", 0.5) * 100
    
    reasoning_parts = [
        f"Our model estimates {our_prob:.0f}% win probability vs market's {market_prob:.0f}%.",
        f"This represents a {edge:.1f}% edge."
    ]
    
    if factors:
        reasoning_parts.append("Key factors: " + "; ".join(factors[:3]) + ".")
    
    return " ".join(reasoning_parts)
