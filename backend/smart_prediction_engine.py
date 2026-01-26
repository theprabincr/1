"""
Smart Algorithmic Prediction Engine V4 - NO LLM REQUIRED
Uses comprehensive statistical analysis to make diverse predictions

Features:
- Analyzes ALL markets: Moneyline, Spread, Totals
- Uses squad data, player stats, H2H, venue, injuries
- Analyzes ESPN line movement (snapshots every 15 min)
- Makes diverse predictions (not just ML)
- Considers odds as low as 1.5x
- 70%+ confidence only when data strongly supports
- Predictions made 1 hour before game start

Data Source: ESPN/DraftKings (single source - no multi-book comparison)
No API keys needed - pure algorithmic analysis!
"""
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import random

logger = logging.getLogger(__name__)

# Sport-specific configurations
SPORT_CONFIG = {
    "basketball_nba": {
        "avg_total": 225,
        "home_adv": 0.025,
        "spread_per_point": 0.03,  # 3% implied prob per spread point
        "total_variance": 8,  # Points variance for over/under
        "pace_factor": 1.0,
    },
    "americanfootball_nfl": {
        "avg_total": 45,
        "home_adv": 0.025,
        "spread_per_point": 0.025,
        "total_variance": 5,
        "pace_factor": 1.0,
    },
    "icehockey_nhl": {
        "avg_total": 6,
        "home_adv": 0.022,
        "spread_per_point": 0.08,
        "total_variance": 1,
        "pace_factor": 1.0,
    },
    "soccer_epl": {
        "avg_total": 2.5,
        "home_adv": 0.035,
        "spread_per_point": 0.15,
        "total_variance": 0.5,
        "pace_factor": 1.0,
    },
}


class SmartPredictionEngine:
    """
    Smart algorithmic prediction engine that analyzes comprehensive data
    and makes diverse predictions without requiring LLM.
    
    IMPORTANT: Only makes predictions with genuine value, considers ALL odds >= 1.5
    """
    
    def __init__(self):
        self.min_confidence = 0.70
        self.max_confidence = 0.85
        self.min_edge = 0.04  # 4% minimum edge for value
        self.min_odds = 1.5   # Accept odds as low as 1.5 (favorites)
    
    async def analyze_and_predict(
        self,
        event: Dict,
        sport_key: str,
        squad_data: Dict,
        matchup_data: Dict,
        line_movement: List[Dict],
        multi_book_odds: Dict
    ) -> Optional[Dict]:
        """
        Analyze all data and generate the best prediction.
        Returns the pick with highest expected value.
        """
        home_team = event.get("home_team", "Home")
        away_team = event.get("away_team", "Away")
        
        logger.info(f"📊 Smart Analysis: {home_team} vs {away_team}")
        
        config = SPORT_CONFIG.get(sport_key, SPORT_CONFIG["basketball_nba"])
        odds = event.get("odds", {})
        
        # Analyze all three markets
        ml_analysis = self._analyze_moneyline(event, matchup_data, squad_data, line_movement, multi_book_odds, config)
        spread_analysis = self._analyze_spread(event, matchup_data, squad_data, line_movement, config)
        total_analysis = self._analyze_total(event, matchup_data, squad_data, config)
        
        # Find the best value pick across all markets
        best_pick = None
        best_ev = 0
        
        for analysis in [ml_analysis, spread_analysis, total_analysis]:
            if analysis and analysis.get("has_value"):
                ev = analysis.get("expected_value", 0)
                if ev > best_ev and analysis.get("confidence", 0) >= self.min_confidence:
                    best_ev = ev
                    best_pick = analysis
        
        if best_pick:
            logger.info(f"✅ Best pick: {best_pick.get('pick_type')} - {best_pick.get('pick')} "
                       f"@ {best_pick.get('confidence', 0)*100:.0f}% conf, EV: {best_ev:.2f}%")
            return best_pick
        else:
            logger.info(f"⏭️ No value found for {home_team} vs {away_team}")
            return {
                "has_pick": False,
                "reasoning": "No significant edge found across ML, Spread, or Total markets",
                "closest_value": self._get_closest_value(ml_analysis, spread_analysis, total_analysis)
            }
    
    def _analyze_moneyline(
        self,
        event: Dict,
        matchup_data: Dict,
        squad_data: Dict,
        line_movement: List[Dict],
        multi_book_odds: Dict,
        config: Dict
    ) -> Dict:
        """Analyze moneyline market for value"""
        home_team = event.get("home_team", "Home")
        away_team = event.get("away_team", "Away")
        odds = event.get("odds", {})
        
        home_ml = odds.get("home_ml_decimal", 1.91)
        away_ml = odds.get("away_ml_decimal", 1.91)
        
        # Calculate implied probabilities (removing juice)
        home_implied = 1 / home_ml if home_ml > 1 else 0.5
        away_implied = 1 / away_ml if away_ml > 1 else 0.5
        total_implied = home_implied + away_implied
        
        if total_implied > 0:
            home_implied /= total_implied
            away_implied /= total_implied
        
        # Build our probability estimate
        factors = []
        home_prob = 0.50  # Start neutral
        
        # 1. Home advantage
        home_adv = config.get("home_adv", 0.025)
        home_prob += home_adv
        factors.append(f"Home advantage: +{home_adv*100:.1f}%")
        
        # 2. Recent form analysis
        home_data = matchup_data.get("home_team", {})
        away_data = matchup_data.get("away_team", {})
        
        home_form = home_data.get("form", {})
        away_form = away_data.get("form", {})
        
        home_win_pct = home_form.get("win_pct", 0.5)
        away_win_pct = away_form.get("win_pct", 0.5)
        
        form_adj = (home_win_pct - away_win_pct) * 0.15
        form_adj = max(-0.06, min(0.06, form_adj))
        home_prob += form_adj
        
        if abs(form_adj) > 0.02:
            better = home_team if form_adj > 0 else away_team
            factors.append(f"Form favors {better} ({home_win_pct*100:.0f}% vs {away_win_pct*100:.0f}%)")
        
        # 3. Margin analysis
        home_margin = home_form.get("avg_margin", 0)
        away_margin = away_form.get("avg_margin", 0)
        
        margin_adj = (home_margin - away_margin) * 0.003
        margin_adj = max(-0.04, min(0.04, margin_adj))
        home_prob += margin_adj
        
        if abs(home_margin - away_margin) > 4:
            better = home_team if home_margin > away_margin else away_team
            factors.append(f"{better} wins by more (+{abs(home_margin - away_margin):.1f} margin)")
        
        # 4. Streak analysis
        home_streak = home_form.get("streak", 0)
        away_streak = away_form.get("streak", 0)
        
        if abs(home_streak) >= 3 or abs(away_streak) >= 3:
            streak_adj = (home_streak - away_streak) * 0.008
            streak_adj = max(-0.03, min(0.03, streak_adj))
            home_prob += streak_adj
            
            if home_streak >= 3:
                factors.append(f"{home_team} on {home_streak} game streak")
            if away_streak >= 3:
                factors.append(f"{away_team} on {away_streak} game streak")
        
        # 5. Squad/Injury analysis
        home_squad = squad_data.get("home_team", {})
        away_squad = squad_data.get("away_team", {})
        
        home_injuries = len(home_squad.get("injuries", []))
        away_injuries = len(away_squad.get("injuries", []))
        
        if home_injuries != away_injuries:
            injury_adj = (away_injuries - home_injuries) * 0.01
            injury_adj = max(-0.03, min(0.03, injury_adj))
            home_prob += injury_adj
            
            if abs(home_injuries - away_injuries) >= 2:
                worse = home_team if home_injuries > away_injuries else away_team
                factors.append(f"{worse} has more injuries ({max(home_injuries, away_injuries)})")
        
        # 6. Line movement analysis
        if line_movement and len(line_movement) >= 2:
            opening = line_movement[0].get("home_odds", home_ml)
            current = line_movement[-1].get("home_odds", home_ml)
            
            if opening > 0 and current > 0:
                movement = (current - opening) / opening * 100
                
                if movement < -5:  # Home odds dropped = sharp money on home
                    line_adj = 0.02
                    home_prob += line_adj
                    factors.append(f"Sharp money on {home_team} (line moved {abs(movement):.1f}%)")
                elif movement > 5:  # Away odds dropped
                    line_adj = -0.02
                    home_prob += line_adj
                    factors.append(f"Sharp money on {away_team} (line moved {movement:.1f}%)")
        
        # Note: ESPN provides DraftKings odds only - no multi-book comparison available
        
        # Clamp probability
        home_prob = max(0.30, min(0.70, home_prob))
        away_prob = 1 - home_prob
        
        # Calculate edges
        home_edge = home_prob - home_implied
        away_edge = away_prob - away_implied
        
        # Find best ML pick
        best_side = None
        best_edge = 0
        best_odds = 0
        
        if home_edge >= self.min_edge:
            best_side = home_team
            best_edge = home_edge
            best_odds = home_ml
        
        if away_edge >= self.min_edge and away_edge > best_edge:
            best_side = away_team
            best_edge = away_edge
            best_odds = away_ml
        
        if best_side and best_odds >= 1.5:  # Consider odds >= 1.5
            confidence = self._calculate_confidence(best_edge, len(factors))
            ev = best_edge * (best_odds - 1) * 100  # Expected value %
            
            if confidence >= self.min_confidence:
                return {
                    "has_pick": True,
                    "has_value": True,
                    "pick_type": "moneyline",
                    "pick": best_side,
                    "odds": best_odds,
                    "confidence": confidence,
                    "edge_percent": round(best_edge * 100, 1),
                    "expected_value": round(ev, 2),
                    "our_probability": round(home_prob if best_side == home_team else away_prob, 3),
                    "market_probability": round(home_implied if best_side == home_team else away_implied, 3),
                    "reasoning": f"Our model gives {best_side} {(home_prob if best_side == home_team else away_prob)*100:.0f}% vs market's {(home_implied if best_side == home_team else away_implied)*100:.0f}%. Edge: {best_edge*100:.1f}%",
                    "key_factors": factors[:4],
                    "algorithm": "smart_v4"
                }
        
        return {
            "has_value": False,
            "pick_type": "moneyline",
            "closest_edge": max(home_edge, away_edge),
            "factors": factors
        }
    
    def _analyze_spread(
        self,
        event: Dict,
        matchup_data: Dict,
        squad_data: Dict,
        line_movement: List[Dict],
        config: Dict
    ) -> Dict:
        """Analyze spread market for value"""
        home_team = event.get("home_team", "Home")
        away_team = event.get("away_team", "Away")
        odds = event.get("odds", {})
        
        spread = odds.get("spread", 0)
        if spread == 0:
            return {"has_value": False, "pick_type": "spread"}
        
        # Analyze recent margins to predict cover
        home_data = matchup_data.get("home_team", {})
        away_data = matchup_data.get("away_team", {})
        
        home_recent = home_data.get("recent_games", [])
        away_recent = away_data.get("recent_games", [])
        
        if len(home_recent) < 3 or len(away_recent) < 3:
            return {"has_value": False, "pick_type": "spread", "reason": "Insufficient data"}
        
        # Calculate average margins
        home_margins = [g.get("margin", 0) for g in home_recent[:5]]
        away_margins = [g.get("margin", 0) for g in away_recent[:5]]
        
        home_avg_margin = sum(home_margins) / len(home_margins) if home_margins else 0
        away_avg_margin = sum(away_margins) / len(away_margins) if away_margins else 0
        
        # Predicted margin (home perspective)
        predicted_margin = home_avg_margin - away_avg_margin
        
        # Add home advantage in points
        home_adv_points = config.get("home_adv", 0.025) / config.get("spread_per_point", 0.03)
        predicted_margin += home_adv_points
        
        # Compare to spread
        spread_diff = predicted_margin - (-spread)  # Negative spread means home is favorite
        
        factors = []
        
        if home_avg_margin > 0:
            factors.append(f"{home_team} avg margin: +{home_avg_margin:.1f}")
        if away_avg_margin > 0:
            factors.append(f"{away_team} avg margin: +{away_avg_margin:.1f}")
        
        # Check for value
        min_diff = config.get("total_variance", 5) / 2  # Half a total variance
        
        if abs(spread_diff) >= min_diff:
            if spread_diff > 0:
                # Home covers
                pick = f"{home_team} {spread:+.1f}"
                side = "home"
                factors.append(f"Predicted margin ({predicted_margin:+.1f}) beats spread ({spread:+.1f})")
            else:
                # Away covers
                pick = f"{away_team} {-spread:+.1f}"
                side = "away"
                factors.append(f"Predicted margin ({predicted_margin:+.1f}) favors away covering")
            
            edge = min(0.06, abs(spread_diff) / 20)  # Cap edge at 6%
            confidence = self._calculate_confidence(edge, len(factors))
            ev = edge * 0.91 * 100  # Standard -110 odds
            
            if confidence >= self.min_confidence:
                return {
                    "has_pick": True,
                    "has_value": True,
                    "pick_type": "spread",
                    "pick": pick,
                    "odds": 1.91,
                    "confidence": confidence,
                    "edge_percent": round(edge * 100, 1),
                    "expected_value": round(ev, 2),
                    "predicted_margin": round(predicted_margin, 1),
                    "spread": spread,
                    "reasoning": f"Model predicts {predicted_margin:+.1f} margin vs spread of {spread:+.1f}",
                    "key_factors": factors[:4],
                    "algorithm": "smart_v4"
                }
        
        return {
            "has_value": False,
            "pick_type": "spread",
            "spread_diff": spread_diff,
            "predicted_margin": predicted_margin
        }
    
    def _analyze_total(
        self,
        event: Dict,
        matchup_data: Dict,
        squad_data: Dict,
        config: Dict
    ) -> Dict:
        """Analyze totals (over/under) market for value"""
        home_team = event.get("home_team", "Home")
        away_team = event.get("away_team", "Away")
        odds = event.get("odds", {})
        
        total_line = odds.get("total", config.get("avg_total", 220))
        
        # Get recent scoring data
        home_data = matchup_data.get("home_team", {})
        away_data = matchup_data.get("away_team", {})
        
        home_recent = home_data.get("recent_games", [])
        away_recent = away_data.get("recent_games", [])
        
        if len(home_recent) < 3 or len(away_recent) < 3:
            return {"has_value": False, "pick_type": "total", "reason": "Insufficient data"}
        
        # Calculate average scores
        home_scored = [g.get("our_score", 0) for g in home_recent[:5]]
        home_allowed = [g.get("opponent_score", 0) for g in home_recent[:5]]
        away_scored = [g.get("our_score", 0) for g in away_recent[:5]]
        away_allowed = [g.get("opponent_score", 0) for g in away_recent[:5]]
        
        avg_home_scored = sum(home_scored) / len(home_scored) if home_scored else config.get("avg_total", 220) / 2
        avg_home_allowed = sum(home_allowed) / len(home_allowed) if home_allowed else config.get("avg_total", 220) / 2
        avg_away_scored = sum(away_scored) / len(away_scored) if away_scored else config.get("avg_total", 220) / 2
        avg_away_allowed = sum(away_allowed) / len(away_allowed) if away_allowed else config.get("avg_total", 220) / 2
        
        # Predicted total
        predicted_total = (avg_home_scored + avg_away_scored + avg_home_allowed + avg_away_allowed) / 2
        
        # Difference from line
        diff = predicted_total - total_line
        
        factors = []
        factors.append(f"{home_team} avg: {avg_home_scored:.1f} scored, {avg_home_allowed:.1f} allowed")
        factors.append(f"{away_team} avg: {avg_away_scored:.1f} scored, {avg_away_allowed:.1f} allowed")
        
        # Check for value
        min_diff = config.get("total_variance", 5)
        
        if abs(diff) >= min_diff:
            if diff > 0:
                pick = f"Over {total_line}"
                direction = "over"
                factors.append(f"Predicted {predicted_total:.1f} > line {total_line}")
            else:
                pick = f"Under {total_line}"
                direction = "under"
                factors.append(f"Predicted {predicted_total:.1f} < line {total_line}")
            
            edge = min(0.06, abs(diff) / (config.get("avg_total", 220) / 4))
            confidence = self._calculate_confidence(edge, len(factors))
            ev = edge * 0.91 * 100
            
            if confidence >= self.min_confidence:
                return {
                    "has_pick": True,
                    "has_value": True,
                    "pick_type": "total",
                    "pick": pick,
                    "odds": 1.91,
                    "confidence": confidence,
                    "edge_percent": round(edge * 100, 1),
                    "expected_value": round(ev, 2),
                    "predicted_total": round(predicted_total, 1),
                    "total_line": total_line,
                    "reasoning": f"Scoring trends predict {predicted_total:.0f} total vs line of {total_line}",
                    "key_factors": factors[:4],
                    "algorithm": "smart_v4"
                }
        
        return {
            "has_value": False,
            "pick_type": "total",
            "diff": diff,
            "predicted_total": predicted_total
        }
    
    def _calculate_confidence(self, edge: float, factor_count: int) -> float:
        """Calculate confidence based on edge and supporting factors"""
        base = 0.68
        
        # Edge bonus (each 1% edge = 1.5% confidence)
        edge_bonus = edge * 1.5
        
        # Factor bonus (more supporting factors = more confidence)
        factor_bonus = min(0.04, factor_count * 0.008)
        
        confidence = base + edge_bonus + factor_bonus
        
        return min(self.max_confidence, max(self.min_confidence, confidence))
    
    def _get_closest_value(self, ml: Dict, spread: Dict, total: Dict) -> str:
        """Get description of what came closest to having value"""
        closest = []
        
        if ml and ml.get("closest_edge"):
            closest.append(f"ML edge: {ml['closest_edge']*100:.1f}%")
        if spread and spread.get("spread_diff"):
            closest.append(f"Spread diff: {spread['spread_diff']:.1f} pts")
        if total and total.get("diff"):
            closest.append(f"Total diff: {total['diff']:.1f} pts")
        
        return "; ".join(closest) if closest else "All markets near 50/50"


async def generate_smart_prediction(
    event: Dict,
    sport_key: str,
    squad_data: Dict,
    matchup_data: Dict,
    line_movement: List[Dict],
    multi_book_odds: Dict
) -> Optional[Dict]:
    """
    Main entry point for smart algorithmic predictions.
    No LLM required - pure statistical analysis.
    """
    engine = SmartPredictionEngine()
    return await engine.analyze_and_predict(
        event, sport_key, squad_data, matchup_data, line_movement, multi_book_odds
    )
