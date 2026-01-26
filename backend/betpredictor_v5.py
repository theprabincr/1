"""
BetPredictor V5 - Comprehensive Analysis Engine
Combines deep line movement analysis with squad, H2H, venue, and matchup data.

Key Features:
1. Complete line movement study from opening to 1 hour before game
2. Sharp money detection and reverse line movement identification
3. Squad analysis with injury impact
4. Head-to-head historical analysis
5. Venue and travel factor consideration
6. Recent form and momentum analysis
7. Reasonable confidence calculation based on factor alignment

Philosophy:
- Only recommend when multiple factors align
- Confidence is calculated based on actual analysis depth
- NO PICK is the default - most markets are efficient
- 70%+ confidence only when 4+ major factors agree
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import math

from line_movement_analyzer import LineMovementAnalyzer, analyze_line_movement

logger = logging.getLogger(__name__)


# Sport-specific configurations
SPORT_CONFIG = {
    "basketball_nba": {
        "avg_total": 225,
        "home_adv_pct": 2.5,  # Home team gets 2.5% boost
        "key_player_impact": 0.03,  # 3% per key player out
        "h2h_weight": 0.15,  # How much to weight H2H
        "form_weight": 0.20,  # How much to weight recent form
        "line_movement_weight": 0.25,  # How much to weight line movement
    },
    "americanfootball_nfl": {
        "avg_total": 45,
        "home_adv_pct": 2.2,
        "key_player_impact": 0.04,
        "h2h_weight": 0.10,
        "form_weight": 0.25,
        "line_movement_weight": 0.25,
    },
    "icehockey_nhl": {
        "avg_total": 6,
        "home_adv_pct": 2.2,
        "key_player_impact": 0.025,
        "h2h_weight": 0.12,
        "form_weight": 0.22,
        "line_movement_weight": 0.25,
    },
    "soccer_epl": {
        "avg_total": 2.5,
        "home_adv_pct": 3.5,
        "key_player_impact": 0.03,
        "h2h_weight": 0.15,
        "form_weight": 0.18,
        "line_movement_weight": 0.25,
    },
}


class BetPredictorV5:
    """
    Comprehensive betting prediction engine that analyzes
    all available data to make informed predictions.
    """
    
    def __init__(self):
        self.line_analyzer = LineMovementAnalyzer()
        self.min_confidence = 0.70
        self.min_edge = 0.04  # 4% minimum edge
        self.min_factors = 4  # At least 4 factors must align
    
    async def analyze_and_predict(
        self,
        event: Dict,
        sport_key: str,
        squad_data: Dict,
        matchup_data: Dict,
        line_movement_history: List[Dict],
        opening_odds: Dict,
        current_odds: Dict
    ) -> Dict:
        """
        Comprehensive analysis combining all factors.
        
        Returns:
        - prediction: The recommended pick (or None if no value)
        - confidence: 0-100% confidence level
        - reasoning: Detailed explanation of the analysis
        - factors: List of supporting/opposing factors
        """
        home_team = event.get("home_team", "Home")
        away_team = event.get("away_team", "Away")
        commence_time = event.get("commence_time", "")
        
        logger.info(f"🎯 BetPredictor V5 Analysis: {home_team} vs {away_team}")
        
        config = SPORT_CONFIG.get(sport_key, SPORT_CONFIG["basketball_nba"])
        
        # Initialize analysis result
        analysis = {
            "home_team": home_team,
            "away_team": away_team,
            "home_score": 50.0,  # Start at 50/50
            "factors_home": [],
            "factors_away": [],
            "factor_count": 0,
            "warnings": [],
            "insights": []
        }
        
        # 1. LINE MOVEMENT ANALYSIS (Most important - 25% weight)
        logger.info("📊 Analyzing line movement...")
        line_analysis = await analyze_line_movement(
            line_movement_history,
            opening_odds,
            current_odds or {},
            commence_time,
            home_team,
            away_team
        )
        
        line_adjustment = self._process_line_movement(line_analysis, analysis, config)
        analysis["line_analysis"] = line_analysis
        
        # 2. FORM ANALYSIS (20% weight)
        logger.info("📈 Analyzing recent form...")
        form_adjustment = self._analyze_form(matchup_data, analysis, config)
        
        # 3. SQUAD ANALYSIS with injuries (15% weight)
        logger.info("👥 Analyzing squads and injuries...")
        squad_adjustment = self._analyze_squads(squad_data, analysis, config)
        
        # 4. H2H ANALYSIS (15% weight)
        logger.info("🔄 Analyzing head-to-head history...")
        h2h_adjustment = self._analyze_h2h(matchup_data, analysis, config)
        
        # 5. VENUE/HOME ADVANTAGE (10% weight)
        logger.info("🏟️ Analyzing venue factors...")
        venue_adjustment = self._analyze_venue(event, matchup_data, analysis, config)
        
        # 6. ODDS VALUE ANALYSIS
        logger.info("💰 Analyzing odds value...")
        odds_analysis = self._analyze_odds_value(event, analysis)
        
        # Calculate final probability
        total_adjustment = (
            line_adjustment * config["line_movement_weight"] +
            form_adjustment * config["form_weight"] +
            squad_adjustment * 0.15 +
            h2h_adjustment * config["h2h_weight"] +
            venue_adjustment * 0.10
        )
        
        # Apply adjustments (normalized)
        analysis["home_score"] += total_adjustment * 100
        analysis["home_score"] = max(30, min(70, analysis["home_score"]))
        
        home_prob = analysis["home_score"] / 100
        away_prob = 1 - home_prob
        
        # Generate prediction
        prediction = self._generate_prediction(
            analysis,
            home_prob,
            away_prob,
            odds_analysis,
            line_analysis,
            home_team,
            away_team,
            sport_key
        )
        
        return prediction
    
    def _process_line_movement(
        self,
        line_analysis: Dict,
        analysis: Dict,
        config: Dict
    ) -> float:
        """
        Process line movement analysis and return adjustment.
        Returns: adjustment factor (-0.15 to +0.15)
        """
        adjustment = 0
        home_team = analysis["home_team"]
        away_team = analysis["away_team"]
        
        # Value side from line movement
        value_side = line_analysis.get("value_side")
        value_confidence = line_analysis.get("value_confidence", 0.5)
        
        if value_side:
            base_adj = (value_confidence - 0.5) * 0.5  # Convert to adjustment
            
            if value_side == home_team:
                adjustment += base_adj
                analysis["factors_home"].append({
                    "factor": "Line Movement",
                    "strength": "strong" if base_adj > 0.05 else "moderate",
                    "detail": line_analysis.get("summary", "Market favoring home")
                })
            else:
                adjustment -= base_adj
                analysis["factors_away"].append({
                    "factor": "Line Movement",
                    "strength": "strong" if base_adj > 0.05 else "moderate",
                    "detail": line_analysis.get("summary", "Market favoring away")
                })
            
            analysis["factor_count"] += 1
        
        # Sharp money bonus
        if line_analysis.get("sharp_money_side"):
            sharp_side = line_analysis["sharp_money_side"]
            if sharp_side == home_team:
                adjustment += 0.03
                analysis["factors_home"].append({
                    "factor": "Sharp Money",
                    "strength": "strong",
                    "detail": f"Professional bettors backing {home_team}"
                })
            else:
                adjustment -= 0.03
                analysis["factors_away"].append({
                    "factor": "Sharp Money",
                    "strength": "strong",
                    "detail": f"Professional bettors backing {away_team}"
                })
            analysis["factor_count"] += 1
        
        # RLM bonus
        if line_analysis.get("reverse_line_movement"):
            analysis["insights"].append(
                "⚠️ Reverse Line Movement: Line moving opposite to public betting"
            )
            # RLM side gets a boost
            if line_analysis.get("sharp_money_side") == home_team:
                adjustment += 0.02
            else:
                adjustment -= 0.02
            analysis["factor_count"] += 1
        
        # Steam moves
        if line_analysis.get("steam_moves"):
            for steam in line_analysis["steam_moves"][:2]:
                steam_side = steam.get("side")
                if steam_side == home_team:
                    adjustment += 0.015
                else:
                    adjustment -= 0.015
                analysis["insights"].append(
                    f"Steam move: Rapid price change toward {steam_side}"
                )
        
        # Add key insights
        for insight in line_analysis.get("key_insights", [])[:3]:
            analysis["insights"].append(insight)
        
        return max(-0.15, min(0.15, adjustment))
    
    def _analyze_form(
        self,
        matchup_data: Dict,
        analysis: Dict,
        config: Dict
    ) -> float:
        """
        Analyze recent form and momentum.
        Returns: adjustment factor (-0.15 to +0.15)
        """
        adjustment = 0
        home_team = analysis["home_team"]
        away_team = analysis["away_team"]
        
        home_data = matchup_data.get("home_team", {})
        away_data = matchup_data.get("away_team", {})
        
        home_form = home_data.get("form", {})
        away_form = away_data.get("form", {})
        
        # Win percentage comparison
        home_win_pct = home_form.get("win_pct", 0.5)
        away_win_pct = away_form.get("win_pct", 0.5)
        
        win_pct_diff = home_win_pct - away_win_pct
        
        if abs(win_pct_diff) > 0.15:
            adj = win_pct_diff * 0.3
            adjustment += max(-0.05, min(0.05, adj))
            
            better = home_team if win_pct_diff > 0 else away_team
            better_pct = home_win_pct if win_pct_diff > 0 else away_win_pct
            worse_pct = away_win_pct if win_pct_diff > 0 else home_win_pct
            
            factors_list = analysis["factors_home"] if win_pct_diff > 0 else analysis["factors_away"]
            factors_list.append({
                "factor": "Win Rate",
                "strength": "strong" if abs(win_pct_diff) > 0.25 else "moderate",
                "detail": f"{better} winning {better_pct*100:.0f}% vs {worse_pct*100:.0f}%"
            })
            analysis["factor_count"] += 1
        
        # Average margin comparison
        home_margin = home_form.get("avg_margin", 0)
        away_margin = away_form.get("avg_margin", 0)
        
        margin_diff = home_margin - away_margin
        
        if abs(margin_diff) > 5:
            adj = margin_diff * 0.01
            adjustment += max(-0.04, min(0.04, adj))
            
            better = home_team if margin_diff > 0 else away_team
            better_margin = home_margin if margin_diff > 0 else away_margin
            
            factors_list = analysis["factors_home"] if margin_diff > 0 else analysis["factors_away"]
            factors_list.append({
                "factor": "Scoring Margin",
                "strength": "moderate",
                "detail": f"{better} winning by avg +{abs(better_margin):.1f} points"
            })
            analysis["factor_count"] += 1
        
        # Streak analysis
        home_streak = home_form.get("streak", 0)
        away_streak = away_form.get("streak", 0)
        
        if abs(home_streak) >= 3:
            streak_adj = home_streak * 0.01
            adjustment += max(-0.03, min(0.03, streak_adj))
            
            if home_streak > 0:
                analysis["factors_home"].append({
                    "factor": "Hot Streak",
                    "strength": "moderate",
                    "detail": f"{home_team} on {home_streak}-game win streak"
                })
            else:
                analysis["factors_away"].append({
                    "factor": "Cold Streak",
                    "strength": "moderate",
                    "detail": f"{home_team} on {abs(home_streak)}-game losing streak"
                })
            analysis["factor_count"] += 1
        
        if abs(away_streak) >= 3:
            streak_adj = -away_streak * 0.01
            adjustment += max(-0.03, min(0.03, streak_adj))
            
            if away_streak > 0:
                analysis["factors_away"].append({
                    "factor": "Hot Streak",
                    "strength": "moderate",
                    "detail": f"{away_team} on {away_streak}-game win streak"
                })
            else:
                analysis["factors_home"].append({
                    "factor": "Cold Streak",
                    "strength": "moderate",
                    "detail": f"{away_team} on {abs(away_streak)}-game losing streak"
                })
            analysis["factor_count"] += 1
        
        return max(-0.15, min(0.15, adjustment))
    
    def _analyze_squads(
        self,
        squad_data: Dict,
        analysis: Dict,
        config: Dict
    ) -> float:
        """
        Analyze squad strength and injuries.
        Returns: adjustment factor (-0.10 to +0.10)
        """
        adjustment = 0
        home_team = analysis["home_team"]
        away_team = analysis["away_team"]
        
        home_squad = squad_data.get("home_team", {})
        away_squad = squad_data.get("away_team", {})
        
        # Injury analysis
        home_injuries = home_squad.get("injuries", [])
        away_injuries = away_squad.get("injuries", [])
        
        # Count significant injuries (starters/key players)
        home_key_out = sum(1 for inj in home_injuries 
                         if inj.get("status", "").lower() in ["out", "doubtful"])
        away_key_out = sum(1 for inj in away_injuries 
                         if inj.get("status", "").lower() in ["out", "doubtful"])
        
        injury_diff = away_key_out - home_key_out
        
        if abs(injury_diff) >= 2:
            key_player_impact = config.get("key_player_impact", 0.03)
            adj = injury_diff * key_player_impact
            adjustment += max(-0.06, min(0.06, adj))
            
            if injury_diff > 0:
                analysis["factors_home"].append({
                    "factor": "Injury Advantage",
                    "strength": "strong" if injury_diff >= 3 else "moderate",
                    "detail": f"{away_team} missing {away_key_out} key players vs {home_key_out} for {home_team}"
                })
            else:
                analysis["factors_away"].append({
                    "factor": "Injury Advantage",
                    "strength": "strong" if abs(injury_diff) >= 3 else "moderate",
                    "detail": f"{home_team} missing {home_key_out} key players vs {away_key_out} for {away_team}"
                })
            
            analysis["factor_count"] += 1
        
        # Add injury warnings
        if home_key_out >= 2:
            analysis["warnings"].append(f"⚠️ {home_team} has {home_key_out} key players out/doubtful")
        if away_key_out >= 2:
            analysis["warnings"].append(f"⚠️ {away_team} has {away_key_out} key players out/doubtful")
        
        return max(-0.10, min(0.10, adjustment))
    
    def _analyze_h2h(
        self,
        matchup_data: Dict,
        analysis: Dict,
        config: Dict
    ) -> float:
        """
        Analyze head-to-head history.
        Returns: adjustment factor (-0.08 to +0.08)
        """
        adjustment = 0
        home_team = analysis["home_team"]
        away_team = analysis["away_team"]
        
        h2h = matchup_data.get("h2h", {})
        
        if not h2h:
            return 0
        
        home_h2h_wins = h2h.get("home_wins", 0)
        away_h2h_wins = h2h.get("away_wins", 0)
        total_h2h = home_h2h_wins + away_h2h_wins
        
        if total_h2h >= 3:
            home_h2h_pct = home_h2h_wins / total_h2h if total_h2h > 0 else 0.5
            
            if home_h2h_pct >= 0.65:
                adjustment += 0.03
                analysis["factors_home"].append({
                    "factor": "H2H Dominance",
                    "strength": "moderate",
                    "detail": f"{home_team} won {home_h2h_wins} of last {total_h2h} meetings"
                })
                analysis["factor_count"] += 1
            elif home_h2h_pct <= 0.35:
                adjustment -= 0.03
                analysis["factors_away"].append({
                    "factor": "H2H Dominance",
                    "strength": "moderate",
                    "detail": f"{away_team} won {away_h2h_wins} of last {total_h2h} meetings"
                })
                analysis["factor_count"] += 1
        
        return max(-0.08, min(0.08, adjustment))
    
    def _analyze_venue(
        self,
        event: Dict,
        matchup_data: Dict,
        analysis: Dict,
        config: Dict
    ) -> float:
        """
        Analyze venue factors and home advantage.
        Returns: adjustment factor (0 to +0.05)
        """
        home_team = analysis["home_team"]
        
        # Base home advantage
        home_adv = config.get("home_adv_pct", 2.5) / 100
        
        adjustment = home_adv
        analysis["factors_home"].append({
            "factor": "Home Court",
            "strength": "moderate",
            "detail": f"{home_team} playing at home (+{config.get('home_adv_pct', 2.5)}% baseline)"
        })
        
        # Check home record
        home_data = matchup_data.get("home_team", {})
        home_form = home_data.get("form", {})
        home_record = home_form.get("home_record", {})
        
        if home_record:
            home_wins = home_record.get("wins", 0)
            home_losses = home_record.get("losses", 0)
            
            if home_wins + home_losses >= 5:
                home_win_rate = home_wins / (home_wins + home_losses)
                
                if home_win_rate >= 0.70:
                    adjustment += 0.02
                    analysis["factors_home"].append({
                        "factor": "Home Dominance",
                        "strength": "strong",
                        "detail": f"{home_team} is {home_wins}-{home_losses} at home ({home_win_rate*100:.0f}%)"
                    })
                    analysis["factor_count"] += 1
        
        return max(0, min(0.05, adjustment))
    
    def _analyze_odds_value(
        self,
        event: Dict,
        analysis: Dict
    ) -> Dict:
        """
        Analyze current odds for value.
        """
        odds = event.get("odds", {})
        
        home_ml = odds.get("home_ml_decimal", 1.91)
        away_ml = odds.get("away_ml_decimal", 1.91)
        
        # Calculate implied probabilities
        home_implied = 1 / home_ml if home_ml > 1 else 0.5
        away_implied = 1 / away_ml if away_ml > 1 else 0.5
        
        # Remove vig
        total = home_implied + away_implied
        if total > 0:
            home_implied /= total
            away_implied /= total
        
        return {
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_implied": home_implied,
            "away_implied": away_implied
        }
    
    def _generate_prediction(
        self,
        analysis: Dict,
        home_prob: float,
        away_prob: float,
        odds_analysis: Dict,
        line_analysis: Dict,
        home_team: str,
        away_team: str,
        sport_key: str
    ) -> Dict:
        """
        Generate final prediction with confidence and reasoning.
        """
        # Calculate edges
        home_edge = home_prob - odds_analysis["home_implied"]
        away_edge = away_prob - odds_analysis["away_implied"]
        
        # Determine best pick
        best_side = None
        best_edge = 0
        best_odds = 0
        best_prob = 0
        
        if home_edge >= self.min_edge:
            best_side = home_team
            best_edge = home_edge
            best_odds = odds_analysis["home_ml"]
            best_prob = home_prob
        
        if away_edge > best_edge and away_edge >= self.min_edge:
            best_side = away_team
            best_edge = away_edge
            best_odds = odds_analysis["away_ml"]
            best_prob = away_prob
        
        # Calculate confidence based on factors
        factor_count = analysis["factor_count"]
        base_confidence = 0.60
        
        # Factor bonus
        factor_bonus = min(0.15, factor_count * 0.025)
        
        # Edge bonus
        edge_bonus = min(0.10, best_edge * 1.5)
        
        # Line movement bonus
        line_conf_adj = line_analysis.get("confidence_adjustment", 0)
        
        confidence = base_confidence + factor_bonus + edge_bonus + line_conf_adj
        confidence = max(0.50, min(0.85, confidence))
        
        # Build reasoning
        reasoning_parts = []
        
        # Add line movement insights
        if line_analysis.get("summary"):
            reasoning_parts.append(f"📊 Line Movement: {line_analysis['summary']}")
        
        # Add key factors
        if best_side == home_team and analysis["factors_home"]:
            for f in analysis["factors_home"][:3]:
                reasoning_parts.append(f"✓ {f['factor']}: {f['detail']}")
        elif best_side == away_team and analysis["factors_away"]:
            for f in analysis["factors_away"][:3]:
                reasoning_parts.append(f"✓ {f['factor']}: {f['detail']}")
        
        # Add warnings
        for warning in analysis["warnings"][:2]:
            reasoning_parts.append(warning)
        
        # Decision
        if best_side and confidence >= self.min_confidence and factor_count >= self.min_factors:
            ev = best_edge * (best_odds - 1) * 100
            
            return {
                "has_pick": True,
                "pick": best_side,
                "pick_type": "moneyline",
                "odds": best_odds,
                "confidence": round(confidence * 100, 1),
                "edge_percent": round(best_edge * 100, 1),
                "expected_value": round(ev, 2),
                "our_probability": round(best_prob * 100, 1),
                "market_probability": round(odds_analysis["home_implied" if best_side == home_team else "away_implied"] * 100, 1),
                "factor_count": factor_count,
                "reasoning": "\n".join(reasoning_parts),
                "key_factors": [f["factor"] for f in (analysis["factors_home"] if best_side == home_team else analysis["factors_away"])],
                "line_analysis_summary": line_analysis.get("summary"),
                "insights": analysis["insights"],
                "warnings": analysis["warnings"],
                "algorithm": "betpredictor_v5"
            }
        else:
            # No pick - explain why
            reasons = []
            if factor_count < self.min_factors:
                reasons.append(f"Only {factor_count}/{self.min_factors} required factors aligned")
            if confidence < self.min_confidence:
                reasons.append(f"Confidence {confidence*100:.0f}% below {self.min_confidence*100:.0f}% threshold")
            if best_edge < self.min_edge:
                reasons.append(f"Edge {best_edge*100:.1f}% below {self.min_edge*100:.0f}% minimum")
            
            return {
                "has_pick": False,
                "reasoning": "No pick recommended: " + "; ".join(reasons),
                "closest_confidence": round(confidence * 100, 1),
                "closest_edge": round(max(home_edge, away_edge) * 100, 1),
                "factor_count": factor_count,
                "home_probability": round(home_prob * 100, 1),
                "away_probability": round(away_prob * 100, 1),
                "line_analysis_summary": line_analysis.get("summary"),
                "insights": analysis["insights"],
                "algorithm": "betpredictor_v5"
            }


# Main entry point
async def generate_v5_prediction(
    event: Dict,
    sport_key: str,
    squad_data: Dict,
    matchup_data: Dict,
    line_movement_history: List[Dict],
    opening_odds: Dict,
    current_odds: Dict
) -> Dict:
    """
    Generate a prediction using BetPredictor V5.
    """
    engine = BetPredictorV5()
    return await engine.analyze_and_predict(
        event,
        sport_key,
        squad_data,
        matchup_data,
        line_movement_history,
        opening_odds,
        current_odds
    )
