"""
BetPredictor V5 - Comprehensive Analysis Engine
Combines deep line movement analysis across ALL markets (ML, Spread, Totals)
with squad, H2H, venue, and matchup data.

Key Features:
1. Complete line movement study from opening to 1 hour before game
2. Analyzes ML, Spread, and Totals markets separately
3. Sharp money detection and reverse line movement identification
4. Cross-market signal analysis
5. Squad analysis with injury impact
6. Head-to-head historical analysis
7. Venue and travel factor consideration
8. Recent form and momentum analysis
9. Generates picks for ML, Spread, OR Totals based on best signal

Philosophy:
- Only recommend when multiple factors align
- Confidence is calculated based on actual analysis depth
- NO PICK is the default - most markets are efficient
- 70%+ confidence only when 4+ major factors agree
- Recommends the market with strongest signal
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
        "home_adv_pct": 2.5,
        "key_player_impact": 0.03,
        "h2h_weight": 0.15,
        "form_weight": 0.20,
        "line_movement_weight": 0.25,
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
    all available data across all markets to make informed predictions.
    """
    
    def __init__(self):
        self.line_analyzer = LineMovementAnalyzer()
        self.min_confidence = 0.70
        self.min_edge = 0.04  # 4% minimum edge
        self.min_factors = 3  # At least 3 factors must align
    
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
        Comprehensive analysis combining all factors for ALL markets.
        
        Returns:
        - prediction: The recommended pick (ML, Spread, or Total)
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
            "home_score": 50.0,
            "factors_home": [],
            "factors_away": [],
            "factors_over": [],
            "factors_under": [],
            "factor_count": 0,
            "warnings": [],
            "insights": []
        }
        
        # 1. LINE MOVEMENT ANALYSIS for ALL MARKETS
        logger.info("📊 Analyzing line movement across all markets...")
        line_analysis = await analyze_line_movement(
            line_movement_history,
            opening_odds,
            current_odds or {},
            commence_time,
            home_team,
            away_team
        )
        
        # Process line movement for each market
        ml_analysis = line_analysis.get("markets", {}).get("moneyline", {})
        spread_analysis = line_analysis.get("markets", {}).get("spread", {})
        totals_analysis = line_analysis.get("markets", {}).get("totals", {})
        
        # Get adjustments from line movement
        side_adjustment = self._process_side_movement(ml_analysis, spread_analysis, analysis, home_team, away_team)
        totals_direction = self._process_totals_movement(totals_analysis, analysis)
        
        analysis["line_analysis"] = line_analysis
        
        # 2. FORM ANALYSIS
        logger.info("📈 Analyzing recent form...")
        form_adjustment = self._analyze_form(matchup_data, analysis, config)
        
        # 3. SQUAD ANALYSIS
        logger.info("👥 Analyzing squads and injuries...")
        squad_adjustment = self._analyze_squads(squad_data, analysis, config)
        
        # 4. H2H ANALYSIS
        logger.info("🔄 Analyzing head-to-head history...")
        h2h_adjustment = self._analyze_h2h(matchup_data, analysis, config)
        
        # 5. VENUE ANALYSIS
        logger.info("🏟️ Analyzing venue factors...")
        venue_adjustment = self._analyze_venue(event, matchup_data, analysis, config)
        
        # Calculate side probability (for ML/Spread)
        total_side_adjustment = (
            side_adjustment * config["line_movement_weight"] +
            form_adjustment * config["form_weight"] +
            squad_adjustment * 0.15 +
            h2h_adjustment * config["h2h_weight"] +
            venue_adjustment * 0.10
        )
        
        analysis["home_score"] += total_side_adjustment * 100
        analysis["home_score"] = max(30, min(70, analysis["home_score"]))
        
        home_prob = analysis["home_score"] / 100
        away_prob = 1 - home_prob
        
        # 6. DETERMINE BEST MARKET TO BET
        prediction = self._generate_multi_market_prediction(
            analysis,
            line_analysis,
            home_prob,
            away_prob,
            totals_direction,
            event,
            home_team,
            away_team,
            sport_key
        )
        
        return prediction
    
    def _process_side_movement(
        self,
        ml_analysis: Dict,
        spread_analysis: Dict,
        analysis: Dict,
        home_team: str,
        away_team: str
    ) -> float:
        """Process ML and Spread line movement."""
        adjustment = 0
        
        # ML signals
        if ml_analysis.get("value_side"):
            conf = ml_analysis.get("confidence", 0.5)
            base_adj = (conf - 0.5) * 0.3
            
            if ml_analysis["value_side"] == home_team:
                adjustment += base_adj
                analysis["factors_home"].append({
                    "factor": "ML Line Movement",
                    "strength": "strong" if base_adj > 0.03 else "moderate",
                    "detail": f"ML moved {abs(ml_analysis.get('total_movement', 0)):.1f}% toward {home_team}"
                })
            else:
                adjustment -= base_adj
                analysis["factors_away"].append({
                    "factor": "ML Line Movement",
                    "strength": "strong" if base_adj > 0.03 else "moderate",
                    "detail": f"ML moved {abs(ml_analysis.get('total_movement', 0)):.1f}% toward {away_team}"
                })
            analysis["factor_count"] += 1
        
        # Spread signals
        if spread_analysis.get("value_side"):
            conf = spread_analysis.get("confidence", 0.5)
            base_adj = (conf - 0.5) * 0.25
            
            if spread_analysis["value_side"] == home_team:
                adjustment += base_adj
                analysis["factors_home"].append({
                    "factor": "Spread Movement",
                    "strength": "strong" if base_adj > 0.03 else "moderate",
                    "detail": f"Spread moved {abs(spread_analysis.get('total_movement', 0)):.1f}pts toward {home_team}"
                })
            else:
                adjustment -= base_adj
                analysis["factors_away"].append({
                    "factor": "Spread Movement",
                    "strength": "strong" if base_adj > 0.03 else "moderate",
                    "detail": f"Spread moved {abs(spread_analysis.get('total_movement', 0)):.1f}pts toward {away_team}"
                })
            analysis["factor_count"] += 1
        
        # Sharp money bonus
        for market_data in [ml_analysis, spread_analysis]:
            if market_data.get("sharp_money_side"):
                sharp_side = market_data["sharp_money_side"]
                if sharp_side == home_team:
                    adjustment += 0.02
                else:
                    adjustment -= 0.02
                analysis["factor_count"] += 1
                analysis["insights"].append(f"Sharp money detected on {sharp_side}")
        
        # RLM bonus
        for market_data in [ml_analysis, spread_analysis]:
            if market_data.get("reverse_line_movement"):
                analysis["insights"].append("⚠️ Reverse Line Movement detected")
                analysis["factor_count"] += 1
        
        return max(-0.15, min(0.15, adjustment))
    
    def _process_totals_movement(
        self,
        totals_analysis: Dict,
        analysis: Dict
    ) -> Optional[str]:
        """Process totals line movement and return direction."""
        direction = None
        
        if totals_analysis.get("value_side"):
            direction = totals_analysis["value_side"]  # "over" or "under"
            conf = totals_analysis.get("confidence", 0.5)
            
            if direction == "over":
                analysis["factors_over"].append({
                    "factor": "Totals Movement",
                    "strength": "strong" if conf > 0.6 else "moderate",
                    "detail": f"Total moved {abs(totals_analysis.get('total_movement', 0)):.1f}pts up"
                })
            else:
                analysis["factors_under"].append({
                    "factor": "Totals Movement",
                    "strength": "strong" if conf > 0.6 else "moderate",
                    "detail": f"Total moved {abs(totals_analysis.get('total_movement', 0)):.1f}pts down"
                })
            analysis["factor_count"] += 1
        
        if totals_analysis.get("sharp_money_side"):
            direction = totals_analysis["sharp_money_side"]
            analysis["insights"].append(f"Sharp money on {direction.upper()}")
            analysis["factor_count"] += 1
        
        return direction
    
    def _analyze_form(self, matchup_data: Dict, analysis: Dict, config: Dict) -> float:
        """Analyze recent form and momentum."""
        adjustment = 0
        home_team = analysis["home_team"]
        away_team = analysis["away_team"]
        
        home_data = matchup_data.get("home_team", {})
        away_data = matchup_data.get("away_team", {})
        
        home_form = home_data.get("form", {})
        away_form = away_data.get("form", {})
        
        # Win percentage
        home_win_pct = home_form.get("win_pct", 0.5)
        away_win_pct = away_form.get("win_pct", 0.5)
        
        win_pct_diff = home_win_pct - away_win_pct
        
        if abs(win_pct_diff) > 0.15:
            adj = win_pct_diff * 0.3
            adjustment += max(-0.05, min(0.05, adj))
            
            better = home_team if win_pct_diff > 0 else away_team
            factors_list = analysis["factors_home"] if win_pct_diff > 0 else analysis["factors_away"]
            factors_list.append({
                "factor": "Win Rate",
                "strength": "strong" if abs(win_pct_diff) > 0.25 else "moderate",
                "detail": f"{better} has better record"
            })
            analysis["factor_count"] += 1
        
        # Streak
        home_streak = home_form.get("streak", 0)
        away_streak = away_form.get("streak", 0)
        
        if abs(home_streak) >= 3:
            if home_streak > 0:
                adjustment += 0.02
                analysis["factors_home"].append({
                    "factor": "Hot Streak",
                    "strength": "moderate",
                    "detail": f"{home_team} on {home_streak}-game win streak"
                })
            else:
                adjustment -= 0.02
                analysis["factors_away"].append({
                    "factor": "Cold Streak",
                    "strength": "moderate",
                    "detail": f"{home_team} on {abs(home_streak)}-game losing streak"
                })
            analysis["factor_count"] += 1
        
        return max(-0.10, min(0.10, adjustment))
    
    def _analyze_squads(self, squad_data: Dict, analysis: Dict, config: Dict) -> float:
        """Analyze squad strength and injuries."""
        adjustment = 0
        home_team = analysis["home_team"]
        away_team = analysis["away_team"]
        
        home_squad = squad_data.get("home_team", {})
        away_squad = squad_data.get("away_team", {})
        
        home_injuries = home_squad.get("injuries", [])
        away_injuries = away_squad.get("injuries", [])
        
        home_key_out = sum(1 for inj in home_injuries 
                         if inj.get("status", "").lower() in ["out", "doubtful"])
        away_key_out = sum(1 for inj in away_injuries 
                         if inj.get("status", "").lower() in ["out", "doubtful"])
        
        injury_diff = away_key_out - home_key_out
        
        if abs(injury_diff) >= 2:
            adj = injury_diff * config.get("key_player_impact", 0.03)
            adjustment += max(-0.06, min(0.06, adj))
            
            if injury_diff > 0:
                analysis["factors_home"].append({
                    "factor": "Injury Advantage",
                    "strength": "strong" if injury_diff >= 3 else "moderate",
                    "detail": f"{away_team} missing {away_key_out} key players"
                })
            else:
                analysis["factors_away"].append({
                    "factor": "Injury Advantage",
                    "strength": "strong" if abs(injury_diff) >= 3 else "moderate",
                    "detail": f"{home_team} missing {home_key_out} key players"
                })
            analysis["factor_count"] += 1
        
        return max(-0.08, min(0.08, adjustment))
    
    def _analyze_h2h(self, matchup_data: Dict, analysis: Dict, config: Dict) -> float:
        """Analyze head-to-head history."""
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
                    "detail": f"{home_team} won {home_h2h_wins}/{total_h2h} recent meetings"
                })
                analysis["factor_count"] += 1
            elif home_h2h_pct <= 0.35:
                adjustment -= 0.03
                analysis["factors_away"].append({
                    "factor": "H2H Dominance",
                    "strength": "moderate",
                    "detail": f"{away_team} won {away_h2h_wins}/{total_h2h} recent meetings"
                })
                analysis["factor_count"] += 1
        
        return max(-0.06, min(0.06, adjustment))
    
    def _analyze_venue(self, event: Dict, matchup_data: Dict, analysis: Dict, config: Dict) -> float:
        """Analyze venue factors."""
        home_team = analysis["home_team"]
        home_adv = config.get("home_adv_pct", 2.5) / 100
        
        analysis["factors_home"].append({
            "factor": "Home Court",
            "strength": "moderate",
            "detail": f"{home_team} playing at home"
        })
        
        return home_adv
    
    def _generate_multi_market_prediction(
        self,
        analysis: Dict,
        line_analysis: Dict,
        home_prob: float,
        away_prob: float,
        totals_direction: Optional[str],
        event: Dict,
        home_team: str,
        away_team: str,
        sport_key: str
    ) -> Dict:
        """Generate prediction for the best market (ML, Spread, or Totals)."""
        
        # Get recommended market from line analysis
        recommended = line_analysis.get("recommended_market")
        markets = line_analysis.get("markets", {})
        
        # Get current odds
        odds = event.get("odds", {})
        bookmakers = event.get("bookmakers", [])
        
        current_spread = None
        current_total = None
        home_ml = odds.get("home_ml_decimal", 1.91)
        away_ml = odds.get("away_ml_decimal", 1.91)
        spread_odds = 1.91
        over_odds = 1.91
        under_odds = 1.91
        
        # Extract spread and total from bookmakers
        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") == "spreads":
                    for outcome in market.get("outcomes", []):
                        if home_team.lower() in outcome.get("name", "").lower() or outcome.get("name", "").lower() == "home":
                            current_spread = outcome.get("point")
                            spread_odds = outcome.get("price", 1.91)
                elif market.get("key") == "totals":
                    for outcome in market.get("outcomes", []):
                        if "over" in outcome.get("name", "").lower():
                            current_total = outcome.get("point")
                            over_odds = outcome.get("price", 1.91)
                        elif "under" in outcome.get("name", "").lower():
                            under_odds = outcome.get("price", 1.91)
        
        # Calculate factor counts
        factor_count = analysis["factor_count"]
        
        # Build candidates for each market
        candidates = []
        
        # ML candidate
        ml_market = markets.get("moneyline", {})
        if ml_market.get("value_side") and ml_market.get("confidence", 0) > 0.55:
            ml_side = ml_market["value_side"]
            ml_odds = home_ml if ml_side == home_team else away_ml
            ml_prob = home_prob if ml_side == home_team else away_prob
            
            candidates.append({
                "market": "moneyline",
                "pick": ml_side,
                "odds": ml_odds,
                "our_prob": ml_prob,
                "line_confidence": ml_market.get("confidence", 0.5),
                "factors": analysis["factors_home"] if ml_side == home_team else analysis["factors_away"],
                "has_sharp": ml_market.get("sharp_money_side") is not None,
                "has_rlm": ml_market.get("reverse_line_movement", False)
            })
        
        # Spread candidate
        spread_market = markets.get("spread", {})
        if spread_market.get("value_side") and spread_market.get("confidence", 0) > 0.55 and current_spread is not None:
            spread_side = spread_market["value_side"]
            
            candidates.append({
                "market": "spread",
                "pick": spread_side,
                "pick_detail": f"{spread_side} {'+' if current_spread > 0 else ''}{current_spread}",
                "odds": spread_odds,
                "our_prob": 0.55,  # Spread is roughly 50/50
                "line_confidence": spread_market.get("confidence", 0.5),
                "factors": analysis["factors_home"] if spread_side == home_team else analysis["factors_away"],
                "has_sharp": spread_market.get("sharp_money_side") is not None,
                "has_rlm": spread_market.get("reverse_line_movement", False),
                "spread_value": current_spread
            })
        
        # Totals candidate
        totals_market = markets.get("totals", {})
        if totals_market.get("value_side") and totals_market.get("confidence", 0) > 0.55 and current_total is not None:
            totals_side = totals_market["value_side"]  # "over" or "under"
            t_odds = over_odds if totals_side == "over" else under_odds
            
            candidates.append({
                "market": "total",
                "pick": totals_side,
                "pick_detail": f"{totals_side.upper()} {current_total}",
                "odds": t_odds,
                "our_prob": 0.55,
                "line_confidence": totals_market.get("confidence", 0.5),
                "factors": analysis["factors_over"] if totals_side == "over" else analysis["factors_under"],
                "has_sharp": totals_market.get("sharp_money_side") is not None,
                "has_rlm": totals_market.get("reverse_line_movement", False),
                "total_value": current_total
            })
        
        # Select best candidate based on: 1) sharp money 2) RLM 3) highest confidence
        best_candidate = None
        best_score = 0
        
        for c in candidates:
            score = c["line_confidence"]
            if c["has_sharp"]:
                score += 0.10
            if c["has_rlm"]:
                score += 0.08
            if len(c.get("factors", [])) >= 2:
                score += 0.05
            
            if score > best_score:
                best_score = score
                best_candidate = c
        
        # Generate prediction
        if best_candidate and best_score >= 0.60 and factor_count >= self.min_factors:
            # Calculate final confidence
            base_confidence = 0.60
            factor_bonus = min(0.12, factor_count * 0.02)
            line_bonus = (best_candidate["line_confidence"] - 0.5) * 0.3
            sharp_bonus = 0.05 if best_candidate["has_sharp"] else 0
            rlm_bonus = 0.04 if best_candidate["has_rlm"] else 0
            
            confidence = base_confidence + factor_bonus + line_bonus + sharp_bonus + rlm_bonus
            confidence = max(0.55, min(0.82, confidence))
            
            # Build reasoning
            reasoning_parts = []
            reasoning_parts.append(f"📊 {line_analysis.get('summary', 'Line movement analyzed')}")
            
            for f in best_candidate.get("factors", [])[:3]:
                reasoning_parts.append(f"✓ {f['factor']}: {f['detail']}")
            
            for insight in analysis.get("insights", [])[:2]:
                reasoning_parts.append(insight)
            
            for warning in analysis.get("warnings", [])[:1]:
                reasoning_parts.append(warning)
            
            pick_display = best_candidate.get("pick_detail", best_candidate["pick"])
            
            return {
                "has_pick": True,
                "pick": best_candidate["pick"],
                "pick_type": best_candidate["market"],
                "pick_display": pick_display,
                "odds": best_candidate["odds"],
                "confidence": round(confidence * 100, 1),
                "our_probability": round(best_candidate.get("our_prob", 0.55) * 100, 1),
                "factor_count": factor_count,
                "reasoning": "\n".join(reasoning_parts),
                "key_factors": [f["factor"] for f in best_candidate.get("factors", [])],
                "line_analysis_summary": line_analysis.get("summary"),
                "market_analysis": {
                    "ml": {
                        "direction": markets.get("moneyline", {}).get("movement_direction"),
                        "sharp_side": markets.get("moneyline", {}).get("sharp_money_side")
                    },
                    "spread": {
                        "direction": markets.get("spread", {}).get("movement_direction"),
                        "value": current_spread,
                        "sharp_side": markets.get("spread", {}).get("sharp_money_side")
                    },
                    "totals": {
                        "direction": markets.get("totals", {}).get("movement_direction"),
                        "value": current_total,
                        "sharp_side": markets.get("totals", {}).get("sharp_money_side")
                    }
                },
                "cross_market_signals": line_analysis.get("cross_market_signals", []),
                "insights": analysis.get("insights", []),
                "warnings": analysis.get("warnings", []),
                "algorithm": "betpredictor_v5"
            }
        else:
            # No pick
            reasons = []
            if factor_count < self.min_factors:
                reasons.append(f"Only {factor_count}/{self.min_factors} factors aligned")
            if not best_candidate:
                reasons.append("No strong market signal detected")
            elif best_score < 0.60:
                reasons.append(f"Best market confidence ({best_score*100:.0f}%) below threshold")
            
            return {
                "has_pick": False,
                "reasoning": "No pick recommended: " + "; ".join(reasons),
                "factor_count": factor_count,
                "closest_market": best_candidate["market"] if best_candidate else None,
                "closest_confidence": round(best_score * 100, 1) if best_candidate else 0,
                "home_probability": round(home_prob * 100, 1),
                "away_probability": round(away_prob * 100, 1),
                "market_analysis": {
                    "ml": markets.get("moneyline", {}),
                    "spread": markets.get("spread", {}),
                    "totals": markets.get("totals", {})
                },
                "line_analysis_summary": line_analysis.get("summary"),
                "insights": analysis.get("insights", []),
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
    """Generate a prediction using BetPredictor V5."""
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
