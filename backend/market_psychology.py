"""
Market Psychology Analyzer
Detects public bias, identifies contrarian opportunities, and analyzes market inefficiencies
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class MarketPsychologyAnalyzer:
    """
    Analyzes market psychology and identifies betting opportunities.
    """
    
    def __init__(self, sport_key: str):
        self.sport_key = sport_key
    
    def detect_public_bias(self, event: Dict, line_movement: Dict, matchup_data: Dict) -> Dict:
        """
        Detect public betting bias patterns.
        
        Common biases:
        1. Home favorite bias (public loves home favorites)
        2. Overs bias (public loves overs)
        3. Big name bias (public bets popular teams)
        4. Recency bias (hot teams overvalued)
        5. Primetime bias (national TV games)
        """
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        
        biases = []
        total_bias_score = 0.0
        
        # 1. Home Favorite Bias
        odds = event.get("odds", {})
        home_ml = odds.get("home_ml_decimal", 2.0)
        away_ml = odds.get("away_ml_decimal", 2.0)
        
        home_favorite = home_ml < away_ml
        
        if home_favorite and home_ml < 1.70:  # Strong home favorite
            biases.append({
                "type": "home_favorite",
                "strength": "strong",
                "public_side": home_team,
                "value_side": away_team,
                "description": f"Public likely heavy on home favorite {home_team}"
            })
            total_bias_score += 0.08
        
        # 2. Overs Bias
        totals_movement = line_movement.get("markets", {}).get("totals", {})
        total_direction = totals_movement.get("movement_direction", "neutral")
        
        if total_direction == "over":
            biases.append({
                "type": "overs_bias",
                "strength": "moderate",
                "public_side": "over",
                "value_side": "under",
                "description": "Total moving up - public on over"
            })
            total_bias_score += 0.05
        
        # 3. Recency Bias (hot teams overvalued)
        home_form = matchup_data.get("home_team", {}).get("form", {})
        away_form = matchup_data.get("away_team", {}).get("form", {})
        
        home_streak = home_form.get("streak", 0)
        away_streak = away_form.get("streak", 0)
        
        if home_streak >= 4:  # Hot home team
            biases.append({
                "type": "recency_bias",
                "strength": "moderate",
                "public_side": home_team,
                "value_side": away_team,
                "description": f"{home_team} on {home_streak}-game win streak - may be overvalued"
            })
            total_bias_score += 0.06
        
        if away_streak >= 4:  # Hot away team
            biases.append({
                "type": "recency_bias",
                "strength": "moderate",
                "public_side": away_team,
                "value_side": home_team,
                "description": f"{away_team} on {away_streak}-game win streak - may be overvalued"
            })
            total_bias_score += 0.06
        
        # 4. Popular Team Bias (based on team names - simplified)
        popular_teams = [
            "Lakers", "Warriors", "Celtics", "Knicks",  # NBA
            "Cowboys", "Patriots", "Packers", "Steelers",  # NFL
            "Yankees", "Red Sox", "Dodgers",  # MLB
            "Maple Leafs", "Canadiens", "Rangers"  # NHL
        ]
        
        home_popular = any(team in home_team for team in popular_teams)
        away_popular = any(team in away_team for team in popular_teams)
        
        if home_popular and not away_popular:
            biases.append({
                "type": "popular_team",
                "strength": "moderate",
                "public_side": home_team,
                "value_side": away_team,
                "description": f"{home_team} is popular team - public likely overweighting"
            })
            total_bias_score += 0.04
        
        return {
            "has_public_bias": len(biases) > 0,
            "total_bias_score": round(total_bias_score, 3),
            "bias_count": len(biases),
            "biases": biases
        }
    
    def identify_contrarian_opportunity(self, event: Dict, line_movement: Dict, public_bias: Dict) -> Dict:
        """
        Identify contrarian betting opportunities.
        Contrarian = betting against public when sharp money disagrees.
        """
        markets = line_movement.get("markets", {})
        
        opportunities = []
        
        # Check each market for contrarian signals
        for market_name, market_data in markets.items():
            sharp_side = market_data.get("sharp_money_side")
            public_side = market_data.get("public_money_side")
            rlm = market_data.get("reverse_line_movement", False)
            
            # Contrarian opportunity: sharp and public disagree
            if sharp_side and public_side and sharp_side != public_side:
                opportunities.append({
                    "market": market_name,
                    "contrarian_side": sharp_side,
                    "public_side": public_side,
                    "has_rlm": rlm,
                    "confidence": 0.70 if rlm else 0.60,
                    "description": f"Sharp money on {sharp_side}, public on {public_side}"
                })
            
            # RLM alone is also contrarian signal
            elif rlm:
                opportunities.append({
                    "market": market_name,
                    "contrarian_side": sharp_side,
                    "has_rlm": True,
                    "confidence": 0.65,
                    "description": f"Reverse line movement in {market_name}"
                })
        
        # Check if public bias exists and could be faded
        fade_opportunities = []
        if public_bias.get("has_public_bias"):
            for bias in public_bias.get("biases", []):
                if bias.get("strength") in ["strong", "moderate"]:
                    fade_opportunities.append({
                        "bias_type": bias["type"],
                        "fade_side": bias.get("value_side"),
                        "reason": bias.get("description")
                    })
        
        return {
            "has_contrarian_opportunity": len(opportunities) > 0,
            "opportunity_count": len(opportunities),
            "opportunities": opportunities,
            "fade_opportunities": fade_opportunities,
            "contrarian_score": self._calculate_contrarian_score(opportunities, fade_opportunities)
        }
    
    def analyze_market_efficiency(self, event: Dict, line_movement: Dict) -> Dict:
        """
        Analyze market efficiency.
        Less efficient markets = more opportunity.
        """
        markets = line_movement.get("markets", {})
        
        efficiency_scores = []
        
        for market_name, market_data in markets.items():
            # Factors indicating efficient market:
            # 1. Small line movement
            # 2. No RLM
            # 3. Sharp and public agree
            # 4. No steam moves
            
            movement = abs(market_data.get("total_movement", 0))
            has_rlm = market_data.get("reverse_line_movement", False)
            steam_moves = len(market_data.get("steam_moves", []))
            sharp_side = market_data.get("sharp_money_side")
            public_side = market_data.get("public_money_side")
            
            # Calculate efficiency (0-100, higher = more efficient)
            efficiency = 70  # Base
            
            # Large movement = less efficient
            if market_name == "moneyline":
                if movement > 5:
                    efficiency -= 15
                elif movement > 3:
                    efficiency -= 10
            else:  # spread/totals
                if movement > 2:
                    efficiency -= 15
                elif movement > 1:
                    efficiency -= 10
            
            # RLM = inefficiency
            if has_rlm:
                efficiency -= 20
            
            # Steam moves = inefficiency
            if steam_moves > 0:
                efficiency -= (10 * steam_moves)
            
            # Sharp/public disagreement = inefficiency
            if sharp_side and public_side and sharp_side != public_side:
                efficiency -= 15
            
            efficiency = max(0, min(100, efficiency))
            
            efficiency_scores.append({
                "market": market_name,
                "efficiency_score": efficiency,
                "inefficiency_indicators": {
                    "large_movement": movement > 2,
                    "reverse_line_movement": has_rlm,
                    "steam_moves": steam_moves,
                    "sharp_public_split": (sharp_side != public_side) if (sharp_side and public_side) else False
                }
            })
        
        avg_efficiency = sum(m["efficiency_score"] for m in efficiency_scores) / len(efficiency_scores) if efficiency_scores else 70
        
        return {
            "average_efficiency": round(avg_efficiency, 1),
            "market_efficiency": efficiency_scores,
            "opportunity_level": "high" if avg_efficiency < 50 else "moderate" if avg_efficiency < 70 else "low"
        }
    
    def _calculate_contrarian_score(self, opportunities: List[Dict], fade_opportunities: List[Dict]) -> float:
        """Calculate overall contrarian opportunity score."""
        score = 0.0
        
        for opp in opportunities:
            score += opp.get("confidence", 0.5) * 0.1
        
        for fade in fade_opportunities:
            score += 0.05
        
        return min(0.20, round(score, 3))  # Cap at 20% boost


def analyze_market_psychology(sport_key: str, event: Dict, line_movement: Dict, matchup_data: Dict) -> Dict:
    """
    Main entry point for market psychology analysis.
    """
    analyzer = MarketPsychologyAnalyzer(sport_key)
    
    public_bias = analyzer.detect_public_bias(event, line_movement, matchup_data)
    contrarian = analyzer.identify_contrarian_opportunity(event, line_movement, public_bias)
    efficiency = analyzer.analyze_market_efficiency(event, line_movement)
    
    return {
        "public_bias": public_bias,
        "contrarian_opportunities": contrarian,
        "market_efficiency": efficiency,
        "overall_psychology_score": round(
            contrarian.get("contrarian_score", 0.0) + 
            (0.05 if efficiency.get("opportunity_level") == "high" else 0.0),
            3
        )
    }
