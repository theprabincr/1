"""
Line Movement Analyzer V3 - Complete Market Analysis
Studies complete line movement history for ALL markets (ML, Spread, Totals).

Key Features:
1. Analyzes ML, Spread, and Totals movement from opening to game time
2. Identifies movement patterns across all markets
3. Detects sharp money vs public money patterns
4. Correlates movements between markets (e.g., spread moving but ML not)
5. Calculates market-specific confidence scores
6. Identifies value in each market type

Analysis Philosophy:
- Sharp money typically comes in early (12-24 hours before)
- Public money comes late (within 2 hours)
- Reverse line movement = sharp action against public
- Steam moves = coordinated sharp betting
- Cross-market signals = when spread and totals confirm each other
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


class LineMovementAnalyzer:
    """
    Comprehensive line movement analyzer for ALL markets.
    """
    
    def __init__(self):
        self.significant_ml_move = 0.03  # 3% ML move is significant
        self.significant_spread_move = 0.5  # Half point spread move
        self.significant_total_move = 1.0  # 1 point total move
        self.steam_ml_threshold = 0.05  # 5% rapid ML move
        self.steam_spread_threshold = 1.0  # 1 point rapid spread move
        self.steam_total_threshold = 1.5  # 1.5 point rapid total move
        self.sharp_timing_hours = (12, 48)  # Sharp money window
        self.public_timing_hours = (0, 3)  # Public money window
    
    def analyze_complete_movement(
        self,
        line_history: List[Dict],
        opening_odds: Dict,
        current_odds: Dict,
        commence_time: str,
        home_team: str,
        away_team: str
    ) -> Dict:
        """
        Perform complete analysis of line movement for ALL markets.
        """
        analysis = {
            "has_meaningful_movement": False,
            "markets": {
                "moneyline": self._create_empty_market_analysis(),
                "spread": self._create_empty_market_analysis(),
                "totals": self._create_empty_market_analysis()
            },
            "cross_market_signals": [],
            "overall_direction": "neutral",  # home, away, or neutral
            "sharp_money_detected": False,
            "confidence_adjustment": 0,
            "recommended_market": None,
            "reasoning": [],
            "key_insights": [],
            "summary": ""
        }
        
        if not line_history or len(line_history) < 2:
            analysis["reasoning"].append("Insufficient line movement data for analysis")
            return analysis
        
        # Parse commence time
        try:
            if commence_time:
                game_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            else:
                game_time = datetime.now(timezone.utc) + timedelta(hours=6)
        except:
            game_time = datetime.now(timezone.utc) + timedelta(hours=6)
        
        # Analyze each market
        analysis["markets"]["moneyline"] = self._analyze_moneyline_movement(
            line_history, opening_odds, current_odds, game_time, home_team, away_team
        )
        
        analysis["markets"]["spread"] = self._analyze_spread_movement(
            line_history, opening_odds, current_odds, game_time, home_team, away_team
        )
        
        analysis["markets"]["totals"] = self._analyze_totals_movement(
            line_history, opening_odds, current_odds, game_time
        )
        
        # Cross-market analysis
        cross_signals = self._analyze_cross_market_signals(analysis["markets"], home_team, away_team)
        analysis["cross_market_signals"] = cross_signals
        
        # Determine overall direction
        analysis["overall_direction"] = self._determine_overall_direction(analysis["markets"])
        
        # Check for sharp money
        for market_name, market_data in analysis["markets"].items():
            if market_data.get("sharp_money_side"):
                analysis["sharp_money_detected"] = True
                break
        
        # Has meaningful movement in any market?
        analysis["has_meaningful_movement"] = any(
            m.get("has_meaningful_movement") for m in analysis["markets"].values()
        )
        
        # Calculate confidence adjustment
        conf_adj = self._calculate_total_confidence_adjustment(analysis)
        analysis["confidence_adjustment"] = conf_adj
        
        # Determine best market to bet
        analysis["recommended_market"] = self._determine_recommended_market(analysis, home_team, away_team)
        
        # Generate summary
        analysis["summary"] = self._generate_comprehensive_summary(analysis, home_team, away_team)
        
        return analysis
    
    def _create_empty_market_analysis(self) -> Dict:
        return {
            "has_meaningful_movement": False,
            "movement_direction": "neutral",
            "total_movement": 0,
            "opening_value": None,
            "current_value": None,
            "sharp_money_side": None,
            "public_money_side": None,
            "reverse_line_movement": False,
            "steam_moves": [],
            "phases": [],
            "value_side": None,
            "confidence": 0.50
        }
    
    def _analyze_moneyline_movement(
        self,
        line_history: List[Dict],
        opening_odds: Dict,
        current_odds: Dict,
        game_time: datetime,
        home_team: str,
        away_team: str
    ) -> Dict:
        """Analyze moneyline movement."""
        result = self._create_empty_market_analysis()
        
        # Get opening and current ML
        opening_home = opening_odds.get("home_odds", line_history[0].get("home_odds", 1.91))
        opening_away = opening_odds.get("away_odds", line_history[0].get("away_odds", 1.91))
        
        current_home = current_odds.get("home", line_history[-1].get("home_odds", 1.91))
        current_away = current_odds.get("away", line_history[-1].get("away_odds", 1.91))
        
        if not opening_home or opening_home <= 0:
            return result
        
        result["opening_value"] = {"home": opening_home, "away": opening_away}
        result["current_value"] = {"home": current_home, "away": current_away}
        
        # Calculate total movement
        total_move = (current_home - opening_home) / opening_home * 100
        result["total_movement"] = round(total_move, 2)
        
        # Direction
        if total_move < -3:
            result["movement_direction"] = "home"
            result["has_meaningful_movement"] = True
        elif total_move > 3:
            result["movement_direction"] = "away"
            result["has_meaningful_movement"] = True
        
        # Analyze phases
        ml_history = [(h.get("timestamp"), h.get("home_odds")) for h in line_history if h.get("home_odds")]
        result["phases"] = self._analyze_phases(ml_history, game_time, "ml")
        
        # Sharp/public detection
        sharp_public = self._detect_sharp_public(result["phases"], home_team, away_team, "ml")
        result["sharp_money_side"] = sharp_public.get("sharp_side")
        result["public_money_side"] = sharp_public.get("public_side")
        result["reverse_line_movement"] = sharp_public.get("rlm", False)
        
        # Steam moves
        result["steam_moves"] = self._detect_steam_moves_ml(line_history, home_team, away_team)
        
        # Value side
        result["value_side"], result["confidence"] = self._determine_ml_value(result, home_team, away_team)
        
        return result
    
    def _analyze_spread_movement(
        self,
        line_history: List[Dict],
        opening_odds: Dict,
        current_odds: Dict,
        game_time: datetime,
        home_team: str,
        away_team: str
    ) -> Dict:
        """Analyze spread movement."""
        result = self._create_empty_market_analysis()
        
        # Get opening and current spread
        opening_spread = opening_odds.get("spread")
        current_spread = None
        
        # Find current spread from history
        for h in reversed(line_history):
            if h.get("spread") is not None:
                current_spread = h.get("spread")
                break
        
        if opening_spread is None:
            # Try to get from first history record
            for h in line_history:
                if h.get("spread") is not None:
                    opening_spread = h.get("spread")
                    break
        
        if opening_spread is None or current_spread is None:
            return result
        
        result["opening_value"] = opening_spread
        result["current_value"] = current_spread
        
        # Calculate movement (negative = moved toward home, positive = moved toward away)
        total_move = current_spread - opening_spread
        result["total_movement"] = round(total_move, 1)
        
        # Direction (spread decreased = more money on home)
        if total_move <= -0.5:
            result["movement_direction"] = "home"
            result["has_meaningful_movement"] = True
        elif total_move >= 0.5:
            result["movement_direction"] = "away"
            result["has_meaningful_movement"] = True
        
        # Analyze phases
        spread_history = [(h.get("timestamp"), h.get("spread")) for h in line_history if h.get("spread") is not None]
        result["phases"] = self._analyze_phases(spread_history, game_time, "spread")
        
        # Sharp/public detection
        sharp_public = self._detect_sharp_public(result["phases"], home_team, away_team, "spread")
        result["sharp_money_side"] = sharp_public.get("sharp_side")
        result["public_money_side"] = sharp_public.get("public_side")
        result["reverse_line_movement"] = sharp_public.get("rlm", False)
        
        # Steam moves
        result["steam_moves"] = self._detect_steam_moves_spread(line_history, home_team, away_team)
        
        # Value side
        result["value_side"], result["confidence"] = self._determine_spread_value(result, home_team, away_team)
        
        return result
    
    def _analyze_totals_movement(
        self,
        line_history: List[Dict],
        opening_odds: Dict,
        current_odds: Dict,
        game_time: datetime
    ) -> Dict:
        """Analyze totals (over/under) movement."""
        result = self._create_empty_market_analysis()
        
        # Get opening and current total
        opening_total = opening_odds.get("total")
        current_total = None
        
        for h in reversed(line_history):
            if h.get("total") is not None:
                current_total = h.get("total")
                break
        
        if opening_total is None:
            for h in line_history:
                if h.get("total") is not None:
                    opening_total = h.get("total")
                    break
        
        if opening_total is None or current_total is None:
            return result
        
        result["opening_value"] = opening_total
        result["current_value"] = current_total
        
        # Calculate movement
        total_move = current_total - opening_total
        result["total_movement"] = round(total_move, 1)
        
        # Direction (increase = over money, decrease = under money)
        if total_move >= 1.0:
            result["movement_direction"] = "over"
            result["has_meaningful_movement"] = True
        elif total_move <= -1.0:
            result["movement_direction"] = "under"
            result["has_meaningful_movement"] = True
        
        # Analyze phases
        total_history = [(h.get("timestamp"), h.get("total")) for h in line_history if h.get("total") is not None]
        result["phases"] = self._analyze_phases(total_history, game_time, "total")
        
        # Sharp/public detection for totals
        sharp_public = self._detect_sharp_public_totals(result["phases"])
        result["sharp_money_side"] = sharp_public.get("sharp_side")
        result["public_money_side"] = sharp_public.get("public_side")
        result["reverse_line_movement"] = sharp_public.get("rlm", False)
        
        # Steam moves
        result["steam_moves"] = self._detect_steam_moves_total(line_history)
        
        # Value side
        result["value_side"], result["confidence"] = self._determine_totals_value(result)
        
        return result
    
    def _analyze_phases(
        self,
        history: List[Tuple],
        game_time: datetime,
        market_type: str
    ) -> List[Dict]:
        """Break down movement into early/mid/late phases."""
        phases = []
        
        early_vals = []
        mid_vals = []
        late_vals = []
        
        for ts_str, value in history:
            if not ts_str or value is None:
                continue
            try:
                snap_time = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                hours_to_game = (game_time - snap_time).total_seconds() / 3600
                
                if hours_to_game > 24:
                    early_vals.append(value)
                elif hours_to_game > 6:
                    mid_vals.append(value)
                else:
                    late_vals.append(value)
            except:
                continue
        
        # Calculate movement in each phase
        for phase_name, vals in [("early", early_vals), ("mid", mid_vals), ("late", late_vals)]:
            if len(vals) >= 2:
                if market_type == "ml":
                    move = (vals[-1] - vals[0]) / vals[0] * 100 if vals[0] > 0 else 0
                else:
                    move = vals[-1] - vals[0]
                
                phases.append({
                    "phase": phase_name,
                    "movement": round(move, 2),
                    "snapshots": len(vals)
                })
        
        return phases
    
    def _detect_sharp_public(
        self,
        phases: List[Dict],
        home_team: str,
        away_team: str,
        market_type: str
    ) -> Dict:
        """Detect sharp vs public money patterns."""
        result = {"sharp_side": None, "public_side": None, "rlm": False}
        
        early_move = 0
        late_move = 0
        
        for phase in phases:
            if phase["phase"] == "early":
                early_move = phase["movement"]
            elif phase["phase"] == "late":
                late_move = phase["movement"]
        
        # Thresholds depend on market type
        if market_type == "ml":
            sharp_threshold = 3
            public_threshold = 2
        else:  # spread
            sharp_threshold = 0.5
            public_threshold = 0.5
        
        # Sharp money detection
        if abs(early_move) >= sharp_threshold:
            if market_type == "ml":
                result["sharp_side"] = home_team if early_move < 0 else away_team
            else:  # spread - negative move = toward home
                result["sharp_side"] = home_team if early_move < 0 else away_team
        
        # Public money detection
        if abs(late_move) >= public_threshold:
            if market_type == "ml":
                result["public_side"] = home_team if late_move < 0 else away_team
            else:
                result["public_side"] = home_team if late_move < 0 else away_team
        
        # RLM
        if result["sharp_side"] and result["public_side"]:
            if result["sharp_side"] != result["public_side"]:
                result["rlm"] = True
        
        return result
    
    def _detect_sharp_public_totals(self, phases: List[Dict]) -> Dict:
        """Detect sharp vs public for totals."""
        result = {"sharp_side": None, "public_side": None, "rlm": False}
        
        early_move = 0
        late_move = 0
        
        for phase in phases:
            if phase["phase"] == "early":
                early_move = phase["movement"]
            elif phase["phase"] == "late":
                late_move = phase["movement"]
        
        # Sharp money
        if abs(early_move) >= 1.0:
            result["sharp_side"] = "over" if early_move > 0 else "under"
        
        # Public money
        if abs(late_move) >= 0.5:
            result["public_side"] = "over" if late_move > 0 else "under"
        
        # RLM
        if result["sharp_side"] and result["public_side"]:
            if result["sharp_side"] != result["public_side"]:
                result["rlm"] = True
        
        return result
    
    def _detect_steam_moves_ml(self, line_history: List[Dict], home_team: str, away_team: str) -> List[Dict]:
        """Detect steam moves in ML."""
        steam_moves = []
        
        for i in range(1, len(line_history)):
            prev = line_history[i-1].get("home_odds", 0)
            curr = line_history[i].get("home_odds", 0)
            
            if prev <= 0:
                continue
            
            move_pct = (curr - prev) / prev * 100
            
            if abs(move_pct) >= 3:
                steam_moves.append({
                    "timestamp": line_history[i].get("timestamp"),
                    "side": home_team if move_pct < 0 else away_team,
                    "move": f"{move_pct:+.1f}%"
                })
        
        return steam_moves
    
    def _detect_steam_moves_spread(self, line_history: List[Dict], home_team: str, away_team: str) -> List[Dict]:
        """Detect steam moves in spread."""
        steam_moves = []
        prev_spread = None
        
        for h in line_history:
            curr_spread = h.get("spread")
            if curr_spread is None:
                continue
            
            if prev_spread is not None:
                move = curr_spread - prev_spread
                
                if abs(move) >= 1.0:
                    steam_moves.append({
                        "timestamp": h.get("timestamp"),
                        "side": home_team if move < 0 else away_team,
                        "move": f"{move:+.1f} points"
                    })
            
            prev_spread = curr_spread
        
        return steam_moves
    
    def _detect_steam_moves_total(self, line_history: List[Dict]) -> List[Dict]:
        """Detect steam moves in total."""
        steam_moves = []
        prev_total = None
        
        for h in line_history:
            curr_total = h.get("total")
            if curr_total is None:
                continue
            
            if prev_total is not None:
                move = curr_total - prev_total
                
                if abs(move) >= 1.5:
                    steam_moves.append({
                        "timestamp": h.get("timestamp"),
                        "side": "over" if move > 0 else "under",
                        "move": f"{move:+.1f} points"
                    })
            
            prev_total = curr_total
        
        return steam_moves
    
    def _determine_ml_value(self, result: Dict, home_team: str, away_team: str) -> Tuple[Optional[str], float]:
        """Determine value side for ML."""
        confidence = 0.50
        value_side = None
        
        if result.get("sharp_money_side"):
            value_side = result["sharp_money_side"]
            confidence = 0.62
            
            if result.get("reverse_line_movement"):
                confidence += 0.05
            
            if result.get("steam_moves"):
                confidence += 0.02
        elif result.get("has_meaningful_movement"):
            if result["movement_direction"] == "home":
                value_side = home_team
            else:
                value_side = away_team
            confidence = 0.55
        
        return value_side, min(0.75, confidence)
    
    def _determine_spread_value(self, result: Dict, home_team: str, away_team: str) -> Tuple[Optional[str], float]:
        """Determine value side for spread."""
        confidence = 0.50
        value_side = None
        
        if result.get("sharp_money_side"):
            value_side = result["sharp_money_side"]
            confidence = 0.60
            
            if result.get("reverse_line_movement"):
                confidence += 0.05
        elif result.get("has_meaningful_movement"):
            if result["movement_direction"] == "home":
                value_side = home_team
            else:
                value_side = away_team
            confidence = 0.55
        
        return value_side, min(0.72, confidence)
    
    def _determine_totals_value(self, result: Dict) -> Tuple[Optional[str], float]:
        """Determine value side for totals."""
        confidence = 0.50
        value_side = None
        
        if result.get("sharp_money_side"):
            value_side = result["sharp_money_side"]
            confidence = 0.60
            
            if result.get("reverse_line_movement"):
                confidence += 0.05
        elif result.get("has_meaningful_movement"):
            value_side = result["movement_direction"]  # "over" or "under"
            confidence = 0.55
        
        return value_side, min(0.70, confidence)
    
    def _analyze_cross_market_signals(
        self,
        markets: Dict,
        home_team: str,
        away_team: str
    ) -> List[str]:
        """Analyze signals that appear across multiple markets."""
        signals = []
        
        ml = markets.get("moneyline", {})
        spread = markets.get("spread", {})
        totals = markets.get("totals", {})
        
        # ML and Spread agreement
        if ml.get("value_side") and spread.get("value_side"):
            if ml["value_side"] == spread["value_side"]:
                signals.append(f"✓ ML and Spread both favor {ml['value_side']} - strong signal")
            else:
                signals.append(f"⚠️ ML favors {ml['value_side']} but Spread favors {spread['value_side']} - mixed signals")
        
        # Sharp money across markets
        sharp_sides = []
        for market_name, market_data in markets.items():
            if market_data.get("sharp_money_side"):
                sharp_sides.append(market_data["sharp_money_side"])
        
        if len(sharp_sides) >= 2:
            if len(set(sharp_sides)) == 1:
                signals.append(f"💰 Sharp money detected across multiple markets on {sharp_sides[0]}")
            else:
                signals.append("⚠️ Sharp money split across different sides in different markets")
        
        # RLM in any market
        for market_name, market_data in markets.items():
            if market_data.get("reverse_line_movement"):
                signals.append(f"🔄 Reverse Line Movement detected in {market_name.upper()}")
        
        return signals
    
    def _determine_overall_direction(self, markets: Dict) -> str:
        """Determine overall betting direction across all markets."""
        home_signals = 0
        away_signals = 0
        
        ml = markets.get("moneyline", {})
        spread = markets.get("spread", {})
        
        if ml.get("movement_direction") == "home":
            home_signals += 1
        elif ml.get("movement_direction") == "away":
            away_signals += 1
        
        if spread.get("movement_direction") == "home":
            home_signals += 1
        elif spread.get("movement_direction") == "away":
            away_signals += 1
        
        if home_signals > away_signals:
            return "home"
        elif away_signals > home_signals:
            return "away"
        return "neutral"
    
    def _calculate_total_confidence_adjustment(self, analysis: Dict) -> float:
        """Calculate total confidence adjustment based on all markets."""
        adjustment = 0
        
        for market_name, market_data in analysis["markets"].items():
            if market_data.get("sharp_money_side"):
                adjustment += 0.02
            if market_data.get("reverse_line_movement"):
                adjustment += 0.02
            if market_data.get("steam_moves"):
                adjustment += 0.01
        
        if len(analysis.get("cross_market_signals", [])) >= 2:
            adjustment += 0.02
        
        return min(0.10, adjustment)
    
    def _determine_recommended_market(
        self,
        analysis: Dict,
        home_team: str,
        away_team: str
    ) -> Optional[Dict]:
        """Determine the best market to bet based on line movement."""
        markets = analysis["markets"]
        best_market = None
        best_confidence = 0.50
        
        for market_name, market_data in markets.items():
            if market_data.get("value_side") and market_data.get("confidence", 0) > best_confidence:
                if market_data.get("sharp_money_side") or market_data.get("reverse_line_movement"):
                    best_confidence = market_data["confidence"]
                    best_market = {
                        "market": market_name,
                        "side": market_data["value_side"],
                        "confidence": market_data["confidence"],
                        "opening": market_data.get("opening_value"),
                        "current": market_data.get("current_value"),
                        "movement": market_data.get("total_movement"),
                        "reason": self._get_market_reason(market_data, market_name)
                    }
        
        return best_market
    
    def _get_market_reason(self, market_data: Dict, market_name: str) -> str:
        """Generate reason for market recommendation."""
        reasons = []
        
        if market_data.get("sharp_money_side"):
            reasons.append(f"Sharp money on {market_data['sharp_money_side']}")
        
        if market_data.get("reverse_line_movement"):
            reasons.append("Reverse line movement detected")
        
        if market_data.get("steam_moves"):
            reasons.append(f"{len(market_data['steam_moves'])} steam move(s)")
        
        if not reasons and market_data.get("has_meaningful_movement"):
            reasons.append(f"Significant {market_name} movement")
        
        return "; ".join(reasons) if reasons else "No strong signal"
    
    def _generate_comprehensive_summary(
        self,
        analysis: Dict,
        home_team: str,
        away_team: str
    ) -> str:
        """Generate summary of all market analysis."""
        parts = []
        
        ml = analysis["markets"].get("moneyline", {})
        spread = analysis["markets"].get("spread", {})
        totals = analysis["markets"].get("totals", {})
        
        # ML summary
        if ml.get("has_meaningful_movement"):
            parts.append(f"ML: {abs(ml.get('total_movement', 0)):.1f}% toward {ml.get('movement_direction', 'neutral')}")
        
        # Spread summary
        if spread.get("has_meaningful_movement"):
            parts.append(f"Spread: {abs(spread.get('total_movement', 0)):.1f}pts toward {spread.get('movement_direction', 'neutral')}")
        
        # Totals summary
        if totals.get("has_meaningful_movement"):
            parts.append(f"Total: {abs(totals.get('total_movement', 0)):.1f}pts toward {totals.get('movement_direction', 'neutral')}")
        
        # Sharp money
        if analysis.get("sharp_money_detected"):
            parts.append("Sharp money detected")
        
        # Recommended market
        if analysis.get("recommended_market"):
            rec = analysis["recommended_market"]
            parts.append(f"Best bet: {rec['market'].upper()} {rec['side']} ({rec['confidence']*100:.0f}% conf)")
        
        return " | ".join(parts) if parts else "No significant line movement detected"


# Convenience function
async def analyze_line_movement(
    line_history: List[Dict],
    opening_odds: Dict,
    current_odds: Dict,
    commence_time: str,
    home_team: str,
    away_team: str
) -> Dict:
    """Main entry point for line movement analysis."""
    analyzer = LineMovementAnalyzer()
    return analyzer.analyze_complete_movement(
        line_history,
        opening_odds,
        current_odds,
        commence_time,
        home_team,
        away_team
    )
