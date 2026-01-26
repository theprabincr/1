"""
Line Movement Analyzer V2 - Deep Line Movement Analysis
Studies complete line movement history from opening to 1 hour before game.

Key Features:
1. Analyzes complete line movement trajectory (not just opening vs current)
2. Identifies movement patterns (gradual drift, sharp moves, reversals)
3. Detects sharp money vs public money patterns
4. Correlates movement timing with game factors
5. Calculates line movement confidence score
6. Identifies value based on overreaction/underreaction to news

Analysis Philosophy:
- Sharp money typically comes in early (12-24 hours before)
- Public money comes late (within 2 hours)
- Reverse line movement = sharp action against public
- Steam moves = coordinated sharp betting
- Line settling = market found true price
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


class LineMovementAnalyzer:
    """
    Comprehensive line movement analyzer that studies the complete
    trajectory of odds from opening to pre-game.
    """
    
    def __init__(self):
        self.significant_move_threshold = 0.03  # 3% move is significant
        self.steam_move_threshold = 0.05  # 5% rapid move
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
        Perform complete analysis of line movement history.
        
        Returns comprehensive analysis including:
        - Movement trajectory
        - Sharp money indicators
        - Steam moves detected
        - Reverse line movement
        - Confidence adjustment
        - Reasoning for why line moved
        """
        analysis = {
            "has_meaningful_movement": False,
            "movement_direction": "neutral",  # home, away, or neutral
            "total_movement_pct": 0,
            "sharp_money_side": None,
            "public_money_side": None,
            "reverse_line_movement": False,
            "steam_moves": [],
            "movement_phases": [],
            "confidence_adjustment": 0,
            "value_side": None,
            "reasoning": [],
            "key_insights": []
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
        
        # 1. Calculate total movement
        opening_home = opening_odds.get("home_odds", line_history[0].get("home_odds", 1.91))
        opening_away = opening_odds.get("away_odds", line_history[0].get("away_odds", 1.91))
        
        current_home = current_odds.get("home", line_history[-1].get("home_odds", 1.91))
        current_away = current_odds.get("away", line_history[-1].get("away_odds", 1.91))
        
        if opening_home and opening_home > 0:
            total_home_move = (current_home - opening_home) / opening_home * 100
        else:
            total_home_move = 0
        
        analysis["total_movement_pct"] = total_home_move
        
        # 2. Determine overall movement direction
        if total_home_move < -3:  # Home odds dropped = money on home
            analysis["movement_direction"] = "home"
            analysis["has_meaningful_movement"] = True
        elif total_home_move > 3:  # Home odds increased = money on away
            analysis["movement_direction"] = "away"
            analysis["has_meaningful_movement"] = True
        else:
            analysis["movement_direction"] = "neutral"
        
        # 3. Analyze movement phases (early, mid, late)
        phases = self._analyze_movement_phases(line_history, game_time)
        analysis["movement_phases"] = phases
        
        # 4. Detect sharp vs public money patterns
        sharp_analysis = self._detect_sharp_money(phases, home_team, away_team)
        analysis["sharp_money_side"] = sharp_analysis.get("sharp_side")
        analysis["public_money_side"] = sharp_analysis.get("public_side")
        analysis["reverse_line_movement"] = sharp_analysis.get("rlm", False)
        
        if sharp_analysis.get("sharp_side"):
            analysis["key_insights"].append(
                f"Sharp money detected on {sharp_analysis['sharp_side']} "
                f"(moved {abs(sharp_analysis.get('sharp_move_pct', 0)):.1f}% in sharp window)"
            )
        
        if sharp_analysis.get("rlm"):
            analysis["key_insights"].append(
                f"⚠️ Reverse Line Movement: Line moving opposite to public action"
            )
        
        # 5. Detect steam moves
        steam_moves = self._detect_steam_moves(line_history, home_team, away_team)
        analysis["steam_moves"] = steam_moves
        
        if steam_moves:
            analysis["key_insights"].append(
                f"Steam move(s) detected: {len(steam_moves)} rapid price changes"
            )
        
        # 6. Analyze line trajectory pattern
        trajectory = self._analyze_trajectory_pattern(line_history)
        analysis["trajectory_pattern"] = trajectory["pattern"]
        
        if trajectory.get("insight"):
            analysis["key_insights"].append(trajectory["insight"])
        
        # 7. Calculate confidence adjustment
        conf_adj = self._calculate_confidence_adjustment(analysis, phases)
        analysis["confidence_adjustment"] = conf_adj["adjustment"]
        analysis["reasoning"].extend(conf_adj["reasons"])
        
        # 8. Determine value side based on analysis
        value_analysis = self._determine_value_side(analysis, home_team, away_team)
        analysis["value_side"] = value_analysis.get("side")
        analysis["value_confidence"] = value_analysis.get("confidence", 0)
        
        if value_analysis.get("reason"):
            analysis["reasoning"].append(value_analysis["reason"])
        
        # 9. Generate comprehensive reasoning
        analysis["summary"] = self._generate_summary(analysis, home_team, away_team)
        
        return analysis
    
    def _analyze_movement_phases(
        self,
        line_history: List[Dict],
        game_time: datetime
    ) -> List[Dict]:
        """
        Break down movement into phases:
        - Early (>24 hours before): Sharp money window
        - Mid (6-24 hours): Mixed action
        - Late (<6 hours): Public money window
        """
        phases = []
        now = datetime.now(timezone.utc)
        
        early_moves = []
        mid_moves = []
        late_moves = []
        
        for i, snap in enumerate(line_history):
            try:
                snap_time = datetime.fromisoformat(snap.get("timestamp", "").replace('Z', '+00:00'))
            except:
                continue
            
            hours_to_game = (game_time - snap_time).total_seconds() / 3600
            
            home_odds = snap.get("home_odds", 0)
            
            move_data = {
                "timestamp": snap.get("timestamp"),
                "home_odds": home_odds,
                "hours_to_game": hours_to_game
            }
            
            if hours_to_game > 24:
                early_moves.append(move_data)
            elif hours_to_game > 6:
                mid_moves.append(move_data)
            else:
                late_moves.append(move_data)
        
        # Calculate movement in each phase
        if len(early_moves) >= 2:
            early_start = early_moves[0]["home_odds"]
            early_end = early_moves[-1]["home_odds"]
            if early_start > 0:
                early_move = (early_end - early_start) / early_start * 100
                phases.append({
                    "phase": "early",
                    "label": "Sharp Window (>24h)",
                    "movement_pct": early_move,
                    "snapshots": len(early_moves)
                })
        
        if len(mid_moves) >= 2:
            mid_start = mid_moves[0]["home_odds"]
            mid_end = mid_moves[-1]["home_odds"]
            if mid_start > 0:
                mid_move = (mid_end - mid_start) / mid_start * 100
                phases.append({
                    "phase": "mid",
                    "label": "Mixed Action (6-24h)",
                    "movement_pct": mid_move,
                    "snapshots": len(mid_moves)
                })
        
        if len(late_moves) >= 2:
            late_start = late_moves[0]["home_odds"]
            late_end = late_moves[-1]["home_odds"]
            if late_start > 0:
                late_move = (late_end - late_start) / late_start * 100
                phases.append({
                    "phase": "late",
                    "label": "Public Window (<6h)",
                    "movement_pct": late_move,
                    "snapshots": len(late_moves)
                })
        
        return phases
    
    def _detect_sharp_money(
        self,
        phases: List[Dict],
        home_team: str,
        away_team: str
    ) -> Dict:
        """
        Detect sharp money patterns based on timing and direction.
        Sharp money characteristics:
        - Comes early (>12 hours before game)
        - Large single moves
        - Often opposite to public sentiment
        """
        result = {
            "sharp_side": None,
            "public_side": None,
            "rlm": False,
            "sharp_move_pct": 0
        }
        
        early_move = 0
        late_move = 0
        
        for phase in phases:
            if phase["phase"] == "early":
                early_move = phase["movement_pct"]
            elif phase["phase"] == "late":
                late_move = phase["movement_pct"]
        
        # Sharp money detection: significant early movement
        if abs(early_move) >= 3:
            if early_move < 0:  # Home odds dropped early
                result["sharp_side"] = home_team
                result["sharp_move_pct"] = early_move
            else:
                result["sharp_side"] = away_team
                result["sharp_move_pct"] = early_move
        
        # Public money detection: late movement
        if abs(late_move) >= 2:
            if late_move < 0:  # Home odds dropped late
                result["public_side"] = home_team
            else:
                result["public_side"] = away_team
        
        # Reverse Line Movement: when sharp and public are opposite
        if result["sharp_side"] and result["public_side"]:
            if result["sharp_side"] != result["public_side"]:
                result["rlm"] = True
        
        return result
    
    def _detect_steam_moves(
        self,
        line_history: List[Dict],
        home_team: str,
        away_team: str
    ) -> List[Dict]:
        """
        Detect steam moves - rapid, coordinated price changes
        indicating sharp syndicate action.
        """
        steam_moves = []
        
        for i in range(1, len(line_history)):
            prev = line_history[i-1]
            curr = line_history[i]
            
            prev_home = prev.get("home_odds", 0)
            curr_home = curr.get("home_odds", 0)
            
            if prev_home <= 0:
                continue
            
            move_pct = (curr_home - prev_home) / prev_home * 100
            
            # Steam move = rapid significant move (>3% in single update)
            if abs(move_pct) >= 3:
                side = home_team if move_pct < 0 else away_team
                steam_moves.append({
                    "timestamp": curr.get("timestamp"),
                    "side": side,
                    "move_pct": move_pct,
                    "from_odds": prev_home,
                    "to_odds": curr_home
                })
        
        return steam_moves
    
    def _analyze_trajectory_pattern(self, line_history: List[Dict]) -> Dict:
        """
        Identify the overall trajectory pattern:
        - Steady drift: Gradual one-direction movement
        - Oscillation: Back and forth
        - Sharp correction: Sudden reversal
        - Settling: Converging to stable price
        """
        if len(line_history) < 3:
            return {"pattern": "insufficient_data", "insight": None}
        
        # Calculate direction changes
        directions = []
        for i in range(1, len(line_history)):
            prev = line_history[i-1].get("home_odds", 0)
            curr = line_history[i].get("home_odds", 0)
            
            if prev > 0 and curr > 0:
                if curr > prev:
                    directions.append("up")
                elif curr < prev:
                    directions.append("down")
                else:
                    directions.append("flat")
        
        if not directions:
            return {"pattern": "no_movement", "insight": None}
        
        # Count direction changes
        changes = sum(1 for i in range(1, len(directions)) 
                     if directions[i] != directions[i-1] and directions[i] != "flat")
        
        # Determine pattern
        if changes <= 1:
            if directions.count("down") > directions.count("up"):
                return {
                    "pattern": "steady_drift_home",
                    "insight": "Consistent movement toward home team - market confidence"
                }
            elif directions.count("up") > directions.count("down"):
                return {
                    "pattern": "steady_drift_away",
                    "insight": "Consistent movement toward away team - market confidence"
                }
        elif changes >= len(directions) * 0.5:
            return {
                "pattern": "oscillation",
                "insight": "Market uncertainty - no clear consensus"
            }
        
        # Check for late reversal
        if len(directions) >= 4:
            early_trend = directions[:len(directions)//2]
            late_trend = directions[len(directions)//2:]
            
            early_down = early_trend.count("down") > early_trend.count("up")
            late_up = late_trend.count("up") > late_trend.count("down")
            
            if early_down and late_up:
                return {
                    "pattern": "late_reversal_away",
                    "insight": "Late reversal toward away - possible sharp correction"
                }
            elif not early_down and not late_up:
                return {
                    "pattern": "late_reversal_home",
                    "insight": "Late reversal toward home - possible sharp correction"
                }
        
        return {"pattern": "mixed", "insight": None}
    
    def _calculate_confidence_adjustment(
        self,
        analysis: Dict,
        phases: List[Dict]
    ) -> Dict:
        """
        Calculate confidence adjustment based on line movement analysis.
        Returns adjustment factor (-0.10 to +0.10) and reasoning.
        """
        adjustment = 0
        reasons = []
        
        # 1. Sharp money alignment (+3-5%)
        if analysis.get("sharp_money_side"):
            adjustment += 0.03
            reasons.append(f"+3% confidence: Sharp money detected on {analysis['sharp_money_side']}")
        
        # 2. Reverse Line Movement (+3-5%)
        if analysis.get("reverse_line_movement"):
            adjustment += 0.04
            reasons.append("+4% confidence: Reverse line movement (sharp vs public)")
        
        # 3. Steam moves (+2% per steam move, max +4%)
        steam_bonus = min(0.04, len(analysis.get("steam_moves", [])) * 0.02)
        if steam_bonus > 0:
            adjustment += steam_bonus
            reasons.append(f"+{steam_bonus*100:.0f}% confidence: Steam move(s) detected")
        
        # 4. Trajectory pattern
        pattern = analysis.get("trajectory_pattern", "")
        if pattern in ["steady_drift_home", "steady_drift_away"]:
            adjustment += 0.02
            reasons.append("+2% confidence: Consistent market direction")
        elif pattern == "oscillation":
            adjustment -= 0.02
            reasons.append("-2% confidence: Market uncertainty (oscillating line)")
        
        # 5. Phase agreement (early and late moving same direction)
        early_dir = None
        late_dir = None
        for phase in phases:
            if phase["phase"] == "early" and abs(phase["movement_pct"]) > 2:
                early_dir = "home" if phase["movement_pct"] < 0 else "away"
            elif phase["phase"] == "late" and abs(phase["movement_pct"]) > 2:
                late_dir = "home" if phase["movement_pct"] < 0 else "away"
        
        if early_dir and late_dir:
            if early_dir == late_dir:
                adjustment += 0.02
                reasons.append("+2% confidence: Early and late money aligned")
            else:
                adjustment -= 0.01
                reasons.append("-1% confidence: Early and late money diverged")
        
        # Cap adjustment
        adjustment = max(-0.10, min(0.10, adjustment))
        
        return {
            "adjustment": adjustment,
            "reasons": reasons
        }
    
    def _determine_value_side(
        self,
        analysis: Dict,
        home_team: str,
        away_team: str
    ) -> Dict:
        """
        Determine which side has value based on line movement analysis.
        """
        # Priority: Sharp money > RLM > Steam moves > Total movement
        
        # If we have sharp money, follow it
        if analysis.get("sharp_money_side"):
            return {
                "side": analysis["sharp_money_side"],
                "confidence": 0.65 + analysis.get("confidence_adjustment", 0),
                "reason": f"Following sharp money on {analysis['sharp_money_side']}"
            }
        
        # If RLM, take the sharp side (opposite of public)
        if analysis.get("reverse_line_movement"):
            if analysis.get("public_money_side"):
                value_side = home_team if analysis["public_money_side"] == away_team else away_team
                return {
                    "side": value_side,
                    "confidence": 0.60 + analysis.get("confidence_adjustment", 0),
                    "reason": f"RLM indicates value on {value_side}"
                }
        
        # If steam moves, follow them
        if analysis.get("steam_moves"):
            last_steam = analysis["steam_moves"][-1]
            return {
                "side": last_steam["side"],
                "confidence": 0.58 + analysis.get("confidence_adjustment", 0),
                "reason": f"Steam move detected on {last_steam['side']}"
            }
        
        # If meaningful total movement, follow market
        if analysis.get("has_meaningful_movement"):
            direction = analysis.get("movement_direction")
            if direction == "home":
                return {
                    "side": home_team,
                    "confidence": 0.55 + analysis.get("confidence_adjustment", 0),
                    "reason": f"Market moving toward {home_team}"
                }
            elif direction == "away":
                return {
                    "side": away_team,
                    "confidence": 0.55 + analysis.get("confidence_adjustment", 0),
                    "reason": f"Market moving toward {away_team}"
                }
        
        return {
            "side": None,
            "confidence": 0.50,
            "reason": "No clear value signal from line movement"
        }
    
    def _generate_summary(
        self,
        analysis: Dict,
        home_team: str,
        away_team: str
    ) -> str:
        """Generate a human-readable summary of the line movement analysis."""
        parts = []
        
        # Total movement
        total_move = analysis.get("total_movement_pct", 0)
        if abs(total_move) >= 3:
            direction = home_team if total_move < 0 else away_team
            parts.append(f"Line moved {abs(total_move):.1f}% toward {direction}")
        else:
            parts.append("Line has been stable")
        
        # Key insights
        if analysis.get("key_insights"):
            parts.append("Key signals: " + "; ".join(analysis["key_insights"][:2]))
        
        # Value conclusion
        if analysis.get("value_side"):
            parts.append(
                f"Line movement suggests value on {analysis['value_side']} "
                f"({analysis.get('value_confidence', 0.5)*100:.0f}% confidence)"
            )
        
        return ". ".join(parts)


# Convenience function
async def analyze_line_movement(
    line_history: List[Dict],
    opening_odds: Dict,
    current_odds: Dict,
    commence_time: str,
    home_team: str,
    away_team: str
) -> Dict:
    """
    Main entry point for line movement analysis.
    """
    analyzer = LineMovementAnalyzer()
    return analyzer.analyze_complete_movement(
        line_history,
        opening_odds,
        current_odds,
        commence_time,
        home_team,
        away_team
    )
