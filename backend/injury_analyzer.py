"""
Injury Analyzer
Intelligent injury impact assessment with player importance weighting
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Player position importance weights by sport
POSITION_WEIGHTS = {
    "basketball_nba": {
        "Guard": 1.0,
        "Forward": 1.0,
        "Center": 0.95,
        "Point Guard": 1.1,  # Most important
        "Shooting Guard": 0.95,
        "Small Forward": 1.0,
        "Power Forward": 0.9,
        "default": 1.0
    },
    "americanfootball_nfl": {
        "Quarterback": 2.0,  # Most critical
        "Running Back": 1.0,
        "Wide Receiver": 1.0,
        "Tight End": 0.8,
        "Offensive Line": 0.9,
        "Defensive Line": 0.8,
        "Linebacker": 0.85,
        "Cornerback": 0.9,
        "Safety": 0.8,
        "default": 0.8
    },
    "icehockey_nhl": {
        "Goalie": 1.5,  # Most important
        "Center": 1.1,
        "Left Wing": 1.0,
        "Right Wing": 1.0,
        "Defenseman": 1.0,
        "default": 1.0
    },
    "soccer_epl": {
        "Goalkeeper": 1.3,
        "Striker": 1.2,
        "Midfielder": 1.0,
        "Defender": 0.9,
        "default": 1.0
    }
}

# Injury severity multipliers
INJURY_SEVERITY = {
    "out": 1.0,
    "doubtful": 0.7,
    "questionable": 0.4,
    "probable": 0.2,
    "day-to-day": 0.3,
    "gtd": 0.5,  # Game time decision
    "default": 0.5
}


class InjuryAnalyzer:
    """
    Analyzes injury impact with intelligent weighting.
    """
    
    def __init__(self, sport_key: str):
        self.sport_key = sport_key
        self.position_weights = POSITION_WEIGHTS.get(sport_key, POSITION_WEIGHTS["basketball_nba"])
        
        # Sport-specific max impact
        self.max_impact = {
            "basketball_nba": 0.12,  # Star player can swing 12%
            "americanfootball_nfl": 0.15,  # QB injury huge
            "icehockey_nhl": 0.10,  # Goalie injury significant
            "soccer_epl": 0.08
        }.get(sport_key, 0.10)
    
    def analyze_team_injuries(self, injuries: List[Dict], roster: List[Dict] = None) -> Dict:
        """
        Analyze all injuries for a team and calculate total impact.
        
        Args:
            injuries: List of injury dicts with status, position, name
            roster: Optional roster data for importance calculation
        
        Returns:
            Dict with total impact, key injuries, and details
        """
        if not injuries:
            return {
                "total_impact": 0.0,
                "key_players_out": 0,
                "minor_injuries": 0,
                "star_players_out": [],
                "details": []
            }
        
        total_impact = 0.0
        key_players_out = 0
        minor_injuries = 0
        star_players_out = []
        details = []
        
        for injury in injuries:
            player_name = injury.get("name", "Unknown")
            status = injury.get("status", "").lower()
            position = injury.get("position", "")
            
            # Calculate player importance
            importance = self._calculate_player_importance(player_name, position, roster)
            
            # Get severity multiplier
            severity = INJURY_SEVERITY.get(status, INJURY_SEVERITY["default"])
            
            # Calculate this player's impact
            player_impact = importance * severity
            
            # Scale by max impact
            scaled_impact = player_impact * self.max_impact
            
            total_impact += scaled_impact
            
            # Categorize
            if scaled_impact > self.max_impact * 0.4:  # Major impact
                key_players_out += 1
                star_players_out.append({
                    "name": player_name,
                    "position": position,
                    "status": status,
                    "impact": round(scaled_impact, 3)
                })
            else:
                minor_injuries += 1
            
            details.append({
                "player": player_name,
                "position": position,
                "status": status,
                "importance": round(importance, 2),
                "severity": round(severity, 2),
                "impact": round(scaled_impact, 3)
            })
        
        # Cap total impact at max
        total_impact = min(total_impact, self.max_impact * 1.5)
        
        return {
            "total_impact": round(total_impact, 3),
            "key_players_out": key_players_out,
            "minor_injuries": minor_injuries,
            "star_players_out": star_players_out,
            "total_injuries": len(injuries),
            "details": details
        }
    
    def _calculate_player_importance(self, player_name: str, position: str, roster: List[Dict] = None) -> float:
        """
        Calculate player importance score (0-1).
        Uses position weight and any available stats.
        """
        # Base importance from position
        position_weight = self.position_weights.get(position, self.position_weights.get("default", 1.0))
        
        # Normalize position weight (QB at 2.0 -> 1.0, others scaled)
        max_position_weight = max(self.position_weights.values())
        normalized_position = position_weight / max_position_weight
        
        # TODO: In production, could enhance with:
        # - Player stats (PPG, usage rate, win shares)
        # - Salary/contract value
        # - All-Star/Pro Bowl status
        # - Team's win rate with/without player
        
        # For now, use position-based importance
        importance = normalized_position
        
        # Bonus for certain keywords in player name (star players)
        # This is a simple heuristic - in production would use actual player data
        name_lower = player_name.lower()
        if any(keyword in name_lower for keyword in ["mvp", "star", "all-star"]):
            importance = min(1.0, importance * 1.2)
        
        return importance
    
    def compare_team_health(self, home_injuries: List[Dict], away_injuries: List[Dict]) -> Dict:
        """
        Compare injury situations between two teams.
        """
        home_analysis = self.analyze_team_injuries(home_injuries)
        away_analysis = self.analyze_team_injuries(away_injuries)
        
        # Calculate net advantage
        net_advantage = away_analysis["total_impact"] - home_analysis["total_impact"]
        
        # Determine which team has advantage
        advantage_team = None
        if net_advantage > 0.02:
            advantage_team = "home"
        elif net_advantage < -0.02:
            advantage_team = "away"
        
        return {
            "home_team": home_analysis,
            "away_team": away_analysis,
            "net_advantage": round(net_advantage, 3),
            "advantage_team": advantage_team,
            "significant_difference": abs(net_advantage) > 0.03,
            "summary": self._generate_injury_summary(home_analysis, away_analysis, net_advantage)
        }
    
    def _generate_injury_summary(self, home_analysis: Dict, away_analysis: Dict, net_advantage: float) -> str:
        """Generate human-readable injury summary."""
        parts = []
        
        home_key = home_analysis["key_players_out"]
        away_key = away_analysis["key_players_out"]
        
        if home_key > 0:
            parts.append(f"Home missing {home_key} key player(s)")
        if away_key > 0:
            parts.append(f"Away missing {away_key} key player(s)")
        
        if abs(net_advantage) > 0.03:
            if net_advantage > 0:
                parts.append("Home has significant health advantage")
            else:
                parts.append("Away has significant health advantage")
        elif abs(net_advantage) < 0.01:
            parts.append("Both teams relatively healthy")
        
        return "; ".join(parts) if parts else "No significant injuries"


def analyze_injury_impact(sport_key: str, home_injuries: List[Dict], away_injuries: List[Dict]) -> Dict:
    """
    Main entry point for injury analysis.
    """
    analyzer = InjuryAnalyzer(sport_key)
    return analyzer.compare_team_health(home_injuries, away_injuries)
