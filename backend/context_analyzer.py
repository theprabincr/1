"""
Context Analyzer
Analyzes situational factors: rest days, travel, schedule, time of season
"""
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# City coordinates for travel distance calculation (major sports cities)
CITY_COORDS = {
    # NBA/NHL/NFL Cities
    "Atlanta": (33.7490, -84.3880),
    "Boston": (42.3601, -71.0589),
    "Brooklyn": (40.6782, -73.9442),
    "Charlotte": (35.2271, -80.8431),
    "Chicago": (41.8781, -87.6298),
    "Cleveland": (41.4993, -81.6944),
    "Dallas": (32.7767, -96.7970),
    "Denver": (39.7392, -104.9903),
    "Detroit": (42.3314, -83.0458),
    "Golden State": (37.7749, -122.4194),  # SF
    "Houston": (29.7604, -95.3698),
    "Indianapolis": (39.7684, -86.1581),
    "Los Angeles": (34.0522, -118.2437),
    "Memphis": (35.1495, -90.0490),
    "Miami": (25.7617, -80.1918),
    "Milwaukee": (43.0389, -87.9065),
    "Minnesota": (44.9778, -93.2650),  # Minneapolis
    "New Orleans": (29.9511, -90.0715),
    "New York": (40.7128, -74.0060),
    "Oklahoma City": (35.4676, -97.5164),
    "Orlando": (28.5383, -81.3792),
    "Philadelphia": (39.9526, -75.1652),
    "Phoenix": (33.4484, -112.0740),
    "Portland": (45.5152, -122.6784),
    "Sacramento": (38.5816, -121.4944),
    "San Antonio": (29.4241, -98.4936),
    "Toronto": (43.6532, -79.3832),
    "Utah": (40.7608, -111.8910),  # Salt Lake City
    "Washington": (38.9072, -77.0369),
    # Additional cities
    "Seattle": (47.6062, -122.3321),
    "Tampa": (27.9506, -82.4572),
    "Las Vegas": (36.1699, -115.1398),
    "Kansas City": (39.0997, -94.5786),
    "Baltimore": (39.2904, -76.6122),
    "Pittsburgh": (40.4406, -79.9959),
    "Columbus": (39.9612, -82.9988),
    "Nashville": (36.1627, -86.7816),
    "Buffalo": (42.8864, -78.8784),
    "San Francisco": (37.7749, -122.4194),
    "Oakland": (37.8044, -122.2712)
}

# Altitude adjustments (feet above sea level)
CITY_ALTITUDE = {
    "Denver": 5280,
    "Utah": 4226,  # Salt Lake City
    "Phoenix": 1086,
    "Las Vegas": 2001,
    # Most other cities are near sea level (<500 ft)
}


class ContextAnalyzer:
    """
    Analyzes contextual factors that impact game outcomes.
    """
    
    def __init__(self, sport_key: str):
        self.sport_key = sport_key
        
        # Sport-specific context weights
        self.config = {
            "basketball_nba": {
                "rest_days_weight": 0.08,
                "travel_weight": 0.05,
                "altitude_weight": 0.03,
                "back_to_back_penalty": -0.08,
                "rested_bonus": 0.04,
                "timezone_penalty_per_hour": 0.015
            },
            "americanfootball_nfl": {
                "rest_days_weight": 0.06,
                "travel_weight": 0.04,
                "altitude_weight": 0.04,
                "short_week_penalty": -0.06,
                "extra_rest_bonus": 0.05,
                "timezone_penalty_per_hour": 0.020
            },
            "icehockey_nhl": {
                "rest_days_weight": 0.07,
                "travel_weight": 0.06,
                "altitude_weight": 0.02,
                "back_to_back_penalty": -0.07,
                "rested_bonus": 0.03,
                "timezone_penalty_per_hour": 0.012
            },
            "soccer_epl": {
                "rest_days_weight": 0.06,
                "travel_weight": 0.03,
                "altitude_weight": 0.01,
                "congested_schedule_penalty": -0.05,
                "rested_bonus": 0.03,
                "timezone_penalty_per_hour": 0.010
            }
        }
        
        self.config_data = self.config.get(sport_key, self.config["basketball_nba"])
    
    def analyze_rest_days(self, recent_games: List[Dict], game_time: str) -> Dict:
        """
        Analyze rest days situation.
        Returns impact score and details.
        """
        if not recent_games:
            return {
                "days_since_last_game": 2,
                "is_back_to_back": False,
                "is_well_rested": False,
                "impact_score": 0.0,
                "description": "No recent games data"
            }
        
        try:
            game_dt = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
        except:
            game_dt = datetime.now(timezone.utc)
        
        # Find most recent game
        last_game = recent_games[0]
        try:
            last_game_dt = datetime.fromisoformat(last_game.get("date", "").replace('Z', '+00:00'))
            days_since = (game_dt - last_game_dt).days
        except:
            days_since = 2  # Default assumption
        
        is_back_to_back = (days_since == 0 or days_since == 1)
        is_well_rested = (days_since >= 3)
        
        # Calculate impact
        impact = 0.0
        description = ""
        
        if is_back_to_back:
            impact = self.config_data.get("back_to_back_penalty", -0.08)
            description = f"Back-to-back game (B2B), {days_since} day rest"
        elif is_well_rested:
            impact = self.config_data.get("rested_bonus", 0.04)
            description = f"Well rested, {days_since} days off"
        else:
            # Normal rest (2 days)
            impact = 0.0
            description = f"Normal rest, {days_since} days since last game"
        
        return {
            "days_since_last_game": days_since,
            "is_back_to_back": is_back_to_back,
            "is_well_rested": is_well_rested,
            "impact_score": impact,
            "description": description
        }
    
    def analyze_travel_distance(self, home_team: str, away_team: str, away_recent_games: List[Dict]) -> Dict:
        """
        Analyze travel distance and timezone changes for away team.
        """
        # Extract city from team name
        home_city = self._extract_city(home_team)
        away_city = self._extract_city(away_team)
        
        if not home_city or not away_city:
            return {
                "distance_miles": 0,
                "timezone_change": 0,
                "impact_score": 0.0,
                "description": "Travel data not available"
            }
        
        # Calculate distance
        distance = self._calculate_distance(home_city, away_city)
        
        # Estimate timezone change
        tz_change = self._estimate_timezone_change(home_city, away_city)
        
        # Calculate impact
        impact = 0.0
        
        # Long distance travel penalty
        if distance > 2000:  # Cross-country
            impact -= self.config_data.get("travel_weight", 0.05)
        elif distance > 1000:
            impact -= self.config_data.get("travel_weight", 0.05) * 0.6
        
        # Timezone penalty
        if abs(tz_change) >= 3:
            impact -= self.config_data.get("timezone_penalty_per_hour", 0.015) * abs(tz_change)
        
        description = f"Travel: {distance} miles, {abs(tz_change)} timezone shift"
        
        return {
            "distance_miles": distance,
            "timezone_change": tz_change,
            "impact_score": round(impact, 3),
            "description": description
        }
    
    def analyze_altitude(self, home_team: str, away_team: str) -> Dict:
        """
        Analyze altitude advantage (mainly Denver, Utah).
        """
        home_city = self._extract_city(home_team)
        away_city = self._extract_city(away_team)
        
        home_altitude = CITY_ALTITUDE.get(home_city, 0)
        away_altitude = CITY_ALTITUDE.get(away_city, 0)
        
        altitude_diff = home_altitude - away_altitude
        
        impact = 0.0
        description = "No significant altitude difference"
        
        # Only impactful if home team plays at high altitude and away doesn't
        if altitude_diff > 3000:
            impact = self.config_data.get("altitude_weight", 0.03)
            description = f"Home altitude advantage: {home_altitude} ft vs {away_altitude} ft"
        
        return {
            "home_altitude": home_altitude,
            "away_altitude": away_altitude,
            "altitude_diff": altitude_diff,
            "impact_score": impact,
            "description": description
        }
    
    def analyze_schedule_strength(self, recent_games: List[Dict], game_time: str) -> Dict:
        """
        Analyze schedule difficulty (many games in short time).
        """
        if len(recent_games) < 5:
            return {
                "games_in_7_days": 0,
                "games_in_14_days": 0,
                "is_congested": False,
                "impact_score": 0.0,
                "description": "Insufficient schedule data"
            }
        
        try:
            game_dt = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
        except:
            game_dt = datetime.now(timezone.utc)
        
        # Count games in last 7 and 14 days
        games_7_days = 0
        games_14_days = 0
        
        for g in recent_games:
            try:
                g_dt = datetime.fromisoformat(g.get("date", "").replace('Z', '+00:00'))
                days_ago = (game_dt - g_dt).days
                
                if 0 <= days_ago <= 7:
                    games_7_days += 1
                if 0 <= days_ago <= 14:
                    games_14_days += 1
            except:
                continue
        
        # Congested schedule detection
        is_congested = False
        impact = 0.0
        description = "Normal schedule"
        
        if self.sport_key == "basketball_nba":
            if games_7_days >= 4:
                is_congested = True
                impact = -0.03
                description = f"Congested schedule: {games_7_days} games in 7 days"
        elif self.sport_key == "icehockey_nhl":
            if games_7_days >= 4:
                is_congested = True
                impact = -0.03
                description = f"Congested schedule: {games_7_days} games in 7 days"
        
        return {
            "games_in_7_days": games_7_days,
            "games_in_14_days": games_14_days,
            "is_congested": is_congested,
            "impact_score": impact,
            "description": description
        }
    
    def get_comprehensive_context(self, matchup_data: Dict, game_time: str) -> Dict:
        """
        Get complete contextual analysis for both teams.
        """
        home_team = matchup_data.get("home_team", {}).get("name", "")
        away_team = matchup_data.get("away_team", {}).get("name", "")
        
        home_recent = matchup_data.get("home_team", {}).get("recent_games", [])
        away_recent = matchup_data.get("away_team", {}).get("recent_games", [])
        
        # Analyze home team
        home_context = {
            "rest": self.analyze_rest_days(home_recent, game_time),
            "travel": {"distance_miles": 0, "timezone_change": 0, "impact_score": 0.0, "description": "Home team (no travel)"},
            "altitude": self.analyze_altitude(home_team, away_team),
            "schedule": self.analyze_schedule_strength(home_recent, game_time)
        }
        
        # Analyze away team
        away_context = {
            "rest": self.analyze_rest_days(away_recent, game_time),
            "travel": self.analyze_travel_distance(home_team, away_team, away_recent),
            "altitude": {"home_altitude": 0, "away_altitude": 0, "altitude_diff": 0, "impact_score": 0.0, "description": "Away team (altitude tracked for home)"},
            "schedule": self.analyze_schedule_strength(away_recent, game_time)
        }
        
        # Calculate total impact
        home_total = sum([
            home_context["rest"]["impact_score"],
            home_context["travel"]["impact_score"],
            home_context["altitude"]["impact_score"],
            home_context["schedule"]["impact_score"]
        ])
        
        away_total = sum([
            away_context["rest"]["impact_score"],
            away_context["travel"]["impact_score"],
            away_context["altitude"]["impact_score"],
            away_context["schedule"]["impact_score"]
        ])
        
        net_advantage = home_total - away_total
        
        return {
            "home_team": home_context,
            "away_team": away_context,
            "home_total_impact": round(home_total, 3),
            "away_total_impact": round(away_total, 3),
            "net_context_advantage": round(net_advantage, 3),
            "key_factors": self._identify_key_context_factors(home_context, away_context)
        }
    
    def _identify_key_context_factors(self, home_context: Dict, away_context: Dict) -> List[str]:
        """Identify the most important contextual factors."""
        factors = []
        
        # Rest
        if home_context["rest"]["is_back_to_back"]:
            factors.append(f"Home team on B2B")
        if away_context["rest"]["is_back_to_back"]:
            factors.append(f"Away team on B2B")
        if home_context["rest"]["is_well_rested"]:
            factors.append(f"Home team well rested")
        if away_context["rest"]["is_well_rested"]:
            factors.append(f"Away team well rested")
        
        # Travel
        if away_context["travel"]["distance_miles"] > 2000:
            factors.append(f"Away team cross-country travel")
        if abs(away_context["travel"]["timezone_change"]) >= 3:
            factors.append(f"Away team {abs(away_context['travel']['timezone_change'])} timezone shift")
        
        # Altitude
        if home_context["altitude"]["impact_score"] > 0.02:
            factors.append(f"Altitude advantage (Denver/Utah)")
        
        # Schedule
        if home_context["schedule"]["is_congested"]:
            factors.append(f"Home team congested schedule")
        if away_context["schedule"]["is_congested"]:
            factors.append(f"Away team congested schedule")
        
        return factors
    
    def _extract_city(self, team_name: str) -> Optional[str]:
        """Extract city from team name."""
        for city in CITY_COORDS.keys():
            if city.lower() in team_name.lower():
                return city
        return None
    
    def _calculate_distance(self, city1: str, city2: str) -> int:
        """Calculate distance between two cities using Haversine formula."""
        if city1 not in CITY_COORDS or city2 not in CITY_COORDS:
            return 0
        
        lat1, lon1 = CITY_COORDS[city1]
        lat2, lon2 = CITY_COORDS[city2]
        
        # Haversine formula
        R = 3959  # Earth radius in miles
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        distance = R * c
        return int(distance)
    
    def _estimate_timezone_change(self, city1: str, city2: str) -> int:
        """Estimate timezone change (rough approximation)."""
        # Simple longitude-based estimation
        if city1 not in CITY_COORDS or city2 not in CITY_COORDS:
            return 0
        
        _, lon1 = CITY_COORDS[city1]
        _, lon2 = CITY_COORDS[city2]
        
        # ~15 degrees longitude = 1 hour
        tz_change = (lon2 - lon1) / 15
        return int(round(tz_change))
