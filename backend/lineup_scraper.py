"""
Lineup and Roster Scraper - Fetches playing squads and injury info from ESPN
Uses ESPN's free hidden API endpoints for roster/lineup data
"""
import asyncio
import logging
import httpx
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ESPN API endpoints for rosters and game summaries
ESPN_ROSTER_ENDPOINTS = {
    "basketball_nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster",
    "americanfootball_nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster",
    "baseball_mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams/{team_id}/roster",
    "icehockey_nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{team_id}/roster",
    "soccer_epl": "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/teams/{team_id}/roster",
}

ESPN_GAME_SUMMARY = {
    "basketball_nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}",
    "americanfootball_nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}",
    "baseball_mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event={event_id}",
    "icehockey_nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={event_id}",
    "soccer_epl": "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/summary?event={event_id}",
}

# Team name to ESPN ID mappings (subset - can be expanded)
NBA_TEAM_IDS = {
    "atlanta hawks": "1", "boston celtics": "2", "brooklyn nets": "17", 
    "charlotte hornets": "30", "chicago bulls": "4", "cleveland cavaliers": "5",
    "dallas mavericks": "6", "denver nuggets": "7", "detroit pistons": "8",
    "golden state warriors": "9", "houston rockets": "10", "indiana pacers": "11",
    "los angeles clippers": "12", "la clippers": "12", "los angeles lakers": "13", 
    "la lakers": "13", "memphis grizzlies": "29", "miami heat": "14",
    "milwaukee bucks": "15", "minnesota timberwolves": "16", "new orleans pelicans": "3",
    "new york knicks": "18", "oklahoma city thunder": "25", "orlando magic": "19",
    "philadelphia 76ers": "20", "phoenix suns": "21", "portland trail blazers": "22",
    "sacramento kings": "23", "san antonio spurs": "24", "toronto raptors": "28",
    "utah jazz": "26", "washington wizards": "27"
}

NFL_TEAM_IDS = {
    "arizona cardinals": "22", "atlanta falcons": "1", "baltimore ravens": "33",
    "buffalo bills": "2", "carolina panthers": "29", "chicago bears": "3",
    "cincinnati bengals": "4", "cleveland browns": "5", "dallas cowboys": "6",
    "denver broncos": "7", "detroit lions": "8", "green bay packers": "9",
    "houston texans": "34", "indianapolis colts": "11", "jacksonville jaguars": "30",
    "kansas city chiefs": "12", "las vegas raiders": "13", "los angeles chargers": "24",
    "los angeles rams": "14", "miami dolphins": "15", "minnesota vikings": "16",
    "new england patriots": "17", "new orleans saints": "18", "new york giants": "19",
    "new york jets": "20", "philadelphia eagles": "21", "pittsburgh steelers": "23",
    "san francisco 49ers": "25", "seattle seahawks": "26", "tampa bay buccaneers": "27",
    "tennessee titans": "10", "washington commanders": "28"
}

NHL_TEAM_IDS = {
    "anaheim ducks": "25", "boston bruins": "1", "buffalo sabres": "2",
    "calgary flames": "3", "carolina hurricanes": "4", "chicago blackhawks": "5",
    "colorado avalanche": "6", "columbus blue jackets": "7", "dallas stars": "8",
    "detroit red wings": "9", "edmonton oilers": "10", "florida panthers": "11",
    "los angeles kings": "12", "minnesota wild": "13", "montreal canadiens": "14",
    "nashville predators": "15", "new jersey devils": "16", "new york islanders": "17",
    "new york rangers": "18", "ottawa senators": "19", "philadelphia flyers": "20",
    "pittsburgh penguins": "21", "san jose sharks": "22", "seattle kraken": "30",
    "st louis blues": "23", "tampa bay lightning": "24", "toronto maple leafs": "26",
    "vancouver canucks": "27", "vegas golden knights": "29", "washington capitals": "28",
    "winnipeg jets": "31"
}


def get_team_id(team_name: str, sport_key: str) -> Optional[str]:
    """Get ESPN team ID from team name"""
    team_lower = team_name.lower().strip()
    
    if "nba" in sport_key or "basketball" in sport_key:
        return NBA_TEAM_IDS.get(team_lower)
    elif "nfl" in sport_key or "football" in sport_key:
        return NFL_TEAM_IDS.get(team_lower)
    elif "nhl" in sport_key or "hockey" in sport_key:
        return NHL_TEAM_IDS.get(team_lower)
    
    return None


async def fetch_team_roster(team_name: str, sport_key: str) -> Dict:
    """Fetch team roster from ESPN"""
    team_id = get_team_id(team_name, sport_key)
    
    if not team_id or sport_key not in ESPN_ROSTER_ENDPOINTS:
        return {"team": team_name, "players": [], "injuries": []}
    
    url = ESPN_ROSTER_ENDPOINTS[sport_key].format(team_id=team_id)
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return parse_roster_response(data, team_name)
        except Exception as e:
            logger.error(f"Error fetching roster for {team_name}: {e}")
    
    return {"team": team_name, "players": [], "injuries": []}


def parse_roster_response(data: dict, team_name: str) -> Dict:
    """Parse ESPN roster response"""
    result = {
        "team": team_name,
        "players": [],
        "injuries": [],
        "key_players": []
    }
    
    try:
        athletes = data.get("athletes", [])
        
        for category in athletes:
            items = category.get("items", [])
            position = category.get("position", "")
            
            for player in items:
                player_info = {
                    "name": player.get("displayName", ""),
                    "position": player.get("position", {}).get("abbreviation", position),
                    "jersey": player.get("jersey", ""),
                    "status": "active"
                }
                
                # Check for injuries
                injuries = player.get("injuries", [])
                if injuries:
                    injury = injuries[0]
                    player_info["status"] = injury.get("status", "questionable")
                    player_info["injury"] = injury.get("type", {}).get("description", "")
                    result["injuries"].append(player_info)
                
                result["players"].append(player_info)
                
                # Mark key players (starters typically have lower jersey numbers or specific positions)
                if player_info["position"] in ["PG", "SG", "SF", "PF", "C", "QB", "RB", "WR", "G", "D", "C"]:
                    result["key_players"].append(player_info["name"])
        
    except Exception as e:
        logger.error(f"Error parsing roster: {e}")
    
    return result


async def fetch_game_lineup(espn_event_id: str, sport_key: str) -> Dict:
    """Fetch lineup/boxscore data for a specific game"""
    if sport_key not in ESPN_GAME_SUMMARY:
        return {}
    
    url = ESPN_GAME_SUMMARY[sport_key].format(event_id=espn_event_id)
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return parse_game_summary(data)
        except Exception as e:
            logger.error(f"Error fetching game summary {espn_event_id}: {e}")
    
    return {}


def parse_game_summary(data: dict) -> Dict:
    """Parse ESPN game summary for lineup info"""
    result = {
        "home": {"team": "", "starters": [], "projected": []},
        "away": {"team": "", "starters": [], "projected": []},
        "injuries": []
    }
    
    try:
        # Get boxscore data
        boxscore = data.get("boxscore", {})
        
        # Get team info
        teams = boxscore.get("teams", [])
        for team in teams:
            team_data = team.get("team", {})
            team_name = team_data.get("displayName", "")
            home_away = team.get("homeAway", "")
            
            stats = team.get("statistics", [])
            
            if home_away == "home":
                result["home"]["team"] = team_name
            else:
                result["away"]["team"] = team_name
        
        # Get players (starters typically have more minutes)
        players = boxscore.get("players", [])
        for team_players in players:
            home_away = team_players.get("homeAway", "")
            statistics = team_players.get("statistics", [])
            
            for stat_cat in statistics:
                athletes = stat_cat.get("athletes", [])
                for i, athlete in enumerate(athletes[:5]):  # First 5 are typically starters
                    player_name = athlete.get("athlete", {}).get("displayName", "")
                    if player_name:
                        if home_away == "home":
                            result["home"]["starters"].append(player_name)
                        else:
                            result["away"]["starters"].append(player_name)
        
        # Get injury report
        injuries = data.get("injuries", [])
        for team_injuries in injuries:
            team_name = team_injuries.get("team", {}).get("displayName", "")
            for injury in team_injuries.get("injuries", []):
                result["injuries"].append({
                    "team": team_name,
                    "player": injury.get("athlete", {}).get("displayName", ""),
                    "status": injury.get("status", ""),
                    "type": injury.get("type", {}).get("description", "")
                })
                
    except Exception as e:
        logger.error(f"Error parsing game summary: {e}")
    
    return result


async def get_matchup_context(home_team: str, away_team: str, sport_key: str) -> Dict:
    """Get full matchup context including rosters and injuries for AI analysis"""
    home_roster = await fetch_team_roster(home_team, sport_key)
    away_roster = await fetch_team_roster(away_team, sport_key)
    
    return {
        "home_team": {
            "name": home_team,
            "key_players": home_roster.get("key_players", [])[:5],
            "injuries": home_roster.get("injuries", [])[:5],
            "total_players": len(home_roster.get("players", []))
        },
        "away_team": {
            "name": away_team,
            "key_players": away_roster.get("key_players", [])[:5],
            "injuries": away_roster.get("injuries", [])[:5],
            "total_players": len(away_roster.get("players", []))
        },
        "fetched_at": datetime.now(timezone.utc).isoformat()
    }
