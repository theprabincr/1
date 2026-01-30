"""
Lineup and Roster Scraper - Fetches playing squads and injury info from ESPN
Uses ESPN's free hidden API endpoints for roster/lineup data
"""
import asyncio
import logging
import httpx
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# ESPN API endpoints for rosters and game summaries
ESPN_ROSTER_ENDPOINTS = {
    "basketball_nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster",
    "americanfootball_nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster",
    "baseball_mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams/{team_id}/roster",
    "icehockey_nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{team_id}/roster",
    "soccer_epl": "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/teams/{team_id}/roster",
}

ESPN_TEAM_SCHEDULE = {
    "basketball_nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule",
    "americanfootball_nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/schedule",
    "baseball_mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams/{team_id}/schedule",
    "icehockey_nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{team_id}/schedule",
    "soccer_epl": "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/teams/{team_id}/schedule",
}

ESPN_GAME_SUMMARY = {
    "basketball_nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}",
    "americanfootball_nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}",
    "baseball_mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event={event_id}",
    "icehockey_nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={event_id}",
    "soccer_epl": "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/summary?event={event_id}",
}

ESPN_DEPTH_CHART = {
    "basketball_nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/depthcharts",
    "americanfootball_nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/depthcharts",
    "icehockey_nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{team_id}/depthcharts",
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


async def fetch_projected_starters(team_id: str, sport_key: str) -> List[Dict]:
    """
    Fetch projected starters from ESPN depth chart.
    Returns list of projected starting 5 players (for basketball).
    """
    if sport_key not in ESPN_DEPTH_CHART:
        return []
    
    url = ESPN_DEPTH_CHART[sport_key].format(team_id=team_id)
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(url)
            if response.status_code != 200:
                return []
            
            data = response.json()
            depthchart = data.get("depthchart", [])
            
            if not depthchart:
                return []
            
            # Get positions from first depth chart
            positions = depthchart[0].get("positions", {})
            
            starters = []
            
            # Basketball positions: PG, SG, SF, PF, C
            if "nba" in sport_key or "basketball" in sport_key:
                position_order = ["pg", "sg", "sf", "pf", "c"]
            # Hockey positions: C, LW, RW, D, G
            elif "nhl" in sport_key or "hockey" in sport_key:
                position_order = ["c", "lw", "rw", "ld", "rd", "g"]
            # Football starting positions
            elif "nfl" in sport_key or "football" in sport_key:
                position_order = ["qb", "rb", "wr1", "wr2", "te"]
            else:
                position_order = list(positions.keys())[:5]
            
            for pos_key in position_order:
                pos_data = positions.get(pos_key, {})
                athletes = pos_data.get("athletes", [])
                
                if athletes:
                    # First athlete at each position is the starter
                    starter = athletes[0]
                    position_info = pos_data.get("position", {})
                    starters.append({
                        "name": starter.get("displayName", ""),
                        "position": position_info.get("abbreviation", pos_key.upper())
                    })
            
            return starters[:5]  # Return top 5 starters
            
        except Exception as e:
            logger.error(f"Error fetching depth chart for team {team_id}: {e}")
    
    return []


async def fetch_team_roster(team_name: str, sport_key: str) -> Dict:
    """Fetch team roster from ESPN with top performers based on recent game stats"""
    team_id = get_team_id(team_name, sport_key)
    
    if not team_id or sport_key not in ESPN_ROSTER_ENDPOINTS:
        return {"team": team_name, "players": [], "injuries": [], "key_players": []}
    
    url = ESPN_ROSTER_ENDPOINTS[sport_key].format(team_id=team_id)
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                result = parse_roster_response(data, team_name)
                
                # Fetch top performers based on recent game stats
                try:
                    top_performers = await fetch_top_performers(team_id, sport_key, num_games=5)
                    if top_performers:
                        # Replace key_players with actual top performers
                        result["key_players"] = [
                            f"{p['name']} ({p['avg_pts']} PPG)" for p in top_performers[:5]
                        ]
                        result["top_performers"] = top_performers[:5]
                        logger.info(f"✅ Found top performers for {team_name}: {[p['name'] for p in top_performers[:3]]}")
                except Exception as e:
                    logger.debug(f"Could not fetch top performers for {team_name}: {e}")
                
                return result
        except Exception as e:
            logger.error(f"Error fetching roster for {team_name}: {e}")
    
    return {"team": team_name, "players": [], "injuries": [], "key_players": []}


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
        
        # ESPN now returns a flat array of athletes
        for player in athletes:
            if isinstance(player, dict):
                # Get position
                position_data = player.get("position", {})
                position = position_data.get("abbreviation", "") if isinstance(position_data, dict) else ""
                
                player_info = {
                    "name": player.get("displayName", player.get("fullName", "")),
                    "position": position,
                    "jersey": player.get("jersey", ""),
                    "status": "active"
                }
                
                # Check for injuries
                injuries = player.get("injuries", [])
                if injuries:
                    injury = injuries[0] if isinstance(injuries[0], dict) else {}
                    player_info["status"] = injury.get("status", "questionable")
                    injury_type = injury.get("type", {})
                    if isinstance(injury_type, dict):
                        player_info["injury"] = injury_type.get("description", "")
                    result["injuries"].append({
                        "name": player_info["name"],
                        "position": position,
                        "status": player_info["status"],
                        "injury": player_info.get("injury", "Unknown")
                    })
                
                result["players"].append(player_info)
                
                # Mark key players (starters based on position)
                key_positions = ["PG", "SG", "SF", "PF", "C", "G", "F",  # Basketball
                                "QB", "RB", "WR", "TE",  # Football
                                "G", "D", "C",  # Hockey
                                "GK", "DF", "MF", "FW"]  # Soccer
                if position in key_positions:
                    result["key_players"].append(player_info["name"])
        
        # Limit key players to top 10
        result["key_players"] = result["key_players"][:10]
        
    except Exception as e:
        logger.error(f"Error parsing roster: {e}")
    
    return result


async def fetch_top_performers(team_id: str, sport_key: str, num_games: int = 5) -> List[Dict]:
    """
    Fetch top performing players from recent completed games.
    Returns players sorted by average points/goals scored.
    """
    if sport_key not in ESPN_TEAM_SCHEDULE or sport_key not in ESPN_GAME_SUMMARY:
        return []
    
    try:
        # Get team schedule to find recent completed games
        schedule_url = ESPN_TEAM_SCHEDULE[sport_key].format(team_id=team_id)
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(schedule_url)
            if response.status_code != 200:
                return []
            
            schedule_data = response.json()
            events = schedule_data.get("events", [])
            
            # Find completed games (most recent first)
            completed_game_ids = []
            for event in reversed(events):  # Reverse to get most recent first
                status = event.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("name", "")
                if status in ["STATUS_FINAL", "STATUS_FINAL_OT"]:
                    completed_game_ids.append(event.get("id"))
                    if len(completed_game_ids) >= num_games:
                        break
            
            if not completed_game_ids:
                return []
            
            # Fetch boxscores for these games
            player_stats = defaultdict(lambda: {"name": "", "position": "", "games": 0, "total_pts": 0, "total_mins": 0})
            
            for game_id in completed_game_ids:
                summary_url = ESPN_GAME_SUMMARY[sport_key].format(event_id=game_id)
                try:
                    summary_response = await client.get(summary_url)
                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        boxscore = summary_data.get("boxscore", {})
                        players_sections = boxscore.get("players", [])
                        
                        for team_section in players_sections:
                            section_team_id = team_section.get("team", {}).get("id", "")
                            if section_team_id != team_id:
                                continue
                            
                            stats = team_section.get("statistics", [])
                            if not stats:
                                continue
                            
                            labels = stats[0].get("labels", [])
                            pts_idx = labels.index("PTS") if "PTS" in labels else -1
                            min_idx = labels.index("MIN") if "MIN" in labels else -1
                            
                            for athlete_data in stats[0].get("athletes", []):
                                athlete = athlete_data.get("athlete", {})
                                player_name = athlete.get("displayName", "")
                                player_id = athlete.get("id", player_name)
                                position = athlete.get("position", {}).get("abbreviation", "") if isinstance(athlete.get("position"), dict) else ""
                                stat_values = athlete_data.get("stats", [])
                                
                                if not player_name or not stat_values:
                                    continue
                                
                                pts = 0
                                mins = 0
                                try:
                                    if pts_idx >= 0 and len(stat_values) > pts_idx:
                                        pts = int(stat_values[pts_idx]) if stat_values[pts_idx] != "--" else 0
                                    if min_idx >= 0 and len(stat_values) > min_idx:
                                        min_str = stat_values[min_idx]
                                        if min_str and min_str != "--":
                                            mins = int(min_str.split(":")[0]) if ":" in str(min_str) else int(float(min_str))
                                except (ValueError, IndexError):
                                    pass
                                
                                player_stats[player_id]["name"] = player_name
                                player_stats[player_id]["position"] = position
                                player_stats[player_id]["games"] += 1
                                player_stats[player_id]["total_pts"] += pts
                                player_stats[player_id]["total_mins"] += mins
                                
                except Exception as e:
                    logger.debug(f"Error fetching game {game_id}: {e}")
                    continue
            
            # Calculate averages and sort by avg points
            top_performers = []
            for player_id, stats in player_stats.items():
                if stats["games"] > 0 and stats["total_mins"] > 0:  # Must have played
                    avg_pts = stats["total_pts"] / stats["games"]
                    avg_mins = stats["total_mins"] / stats["games"]
                    top_performers.append({
                        "name": stats["name"],
                        "position": stats["position"],
                        "avg_pts": round(avg_pts, 1),
                        "avg_mins": round(avg_mins, 1),
                        "games": stats["games"]
                    })
            
            # Sort by average points (descending)
            top_performers.sort(key=lambda x: x["avg_pts"], reverse=True)
            
            return top_performers[:10]  # Return top 10
            
    except Exception as e:
        logger.error(f"Error fetching top performers for team {team_id}: {e}")
    
    return []


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


async def fetch_starting_lineup(espn_event_id: str, sport_key: str) -> Dict:
    """
    Fetch confirmed starting lineup for a game.
    ESPN typically releases lineups 1 hour before game time.
    
    Returns:
        Dict with home/away starters, or empty if not yet announced
    """
    result = {
        "home": {
            "team": "",
            "starters": [],
            "confirmed": False
        },
        "away": {
            "team": "",
            "starters": [],
            "confirmed": False
        },
        "lineup_status": "not_available",
        "message": "Starting lineups not yet announced"
    }
    
    if sport_key not in ESPN_GAME_SUMMARY:
        return result
    
    url = ESPN_GAME_SUMMARY[sport_key].format(event_id=espn_event_id)
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(url)
            if response.status_code != 200:
                return result
            
            data = response.json()
            
            # Check game status
            header = data.get("header", {})
            competitions = header.get("competitions", [{}])
            if competitions:
                status = competitions[0].get("status", {})
                status_type = status.get("type", {}).get("name", "")
                
                # If game is in progress or final, starters are from boxscore
                if status_type in ["STATUS_IN_PROGRESS", "STATUS_HALFTIME", "STATUS_FINAL", "STATUS_FINAL_OT"]:
                    result["lineup_status"] = "confirmed"
                    result["message"] = "Lineup confirmed (game started)"
                elif status_type == "STATUS_SCHEDULED":
                    # Check if we have projected starters
                    pass
            
            # Parse boxscore for starters
            boxscore = data.get("boxscore", {})
            
            # Get team names
            teams = boxscore.get("teams", [])
            for team in teams:
                team_data = team.get("team", {})
                team_name = team_data.get("displayName", "")
                home_away = team.get("homeAway", "")
                
                if home_away == "home":
                    result["home"]["team"] = team_name
                else:
                    result["away"]["team"] = team_name
            
            # Parse players - starters typically listed first or have "starter" flag
            players_data = boxscore.get("players", [])
            
            for team_players in players_data:
                home_away = team_players.get("homeAway", "")
                team_key = "home" if home_away == "home" else "away"
                
                statistics = team_players.get("statistics", [])
                
                for stat_cat in statistics:
                    athletes = stat_cat.get("athletes", [])
                    
                    starters_found = []
                    for athlete in athletes:
                        athlete_info = athlete.get("athlete", {})
                        player_name = athlete_info.get("displayName", "")
                        position = athlete_info.get("position", {}).get("abbreviation", "")
                        
                        # Check if marked as starter
                        is_starter = athlete.get("starter", False)
                        
                        # Also check stats - starters typically have more minutes
                        stats = athlete.get("stats", [])
                        minutes = 0
                        if stats and len(stats) > 0:
                            try:
                                # Minutes is usually first stat in NBA
                                min_str = stats[0] if stats else "0"
                                if ":" in str(min_str):
                                    parts = str(min_str).split(":")
                                    minutes = int(parts[0])
                                else:
                                    minutes = int(float(min_str)) if min_str else 0
                            except (ValueError, TypeError):
                                minutes = 0
                        
                        if player_name:
                            starters_found.append({
                                "name": player_name,
                                "position": position,
                                "is_starter": is_starter,
                                "minutes": minutes
                            })
                    
                    # Sort by starter flag, then by minutes
                    starters_found.sort(key=lambda x: (-int(x["is_starter"]), -x["minutes"]))
                    
                    # Take top 5 (typical starting lineup size)
                    for starter in starters_found[:5]:
                        result[team_key]["starters"].append({
                            "name": starter["name"],
                            "position": starter["position"]
                        })
                    
                    if result[team_key]["starters"]:
                        result[team_key]["confirmed"] = True
                        result["lineup_status"] = "confirmed"
                        result["message"] = "Starting lineup confirmed"
                    
                    break  # Only process first stat category
            
            # If no starters from boxscore, try to get projected starters
            if not result["home"]["starters"] and not result["away"]["starters"]:
                # Check for roster data with depth chart info
                rosters = data.get("rosters", [])
                for roster in rosters:
                    home_away = roster.get("homeAway", "")
                    team_key = "home" if home_away == "home" else "away"
                    
                    entries = roster.get("roster", [])
                    starters = []
                    
                    for entry in entries:
                        if entry.get("starter", False) or entry.get("position", {}).get("name", "").lower() == "starter":
                            player = entry.get("athlete", {})
                            starters.append({
                                "name": player.get("displayName", ""),
                                "position": entry.get("position", {}).get("abbreviation", "")
                            })
                    
                    if starters:
                        result[team_key]["starters"] = starters[:5]
                        result[team_key]["confirmed"] = False  # Projected, not confirmed
                        result["lineup_status"] = "projected"
                        result["message"] = "Projected starting lineup (not yet confirmed)"
            
        except Exception as e:
            logger.error(f"Error fetching starting lineup for {espn_event_id}: {e}")
    
    return result


async def get_full_roster_with_starters(team_name: str, sport_key: str, espn_event_id: str = None) -> Dict:
    """
    Get full team roster with starting lineup info when available.
    
    Args:
        team_name: Team name
        sport_key: Sport key
        espn_event_id: Optional event ID to get game-specific starters
    
    Returns:
        Dict with roster, starters, injuries, and key players
    """
    # Get base roster
    roster = await fetch_team_roster(team_name, sport_key)
    
    result = {
        "team": team_name,
        "roster": roster.get("players", []),
        "injuries": roster.get("injuries", []),
        "key_players": roster.get("key_players", []),
        "starters": [],
        "starters_confirmed": False
    }
    
    # If we have an event ID, try to get confirmed starters
    if espn_event_id:
        lineup = await fetch_starting_lineup(espn_event_id, sport_key)
        
        # Determine if this team is home or away
        if team_name.lower() in lineup.get("home", {}).get("team", "").lower():
            result["starters"] = lineup["home"]["starters"]
            result["starters_confirmed"] = lineup["home"]["confirmed"]
        elif team_name.lower() in lineup.get("away", {}).get("team", "").lower():
            result["starters"] = lineup["away"]["starters"]
            result["starters_confirmed"] = lineup["away"]["confirmed"]
    
    return result
