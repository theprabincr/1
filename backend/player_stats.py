"""
Player Stats Module - Fetches and stores detailed player statistics from ESPN
Used by V6 predictor for ML-based predictions
"""
import asyncio
import logging
import httpx
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# ESPN endpoints for player stats
ESPN_GAME_SUMMARY = {
    "basketball_nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}",
    "americanfootball_nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}",
    "icehockey_nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={event_id}",
}

ESPN_TEAM_SCHEDULE = {
    "basketball_nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule",
    "americanfootball_nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/schedule",
    "icehockey_nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{team_id}/schedule",
}

# Stat labels for different sports
NBA_STAT_LABELS = ["MIN", "FG", "3PT", "FT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PTS"]
NHL_STAT_LABELS = ["G", "A", "PTS", "+/-", "PIM", "SOG", "HIT", "BLK", "FW", "FL", "TOI"]


async def fetch_player_game_stats(event_id: str, sport_key: str) -> Dict:
    """
    Fetch detailed player stats from a completed game.
    Returns stats for all players who played.
    """
    if sport_key not in ESPN_GAME_SUMMARY:
        return {}
    
    url = ESPN_GAME_SUMMARY[sport_key].format(event_id=event_id)
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(url)
            if response.status_code != 200:
                return {}
            
            data = response.json()
            boxscore = data.get("boxscore", {})
            players_sections = boxscore.get("players", [])
            
            result = {
                "event_id": event_id,
                "sport_key": sport_key,
                "game_date": data.get("header", {}).get("competitions", [{}])[0].get("date", ""),
                "teams": {}
            }
            
            for team_section in players_sections:
                team_data = team_section.get("team", {})
                team_id = team_data.get("id", "")
                team_name = team_data.get("displayName", "")
                
                statistics = team_section.get("statistics", [])
                if not statistics:
                    continue
                
                labels = statistics[0].get("labels", [])
                athletes = statistics[0].get("athletes", [])
                
                team_players = []
                
                for athlete_data in athletes:
                    athlete = athlete_data.get("athlete", {})
                    stats = athlete_data.get("stats", [])
                    
                    if not athlete.get("displayName") or not stats:
                        continue
                    
                    player_stats = {
                        "player_id": athlete.get("id", ""),
                        "name": athlete.get("displayName", ""),
                        "position": athlete.get("position", {}).get("abbreviation", "") if isinstance(athlete.get("position"), dict) else "",
                        "stats": {}
                    }
                    
                    # Map stats to labels
                    for i, label in enumerate(labels):
                        if i < len(stats):
                            stat_value = stats[i]
                            # Parse stat value
                            if stat_value == "--" or stat_value == "":
                                player_stats["stats"][label] = 0
                            elif "/" in str(stat_value):
                                # Handle fraction stats like "5/10" for FG
                                player_stats["stats"][label] = stat_value
                            elif ":" in str(stat_value):
                                # Handle time format like "32:45" for minutes
                                try:
                                    parts = str(stat_value).split(":")
                                    player_stats["stats"][label] = int(parts[0]) + (int(parts[1]) / 60 if len(parts) > 1 else 0)
                                except:
                                    player_stats["stats"][label] = stat_value
                            else:
                                try:
                                    player_stats["stats"][label] = float(stat_value) if "." in str(stat_value) else int(stat_value)
                                except:
                                    player_stats["stats"][label] = stat_value
                    
                    team_players.append(player_stats)
                
                result["teams"][team_id] = {
                    "team_name": team_name,
                    "players": team_players
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching game stats for {event_id}: {e}")
    
    return {}


async def fetch_team_player_averages(team_id: str, sport_key: str, num_games: int = 10, db=None) -> List[Dict]:
    """
    Fetch player averages from recent games for a team.
    If db is provided, stores/updates in database.
    Returns list of players with their average stats.
    """
    if sport_key not in ESPN_TEAM_SCHEDULE:
        return []
    
    schedule_url = ESPN_TEAM_SCHEDULE[sport_key].format(team_id=team_id)
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.get(schedule_url)
            if response.status_code != 200:
                return []
            
            schedule_data = response.json()
            events = schedule_data.get("events", [])
            
            # Find completed games (most recent first)
            completed_games = []
            for event in reversed(events):
                status = event.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("name", "")
                if status in ["STATUS_FINAL", "STATUS_FINAL_OT"]:
                    completed_games.append(event.get("id"))
                    if len(completed_games) >= num_games:
                        break
            
            if not completed_games:
                return []
            
            # Aggregate player stats across games
            player_totals = defaultdict(lambda: {
                "player_id": "",
                "name": "",
                "position": "",
                "games_played": 0,
                "totals": defaultdict(float),
                "stat_games": defaultdict(int)  # Track games with valid stats
            })
            
            for game_id in completed_games:
                game_stats = await fetch_player_game_stats(game_id, sport_key)
                
                if not game_stats or team_id not in game_stats.get("teams", {}):
                    continue
                
                team_stats = game_stats["teams"][team_id]
                
                for player in team_stats.get("players", []):
                    player_id = player.get("player_id", player.get("name", ""))
                    
                    player_totals[player_id]["player_id"] = player_id
                    player_totals[player_id]["name"] = player.get("name", "")
                    player_totals[player_id]["position"] = player.get("position", "")
                    player_totals[player_id]["games_played"] += 1
                    
                    for stat_key, stat_value in player.get("stats", {}).items():
                        if isinstance(stat_value, (int, float)):
                            player_totals[player_id]["totals"][stat_key] += stat_value
                            player_totals[player_id]["stat_games"][stat_key] += 1
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.2)
            
            # Calculate averages
            player_averages = []
            for player_id, data in player_totals.items():
                if data["games_played"] == 0:
                    continue
                
                averages = {}
                for stat_key, total in data["totals"].items():
                    games_with_stat = data["stat_games"].get(stat_key, 1)
                    averages[stat_key] = round(total / max(games_with_stat, 1), 1)
                
                player_avg = {
                    "player_id": data["player_id"],
                    "name": data["name"],
                    "position": data["position"],
                    "team_id": team_id,
                    "sport_key": sport_key,
                    "games_played": data["games_played"],
                    "averages": averages,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                
                player_averages.append(player_avg)
                
                # Store in database if provided
                if db is not None:
                    try:
                        await db.player_stats.update_one(
                            {"player_id": player_id, "team_id": team_id, "sport_key": sport_key},
                            {"$set": player_avg},
                            upsert=True
                        )
                    except Exception as e:
                        logger.debug(f"Could not store player stats: {e}")
            
            # Sort by points/goals (most important stat)
            if sport_key == "basketball_nba":
                player_averages.sort(key=lambda x: x["averages"].get("PTS", 0), reverse=True)
            elif sport_key == "icehockey_nhl":
                player_averages.sort(key=lambda x: x["averages"].get("PTS", 0), reverse=True)
            else:
                player_averages.sort(key=lambda x: x["averages"].get("PTS", x["averages"].get("TD", 0)), reverse=True)
            
            return player_averages
            
        except Exception as e:
            logger.error(f"Error fetching player averages for team {team_id}: {e}")
    
    return []


async def get_team_player_stats_from_db(team_id: str, sport_key: str, db) -> List[Dict]:
    """
    Get cached player stats from database.
    Returns empty list if not found or stale (>24 hours old).
    """
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        
        players = await db.player_stats.find({
            "team_id": team_id,
            "sport_key": sport_key,
            "updated_at": {"$gte": cutoff.isoformat()}
        }).to_list(50)
        
        if players:
            # Sort by points
            players.sort(key=lambda x: x.get("averages", {}).get("PTS", 0), reverse=True)
            return players
            
    except Exception as e:
        logger.debug(f"Error getting player stats from DB: {e}")
    
    return []


async def update_team_player_stats(team_id: str, team_name: str, sport_key: str, db) -> List[Dict]:
    """
    Update player stats for a team - fetches fresh data and stores in DB.
    """
    logger.info(f"📊 Updating player stats for {team_name} ({team_id})")
    
    stats = await fetch_team_player_averages(team_id, sport_key, num_games=10, db=db)
    
    if stats:
        logger.info(f"✅ Updated stats for {len(stats)} players on {team_name}")
        # Log top 3 players
        for player in stats[:3]:
            avgs = player.get("averages", {})
            logger.info(f"   {player['name']}: {avgs.get('PTS', 0)} PPG, {avgs.get('REB', 0)} RPG, {avgs.get('AST', 0)} APG")
    
    return stats


def calculate_player_impact_score(player_stats: Dict, sport_key: str) -> float:
    """
    Calculate a player's overall impact score based on their stats.
    Used by predictor to weight player importance.
    """
    avgs = player_stats.get("averages", {})
    
    if sport_key == "basketball_nba":
        # NBA impact: Points + (Rebounds * 1.2) + (Assists * 1.5) + (Steals * 2) + (Blocks * 2) - (Turnovers * 1.5)
        pts = avgs.get("PTS", 0)
        reb = avgs.get("REB", 0)
        ast = avgs.get("AST", 0)
        stl = avgs.get("STL", 0)
        blk = avgs.get("BLK", 0)
        tov = avgs.get("TO", 0)
        
        impact = pts + (reb * 1.2) + (ast * 1.5) + (stl * 2) + (blk * 2) - (tov * 1.5)
        return round(impact, 1)
    
    elif sport_key == "icehockey_nhl":
        # NHL impact: Goals * 3 + Assists * 2 + Plus/Minus + SOG * 0.5
        goals = avgs.get("G", 0)
        assists = avgs.get("A", 0)
        plus_minus = avgs.get("+/-", 0)
        sog = avgs.get("SOG", 0)
        
        impact = (goals * 3) + (assists * 2) + plus_minus + (sog * 0.5)
        return round(impact, 1)
    
    # Default: just use points
    return avgs.get("PTS", 0)


def get_starting_lineup_impact(starters: List[Dict], player_stats: List[Dict], sport_key: str) -> Dict:
    """
    Calculate the combined impact of a starting lineup.
    Compares actual starters vs expected based on player stats.
    """
    starter_names = [s.get("name", "").lower() for s in starters]
    
    # Find stats for starters
    lineup_stats = []
    total_impact = 0
    total_pts = 0
    total_reb = 0
    total_ast = 0
    
    for player in player_stats:
        if player.get("name", "").lower() in starter_names:
            impact = calculate_player_impact_score(player, sport_key)
            avgs = player.get("averages", {})
            
            lineup_stats.append({
                "name": player.get("name"),
                "position": player.get("position"),
                "impact": impact,
                "ppg": avgs.get("PTS", 0),
                "rpg": avgs.get("REB", 0),
                "apg": avgs.get("AST", 0)
            })
            
            total_impact += impact
            total_pts += avgs.get("PTS", 0)
            total_reb += avgs.get("REB", 0)
            total_ast += avgs.get("AST", 0)
    
    return {
        "starters_found": len(lineup_stats),
        "total_starters": len(starters),
        "total_impact": round(total_impact, 1),
        "projected_pts": round(total_pts, 1),
        "projected_reb": round(total_reb, 1),
        "projected_ast": round(total_ast, 1),
        "lineup_details": lineup_stats
    }


async def compare_team_stats(home_stats: List[Dict], away_stats: List[Dict], 
                            home_starters: List[Dict], away_starters: List[Dict],
                            sport_key: str) -> Dict:
    """
    Compare two teams based on their player stats.
    Returns analysis useful for predictions.
    """
    home_lineup = get_starting_lineup_impact(home_starters, home_stats, sport_key)
    away_lineup = get_starting_lineup_impact(away_starters, away_stats, sport_key)
    
    # Calculate advantages
    impact_diff = home_lineup["total_impact"] - away_lineup["total_impact"]
    pts_diff = home_lineup["projected_pts"] - away_lineup["projected_pts"]
    reb_diff = home_lineup["projected_reb"] - away_lineup["projected_reb"]
    ast_diff = home_lineup["projected_ast"] - away_lineup["projected_ast"]
    
    # Determine advantages
    advantages = []
    if abs(impact_diff) > 5:
        advantages.append(f"{'Home' if impact_diff > 0 else 'Away'} has significant lineup impact advantage ({abs(impact_diff):.1f})")
    if abs(pts_diff) > 10:
        advantages.append(f"{'Home' if pts_diff > 0 else 'Away'} starters average {abs(pts_diff):.1f} more PPG")
    if abs(reb_diff) > 5:
        advantages.append(f"{'Home' if reb_diff > 0 else 'Away'} has rebounding advantage ({abs(reb_diff):.1f} RPG)")
    if abs(ast_diff) > 3:
        advantages.append(f"{'Home' if ast_diff > 0 else 'Away'} has playmaking advantage ({abs(ast_diff):.1f} APG)")
    
    return {
        "home_lineup": home_lineup,
        "away_lineup": away_lineup,
        "impact_advantage": "home" if impact_diff > 3 else "away" if impact_diff < -3 else "even",
        "impact_diff": round(impact_diff, 1),
        "scoring_advantage": "home" if pts_diff > 5 else "away" if pts_diff < -5 else "even",
        "pts_diff": round(pts_diff, 1),
        "key_advantages": advantages
    }
