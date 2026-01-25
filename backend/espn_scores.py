"""
ESPN Scores API Integration - Free real-time scores for NBA, NFL, MLB, NHL, EPL
Uses ESPN's undocumented public API endpoints for live and final scores
"""
import asyncio
import logging
import httpx
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)

# ESPN API endpoints for each sport
ESPN_ENDPOINTS = {
    "basketball_nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "americanfootball_nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "baseball_mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
    "icehockey_nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
    "soccer_epl": "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard",
    "soccer_spain_la_liga": "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard",
    "mma_mixed_martial_arts": "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard",
}

# Game status mappings
GAME_STATUS = {
    "STATUS_SCHEDULED": "scheduled",
    "STATUS_IN_PROGRESS": "in_progress",
    "STATUS_HALFTIME": "in_progress",
    "STATUS_END_PERIOD": "in_progress",
    "STATUS_FINAL": "final",
    "STATUS_FINAL_OT": "final",
    "STATUS_POSTPONED": "postponed",
    "STATUS_CANCELED": "canceled",
    "STATUS_DELAYED": "delayed",
}


async def fetch_espn_scores(sport_key: str, days_back: int = 3) -> List[Dict]:
    """
    Fetch scores from ESPN API for a specific sport
    Returns list of game data with scores and status
    """
    if sport_key not in ESPN_ENDPOINTS:
        logger.warning(f"Sport {sport_key} not supported for ESPN scores")
        return []
    
    url = ESPN_ENDPOINTS[sport_key]
    all_games = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Fetch today's games
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                games = parse_espn_response(data, sport_key)
                all_games.extend(games)
            
            # Also fetch games from past few days for result checking
            for day_offset in range(1, days_back + 1):
                date = (datetime.now(timezone.utc) - timedelta(days=day_offset)).strftime("%Y%m%d")
                dated_url = f"{url}?dates={date}"
                
                try:
                    response = await client.get(dated_url)
                    if response.status_code == 200:
                        data = response.json()
                        games = parse_espn_response(data, sport_key)
                        all_games.extend(games)
                except Exception as e:
                    logger.error(f"Error fetching ESPN scores for {date}: {e}")
                
                await asyncio.sleep(0.2)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Error fetching ESPN scores for {sport_key}: {e}")
    
    return all_games


def parse_espn_response(data: dict, sport_key: str) -> List[Dict]:
    """Parse ESPN API response into standardized game format"""
    games = []
    
    events = data.get("events", [])
    
    for event in events:
        try:
            game_id = event.get("id", "")
            game_name = event.get("name", "")
            game_date = event.get("date", "")
            
            # Get status
            status_obj = event.get("status", {})
            status_type = status_obj.get("type", {}).get("name", "")
            status = GAME_STATUS.get(status_type, "unknown")
            clock = status_obj.get("displayClock", "")
            period = status_obj.get("period", 0)
            
            # Get competitors (teams)
            competitions = event.get("competitions", [])
            if not competitions:
                continue
            
            competition = competitions[0]
            competitors = competition.get("competitors", [])
            
            if len(competitors) < 2:
                continue
            
            # ESPN returns home team first, then away team
            home_team_data = None
            away_team_data = None
            
            for comp in competitors:
                if comp.get("homeAway") == "home":
                    home_team_data = comp
                elif comp.get("homeAway") == "away":
                    away_team_data = comp
            
            if not home_team_data or not away_team_data:
                # Fallback to position-based
                home_team_data = competitors[0]
                away_team_data = competitors[1]
            
            home_team = home_team_data.get("team", {}).get("displayName", "")
            away_team = away_team_data.get("team", {}).get("displayName", "")
            
            home_score = int(home_team_data.get("score", 0) or 0)
            away_score = int(away_team_data.get("score", 0) or 0)
            
            # Get winner if game is final
            winner = None
            if status == "final":
                if home_score > away_score:
                    winner = home_team
                elif away_score > home_score:
                    winner = away_team
                else:
                    winner = "draw"  # For soccer ties
            
            # Calculate total score
            total_score = home_score + away_score
            
            game = {
                "espn_id": game_id,
                "sport_key": sport_key,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "total_score": total_score,
                "winner": winner,
                "status": status,
                "clock": clock,
                "period": period,
                "game_date": game_date,
                "game_name": game_name,
            }
            
            games.append(game)
            
        except Exception as e:
            logger.error(f"Error parsing ESPN event: {e}")
            continue
    
    return games


def match_teams(prediction_team: str, espn_team: str) -> bool:
    """
    Check if prediction team name matches ESPN team name
    Handles variations like "LA Lakers" vs "Los Angeles Lakers"
    """
    if not prediction_team or not espn_team:
        return False
    
    # Normalize both strings
    pred = normalize_team_name(prediction_team)
    espn = normalize_team_name(espn_team)
    
    # Direct match
    if pred == espn:
        return True
    
    # Check if one contains the other
    if pred in espn or espn in pred:
        return True
    
    # Check key words match (e.g., "Lakers", "Celtics")
    pred_words = set(pred.split())
    espn_words = set(espn.split())
    
    # Common words that should match
    team_keywords = pred_words & espn_words
    if len(team_keywords) >= 1:
        # Exclude common words like "city", "state"
        important_words = team_keywords - {"city", "state", "new", "los", "san", "la"}
        if important_words:
            return True
    
    return False


def normalize_team_name(name: str) -> str:
    """Normalize team name for comparison"""
    name = name.lower().strip()
    
    # Common abbreviations and variations
    replacements = {
        "la lakers": "los angeles lakers",
        "la clippers": "los angeles clippers",
        "ny knicks": "new york knicks",
        "ny jets": "new york jets",
        "ny giants": "new york giants",
        "philly": "philadelphia",
        "76ers": "seventy sixers",
        "man city": "manchester city",
        "man utd": "manchester united",
        "man united": "manchester united",
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Remove common suffixes/prefixes
    name = re.sub(r'\b(fc|sc|cf|afc|united|city)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def determine_bet_result(
    prediction_type: str,
    predicted_outcome: str,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    total_score: int
) -> str:
    """
    Determine if a bet won, lost, or pushed based on the game result
    
    Args:
        prediction_type: 'moneyline', 'spread', or 'total'
        predicted_outcome: The predicted outcome (team name, spread, or over/under)
        home_team: Home team name
        away_team: Away team name
        home_score: Final home score
        away_score: Final away score
        total_score: Combined final score
    
    Returns:
        'win', 'loss', or 'push'
    """
    prediction_type = prediction_type.lower()
    predicted_outcome_lower = predicted_outcome.lower()
    
    if prediction_type == "moneyline":
        # Determine winner
        if home_score > away_score:
            actual_winner = home_team
        elif away_score > home_score:
            actual_winner = away_team
        else:
            return "push"  # Tie game
        
        # Check if predicted team won
        if match_teams(predicted_outcome, actual_winner):
            return "win"
        else:
            return "loss"
    
    elif prediction_type == "spread":
        # Parse spread from predicted_outcome (e.g., "Boston Celtics -5.5")
        spread_match = re.search(r'([+-]?\d+\.?\d*)\s*$', predicted_outcome)
        
        if not spread_match:
            logger.warning(f"Could not parse spread from: {predicted_outcome}")
            return "pending"
        
        spread = float(spread_match.group(1))
        
        # Determine which team was bet on
        if match_teams(predicted_outcome, home_team):
            # Bet on home team with spread
            adjusted_home = home_score + spread
            if adjusted_home > away_score:
                return "win"
            elif adjusted_home < away_score:
                return "loss"
            else:
                return "push"
        else:
            # Bet on away team with spread
            adjusted_away = away_score + spread
            if adjusted_away > home_score:
                return "win"
            elif adjusted_away < home_score:
                return "loss"
            else:
                return "push"
    
    elif prediction_type == "total":
        # Parse total from predicted_outcome (e.g., "Over 225.5" or "Under 225.5")
        total_match = re.search(r'(over|under)\s*(\d+\.?\d*)', predicted_outcome_lower)
        
        if not total_match:
            logger.warning(f"Could not parse total from: {predicted_outcome}")
            return "pending"
        
        direction = total_match.group(1)
        line = float(total_match.group(2))
        
        if direction == "over":
            if total_score > line:
                return "win"
            elif total_score < line:
                return "loss"
            else:
                return "push"
        else:  # under
            if total_score < line:
                return "win"
            elif total_score > line:
                return "loss"
            else:
                return "push"
    
    return "pending"


async def find_matching_game(
    sport_key: str,
    home_team: str,
    away_team: str,
    commence_time: str,
    games: List[Dict] = None
) -> Optional[Dict]:
    """
    Find a matching game from ESPN scores for a prediction
    
    Args:
        sport_key: Sport identifier
        home_team: Home team name from prediction
        away_team: Away team name from prediction
        commence_time: Event start time
        games: Optional pre-fetched games list
    
    Returns:
        Matching game dict or None
    """
    if games is None:
        games = await fetch_espn_scores(sport_key, days_back=5)
    
    for game in games:
        # Check if teams match (in either order)
        home_match = match_teams(home_team, game["home_team"]) or match_teams(home_team, game["away_team"])
        away_match = match_teams(away_team, game["away_team"]) or match_teams(away_team, game["home_team"])
        
        if home_match and away_match:
            # Verify date is close (within 1 day)
            try:
                pred_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                game_time = datetime.fromisoformat(game["game_date"].replace('Z', '+00:00'))
                
                time_diff = abs((pred_time - game_time).total_seconds())
                if time_diff < 86400 * 2:  # Within 2 days
                    return game
            except Exception:
                # If date parsing fails, still return match
                return game
    
    return None


async def get_live_games(sport_key: str) -> List[Dict]:
    """Get currently in-progress games for a sport"""
    games = await fetch_espn_scores(sport_key, days_back=0)
    return [g for g in games if g["status"] == "in_progress"]


async def get_final_games(sport_key: str, days_back: int = 3) -> List[Dict]:
    """Get completed games for a sport"""
    games = await fetch_espn_scores(sport_key, days_back=days_back)
    return [g for g in games if g["status"] == "final"]
