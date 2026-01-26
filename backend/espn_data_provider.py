"""
ESPN Data Provider - Real odds, scores, team stats, and betting data
Uses ESPN's hidden API for comprehensive sports data
"""
import asyncio
import logging
import httpx
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)

# ESPN API Base URLs
ESPN_BASE = "http://site.api.espn.com/apis/site/v2/sports"

# Sport configurations
SPORT_CONFIG = {
    "basketball_nba": {"sport": "basketball", "league": "nba"},
    "americanfootball_nfl": {"sport": "football", "league": "nfl"},
    "baseball_mlb": {"sport": "baseball", "league": "mlb"},
    "icehockey_nhl": {"sport": "hockey", "league": "nhl"},
    "soccer_epl": {"sport": "soccer", "league": "eng.1"},
    "soccer_spain_la_liga": {"sport": "soccer", "league": "esp.1"},
}

# Game status mapping
STATUS_MAP = {
    "STATUS_SCHEDULED": "scheduled",
    "STATUS_IN_PROGRESS": "in_progress",
    "STATUS_HALFTIME": "in_progress",
    "STATUS_END_PERIOD": "in_progress",
    "STATUS_FINAL": "final",
    "STATUS_FINAL_OT": "final",
    "STATUS_POSTPONED": "postponed",
    "STATUS_CANCELED": "canceled",
}


async def fetch_espn_events_with_odds(sport_key: str, days_ahead: int = 3) -> List[Dict]:
    """
    Fetch upcoming events with REAL odds from ESPN API
    Returns events with commence_time, odds (spread, ML, total), teams, venue
    """
    if sport_key not in SPORT_CONFIG:
        logger.warning(f"Sport {sport_key} not configured for ESPN")
        return []
    
    config = SPORT_CONFIG[sport_key]
    url = f"{ESPN_BASE}/{config['sport']}/{config['league']}/scoreboard"
    
    events = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Fetch yesterday (for late-night games), today and upcoming days
            for day_offset in range(-1, days_ahead + 1):
                date = (datetime.now(timezone.utc) + timedelta(days=day_offset)).strftime("%Y%m%d")
                params = {"dates": date}
                
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    parsed_events = parse_espn_events(data, sport_key)
                    events.extend(parsed_events)
                
                await asyncio.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error fetching ESPN events for {sport_key}: {e}")
    
    # Remove duplicates by event ID
    seen = set()
    unique_events = []
    for event in events:
        if event["id"] not in seen:
            seen.add(event["id"])
            unique_events.append(event)
    
    return unique_events


def parse_espn_events(data: dict, sport_key: str) -> List[Dict]:
    """Parse ESPN scoreboard response into standardized event format"""
    events = []
    
    for espn_event in data.get("events", []):
        try:
            event_id = espn_event.get("id", "")
            event_name = espn_event.get("name", "")
            event_date = espn_event.get("date", "")
            
            # Get status
            status_obj = espn_event.get("status", {})
            status_type = status_obj.get("type", {}).get("name", "")
            status = STATUS_MAP.get(status_type, "unknown")
            
            # Only include scheduled (pre-match) events
            if status != "scheduled":
                continue
            
            # Get competition details
            competitions = espn_event.get("competitions", [])
            if not competitions:
                continue
            
            competition = competitions[0]
            competitors = competition.get("competitors", [])
            
            if len(competitors) < 2:
                continue
            
            # Get home and away teams
            home_team_data = None
            away_team_data = None
            
            for comp in competitors:
                if comp.get("homeAway") == "home":
                    home_team_data = comp
                else:
                    away_team_data = comp
            
            if not home_team_data or not away_team_data:
                continue
            
            home_team = home_team_data.get("team", {}).get("displayName", "")
            away_team = away_team_data.get("team", {}).get("displayName", "")
            home_team_id = home_team_data.get("team", {}).get("id", "")
            away_team_id = away_team_data.get("team", {}).get("id", "")
            
            # Get venue info
            venue = competition.get("venue", {})
            venue_name = venue.get("fullName", "")
            venue_city = venue.get("address", {}).get("city", "")
            
            # Get REAL odds from ESPN (from DraftKings/other providers)
            odds_data = parse_espn_odds(competition.get("odds", []))
            
            # Get team records
            home_record = home_team_data.get("records", [{}])[0].get("summary", "") if home_team_data.get("records") else ""
            away_record = away_team_data.get("records", [{}])[0].get("summary", "") if away_team_data.get("records") else ""
            
            # Build bookmakers list from ESPN odds
            bookmakers = []
            if odds_data:
                bookmakers.append({
                    "key": odds_data.get("provider_key", "draftkings"),
                    "title": odds_data.get("provider_name", "DraftKings"),
                    "last_update": datetime.now(timezone.utc).isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": datetime.now(timezone.utc).isoformat(),
                            "outcomes": [
                                {"name": home_team, "price": odds_data.get("home_ml_decimal", 1.91)},
                                {"name": away_team, "price": odds_data.get("away_ml_decimal", 1.91)}
                            ]
                        },
                        {
                            "key": "spreads",
                            "last_update": datetime.now(timezone.utc).isoformat(),
                            "outcomes": [
                                {"name": home_team, "price": 1.91, "point": odds_data.get("spread", 0)},
                                {"name": away_team, "price": 1.91, "point": -odds_data.get("spread", 0)}
                            ]
                        },
                        {
                            "key": "totals",
                            "last_update": datetime.now(timezone.utc).isoformat(),
                            "outcomes": [
                                {"name": "Over", "price": 1.91, "point": odds_data.get("total", 220)},
                                {"name": "Under", "price": 1.91, "point": odds_data.get("total", 220)}
                            ]
                        }
                    ]
                })
            
            event = {
                "id": event_id,
                "espn_id": event_id,
                "sport_key": sport_key,
                "sport_title": sport_key.replace("_", " ").title(),
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "commence_time": event_date,
                "status": status,
                "venue": {
                    "name": venue_name,
                    "city": venue_city
                },
                "home_record": home_record,
                "away_record": away_record,
                "bookmakers": bookmakers,
                "odds": odds_data,
                "source": "espn",
                "scraped_at": datetime.now(timezone.utc).isoformat()
            }
            
            events.append(event)
            
        except Exception as e:
            logger.error(f"Error parsing ESPN event: {e}")
            continue
    
    return events


def parse_espn_odds(odds_list: list) -> Optional[Dict]:
    """Parse ESPN odds data (from DraftKings/other providers)"""
    if not odds_list:
        return None
    
    # Get first provider (usually DraftKings)
    odds = odds_list[0]
    
    provider = odds.get("provider", {})
    provider_name = provider.get("displayName", "DraftKings")
    provider_key = provider.get("name", "draftkings").lower().replace(" ", "_")
    
    # Get spread
    spread = odds.get("spread", 0)
    
    # Get over/under (total)
    total = odds.get("overUnder", 220)
    
    # Get moneyline odds
    home_ml = None
    away_ml = None
    
    home_odds_data = odds.get("homeTeamOdds", {})
    away_odds_data = odds.get("awayTeamOdds", {})
    
    # Try to get ML from moneyline section
    moneyline = odds.get("moneyline", {})
    if moneyline:
        home_ml_data = moneyline.get("home", {}).get("close", {})
        away_ml_data = moneyline.get("away", {}).get("close", {})
        
        home_ml = home_ml_data.get("odds", "")
        away_ml = away_ml_data.get("odds", "")
    
    # Convert American odds to decimal
    home_ml_decimal = american_to_decimal(home_ml) if home_ml else calculate_ml_from_spread(spread, True)
    away_ml_decimal = american_to_decimal(away_ml) if away_ml else calculate_ml_from_spread(spread, False)
    
    # Determine favorite
    home_favorite = home_odds_data.get("favorite", spread < 0)
    
    return {
        "provider_name": provider_name,
        "provider_key": provider_key,
        "spread": spread,
        "total": total,
        "home_ml_american": home_ml,
        "away_ml_american": away_ml,
        "home_ml_decimal": home_ml_decimal,
        "away_ml_decimal": away_ml_decimal,
        "home_favorite": home_favorite,
        "details": odds.get("details", "")
    }


def american_to_decimal(american_odds: str) -> float:
    """Convert American odds to decimal odds"""
    if not american_odds:
        return 1.91
    
    try:
        # Remove any non-numeric characters except minus sign
        odds_str = re.sub(r'[^\d\-]', '', str(american_odds))
        odds = int(odds_str)
        
        if odds > 0:
            # Positive odds: decimal = (american / 100) + 1
            return round((odds / 100) + 1, 2)
        else:
            # Negative odds: decimal = (100 / abs(american)) + 1
            return round((100 / abs(odds)) + 1, 2)
    except (ValueError, ZeroDivisionError):
        return 1.91


def calculate_ml_from_spread(spread: float, is_home: bool) -> float:
    """Estimate moneyline from spread when ML not available"""
    # Rough conversion: each point of spread ≈ 5-7% probability difference
    # For NBA: 1 point spread ≈ 3% probability
    
    abs_spread = abs(spread)
    
    # Base probability adjustment per point
    prob_per_point = 0.03  # 3% per point
    
    if spread < 0 and is_home:
        # Home team is favorite
        implied_prob = 0.50 + (abs_spread * prob_per_point)
    elif spread > 0 and is_home:
        # Home team is underdog
        implied_prob = 0.50 - (abs_spread * prob_per_point)
    elif spread < 0 and not is_home:
        # Away team is underdog (home favorite)
        implied_prob = 0.50 - (abs_spread * prob_per_point)
    else:
        # Away team is favorite
        implied_prob = 0.50 + (abs_spread * prob_per_point)
    
    # Clamp probability
    implied_prob = max(0.10, min(0.90, implied_prob))
    
    # Convert to decimal odds with ~5% juice
    decimal_odds = round(1 / (implied_prob * 0.95), 2)
    
    return decimal_odds


async def fetch_team_stats(team_id: str, sport_key: str) -> Dict:
    """Fetch team statistics including recent form"""
    if sport_key not in SPORT_CONFIG:
        return {}
    
    config = SPORT_CONFIG[sport_key]
    url = f"{ESPN_BASE}/{config['sport']}/{config['league']}/teams/{team_id}"
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return parse_team_stats(data)
        except Exception as e:
            logger.error(f"Error fetching team stats for {team_id}: {e}")
    
    return {}


def parse_team_stats(data: dict) -> Dict:
    """Parse team statistics from ESPN"""
    team = data.get("team", {})
    
    stats = {
        "team_id": team.get("id", ""),
        "name": team.get("displayName", ""),
        "abbreviation": team.get("abbreviation", ""),
        "record": "",
        "standing": "",
        "home_record": "",
        "away_record": "",
        "streak": "",
        "last_5": []
    }
    
    # Get record from various places
    record_items = team.get("record", {}).get("items", [])
    for item in record_items:
        summary = item.get("summary", "")
        stat_type = item.get("type", "")
        
        if stat_type == "total":
            stats["record"] = summary
        elif stat_type == "home":
            stats["home_record"] = summary
        elif stat_type == "road" or stat_type == "away":
            stats["away_record"] = summary
    
    # Get standing
    standing_summary = team.get("standingSummary", "")
    stats["standing"] = standing_summary
    
    return stats


async def fetch_team_recent_games(team_id: str, sport_key: str, num_games: int = 10) -> List[Dict]:
    """Fetch team's recent game results"""
    if sport_key not in SPORT_CONFIG:
        return []
    
    config = SPORT_CONFIG[sport_key]
    url = f"{ESPN_BASE}/{config['sport']}/{config['league']}/teams/{team_id}/schedule"
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return parse_recent_games(data, team_id, num_games)
        except Exception as e:
            logger.error(f"Error fetching recent games for {team_id}: {e}")
    
    return []


def parse_recent_games(data: dict, team_id: str, num_games: int) -> List[Dict]:
    """Parse recent game results"""
    games = []
    
    events = data.get("events", [])
    
    for event in events:
        try:
            status = event.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("name", "")
            
            if status not in ["STATUS_FINAL", "STATUS_FINAL_OT"]:
                continue
            
            competition = event.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])
            
            our_team = None
            opponent = None
            
            for comp in competitors:
                if comp.get("id") == team_id:
                    our_team = comp
                else:
                    opponent = comp
            
            if not our_team or not opponent:
                continue
            
            is_home = our_team.get("homeAway") == "home"
            
            # Handle score which can be string, int, or dict with 'value' field
            our_score_raw = our_team.get("score", 0)
            opp_score_raw = opponent.get("score", 0)
            
            # Extract score value from dict if needed
            def extract_score(score_val):
                if isinstance(score_val, dict):
                    # ESPN returns {'value': 115.0, 'displayValue': '115'}
                    return int(float(score_val.get('value', 0) or 0))
                elif isinstance(score_val, (int, float)):
                    return int(score_val)
                elif isinstance(score_val, str) and score_val.replace('.','').isdigit():
                    return int(float(score_val))
                return 0
            
            our_score = extract_score(our_score_raw)
            opp_score = extract_score(opp_score_raw)
            
            won = our_score > opp_score
            
            games.append({
                "date": event.get("date", ""),
                "opponent": opponent.get("team", {}).get("displayName", ""),
                "is_home": is_home,
                "our_score": our_score,
                "opponent_score": opp_score,
                "won": won,
                "margin": our_score - opp_score
            })
            
            if len(games) >= num_games:
                break
                
        except Exception as e:
            logger.error(f"Error parsing game: {e}")
            continue
    
    return games


async def get_comprehensive_matchup_data(event: Dict, sport_key: str) -> Dict:
    """
    Get all data needed for betting algorithm:
    - Team stats
    - Recent form (last 10 games)
    - Home/away records
    - Head-to-head (if available)
    - Injuries
    """
    home_team_id = event.get("home_team_id", "")
    away_team_id = event.get("away_team_id", "")
    
    # Fetch data in parallel
    home_stats_task = fetch_team_stats(home_team_id, sport_key)
    away_stats_task = fetch_team_stats(away_team_id, sport_key)
    home_recent_task = fetch_team_recent_games(home_team_id, sport_key, 10)
    away_recent_task = fetch_team_recent_games(away_team_id, sport_key, 10)
    
    home_stats, away_stats, home_recent, away_recent = await asyncio.gather(
        home_stats_task, away_stats_task, home_recent_task, away_recent_task,
        return_exceptions=True
    )
    
    # Handle exceptions
    if isinstance(home_stats, Exception):
        home_stats = {}
    if isinstance(away_stats, Exception):
        away_stats = {}
    if isinstance(home_recent, Exception):
        home_recent = []
    if isinstance(away_recent, Exception):
        away_recent = []
    
    # Calculate form metrics
    def calculate_form(recent_games: List[Dict]) -> Dict:
        if not recent_games:
            return {"wins": 0, "losses": 0, "win_pct": 0.5, "avg_margin": 0, "streak": 0}
        
        wins = sum(1 for g in recent_games if g.get("won", False))
        losses = len(recent_games) - wins
        win_pct = wins / len(recent_games) if recent_games else 0.5
        avg_margin = sum(g.get("margin", 0) for g in recent_games) / len(recent_games) if recent_games else 0
        
        # Calculate streak
        streak = 0
        if recent_games:
            first_result = recent_games[0].get("won", False)
            for g in recent_games:
                if g.get("won", False) == first_result:
                    streak += 1 if first_result else -1
                else:
                    break
        
        return {
            "wins": wins,
            "losses": losses,
            "win_pct": round(win_pct, 3),
            "avg_margin": round(avg_margin, 1),
            "streak": streak,
            "last_5": [g.get("won", False) for g in recent_games[:5]]
        }
    
    home_form = calculate_form(home_recent)
    away_form = calculate_form(away_recent)
    
    return {
        "event": event,
        "home_team": {
            "id": home_team_id,
            "name": event.get("home_team", ""),
            "stats": home_stats,
            "recent_games": home_recent,
            "form": home_form
        },
        "away_team": {
            "id": away_team_id,
            "name": event.get("away_team", ""),
            "stats": away_stats,
            "recent_games": away_recent,
            "form": away_form
        },
        "odds": event.get("odds", {}),
        "venue": event.get("venue", {}),
        "commence_time": event.get("commence_time", "")
    }
