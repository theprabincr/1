"""
OddsPortal Scraper - Complete odds data scraper from oddsportal.com
Scrapes multiple bookmakers, markets (moneyline, spreads, totals), and tracks line movement
"""
import asyncio
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import uuid
import json

logger = logging.getLogger(__name__)

# Sport URL mappings for OddsPortal
ODDSPORTAL_SPORTS = {
    "basketball_nba": {
        "url": "https://www.oddsportal.com/basketball/usa/nba/",
        "name": "NBA Basketball"
    },
    "americanfootball_nfl": {
        "url": "https://www.oddsportal.com/american-football/usa/nfl/",
        "name": "NFL Football"
    },
    "baseball_mlb": {
        "url": "https://www.oddsportal.com/baseball/usa/mlb/",
        "name": "MLB Baseball"
    },
    "icehockey_nhl": {
        "url": "https://www.oddsportal.com/hockey/usa/nhl/",
        "name": "NHL Hockey"
    },
    "soccer_epl": {
        "url": "https://www.oddsportal.com/football/england/premier-league/",
        "name": "English Premier League"
    },
    "soccer_spain_la_liga": {
        "url": "https://www.oddsportal.com/football/spain/laliga/",
        "name": "Spanish La Liga"
    },
    "mma_mixed_martial_arts": {
        "url": "https://www.oddsportal.com/mma/ufc/",
        "name": "UFC MMA"
    },
}

# Common bookmakers on OddsPortal
BOOKMAKERS = [
    "bet365", "Pinnacle", "Unibet", "Betfair", "William Hill",
    "1xBet", "Betway", "888sport", "Betsson", "Bwin",
    "DraftKings", "FanDuel", "BetMGM", "Caesars", "PointsBet"
]

async def scrape_oddsportal_events(sport_key: str) -> List[Dict]:
    """
    Scrape all events with odds from OddsPortal for a given sport
    Returns list of events with bookmaker odds
    """
    if sport_key not in ODDSPORTAL_SPORTS:
        logger.warning(f"Sport {sport_key} not configured for OddsPortal")
        return []
    
    sport_config = ODDSPORTAL_SPORTS[sport_key]
    url = sport_config["url"]
    events = []
    
    try:
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                executable_path="/pw-browsers/chromium-1200/chrome-linux/chrome",
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            page = await context.new_page()
            
            # Navigate to the page
            logger.info(f"Scraping OddsPortal: {url}")
            await page.goto(url, wait_until='networkidle', timeout=60000)
            await asyncio.sleep(3)
            
            # Get page content
            content = await page.content()
            
            # Parse events from the page
            events = await parse_events_from_page(page, content, sport_key)
            
            logger.info(f"Found {len(events)} events for {sport_key}")
            
            # For each event, try to get detailed odds from event page
            for i, event in enumerate(events[:10]):  # Limit to 10 events to save time
                if event.get('source_url'):
                    try:
                        detailed_odds = await scrape_event_details(page, event['source_url'], sport_key)
                        if detailed_odds:
                            event['bookmakers'] = detailed_odds.get('bookmakers', event.get('bookmakers', []))
                            event['opening_odds'] = detailed_odds.get('opening_odds', {})
                            event['markets'] = detailed_odds.get('markets', {})
                        await asyncio.sleep(1)  # Rate limiting
                    except Exception as e:
                        logger.error(f"Error getting details for event: {e}")
            
            await browser.close()
            
    except ImportError:
        logger.error("Playwright not installed")
        events = await scrape_oddsportal_fallback(sport_key)
    except Exception as e:
        logger.error(f"Error scraping OddsPortal: {e}")
        events = await scrape_oddsportal_fallback(sport_key)
    
    return events

async def parse_events_from_page(page, content: str, sport_key: str) -> List[Dict]:
    """Parse events from OddsPortal page content"""
    events = []
    
    # Try to find event links in the page
    # OddsPortal URL pattern: /sport/country/league/team1-team2-EVENTID/
    
    if 'basketball' in sport_key:
        pattern = r'href="(/basketball/usa/nba/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
    elif 'football' in sport_key and 'american' not in sport_key:
        pattern = r'href="(/football/[^/]+/[^/]+/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
    elif 'americanfootball' in sport_key:
        pattern = r'href="(/american-football/usa/nfl/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
    elif 'hockey' in sport_key:
        pattern = r'href="(/hockey/usa/nhl/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
    elif 'baseball' in sport_key:
        pattern = r'href="(/baseball/usa/mlb/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
    elif 'mma' in sport_key:
        pattern = r'href="(/mma/ufc/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
    else:
        pattern = r'href="(/[^/]+/[^/]+/[^/]+/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
    
    matches = re.findall(pattern, content)
    seen_events = set()
    
    for match in matches:
        try:
            event_url, team1_slug, team2_slug, event_id = match
            
            if event_id in seen_events:
                continue
            seen_events.add(event_id)
            
            # Clean team names
            home_team = clean_team_name(team1_slug)
            away_team = clean_team_name(team2_slug)
            
            # Try to find odds near the event link in HTML
            odds_pattern = rf'{re.escape(event_url)}[^>]*>[^<]*</a>[^<]*(?:<[^>]*>)*[^<]*?(\d+\.\d+)[^<]*?(\d+\.\d+)'
            odds_match = re.search(odds_pattern, content)
            
            home_odds = 1.91
            away_odds = 1.91
            
            if odds_match:
                try:
                    home_odds = float(odds_match.group(1))
                    away_odds = float(odds_match.group(2))
                except:
                    pass
            
            # Estimate game time (OddsPortal shows relative times)
            commence_time = estimate_game_time(content, event_url)
            
            events.append({
                "id": event_id,
                "sport_key": sport_key,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": commence_time,
                "bookmakers": [
                    {
                        "key": "oddsportal_best",
                        "title": "Best Available",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": home_team, "price": home_odds},
                                    {"name": away_team, "price": away_odds}
                                ]
                            }
                        ]
                    }
                ],
                "source": "oddsportal",
                "source_url": f"https://www.oddsportal.com{event_url}",
                "scraped_at": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error parsing event match: {e}")
            continue
    
    return events

async def scrape_event_details(page, event_url: str, sport_key: str) -> Dict:
    """
    Scrape detailed odds from a specific event page
    Gets all bookmakers, markets, opening odds, and current odds
    """
    result = {
        "bookmakers": [],
        "opening_odds": {},
        "markets": {
            "h2h": [],  # Moneyline
            "spreads": [],
            "totals": []
        }
    }
    
    try:
        await page.goto(event_url, wait_until='networkidle', timeout=30000)
        await asyncio.sleep(2)
        
        content = await page.content()
        
        # Parse bookmaker odds from the page
        # OddsPortal displays odds in a table format
        
        # Find all bookmaker rows
        bookmaker_pattern = r'(?:bet365|Pinnacle|Unibet|Betfair|William Hill|1xBet|Betway|888sport|DraftKings|FanDuel|BetMGM)[^<]*?(\d+\.\d+)[^<]*?(\d+\.\d+)'
        bookie_matches = re.findall(bookmaker_pattern, content, re.IGNORECASE)
        
        bookmaker_names = ["bet365", "Pinnacle", "Unibet", "Betfair", "DraftKings", "FanDuel", "BetMGM"]
        
        for i, bookie in enumerate(bookmaker_names):
            # Try to find this bookmaker's odds
            bookie_pattern = rf'{bookie}[^<]*?(\d+\.\d+)[^<]*?(\d+\.\d+)'
            match = re.search(bookie_pattern, content, re.IGNORECASE)
            
            if match:
                try:
                    home_odds = float(match.group(1))
                    away_odds = float(match.group(2))
                    
                    result["bookmakers"].append({
                        "key": bookie.lower().replace(' ', '_'),
                        "title": bookie,
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "home", "price": home_odds},
                                    {"name": "away", "price": away_odds}
                                ]
                            }
                        ]
                    })
                except:
                    pass
        
        # Try to find opening odds (usually shown in tooltip or data attributes)
        opening_pattern = r'Opening[:\s]*(\d+\.\d+)[^<]*?(\d+\.\d+)'
        opening_match = re.search(opening_pattern, content, re.IGNORECASE)
        
        if opening_match:
            result["opening_odds"] = {
                "home": float(opening_match.group(1)),
                "away": float(opening_match.group(2))
            }
        
        # Try to find spread/handicap odds
        spread_pattern = r'(?:Spread|Handicap)[^<]*?([+-]?\d+\.?\d*)[^<]*?(\d+\.\d+)[^<]*?(\d+\.\d+)'
        spread_match = re.search(spread_pattern, content, re.IGNORECASE)
        
        if spread_match:
            result["markets"]["spreads"].append({
                "point": float(spread_match.group(1)),
                "home_odds": float(spread_match.group(2)),
                "away_odds": float(spread_match.group(3))
            })
        
        # Try to find totals (over/under)
        totals_pattern = r'(?:Over|Total)[^<]*?(\d+\.?\d*)[^<]*?(\d+\.\d+)[^<]*?Under[^<]*?(\d+\.\d+)'
        totals_match = re.search(totals_pattern, content, re.IGNORECASE)
        
        if totals_match:
            result["markets"]["totals"].append({
                "point": float(totals_match.group(1)),
                "over_odds": float(totals_match.group(2)),
                "under_odds": float(totals_match.group(3))
            })
        
    except Exception as e:
        logger.error(f"Error scraping event details: {e}")
    
    return result

def clean_team_name(slug: str) -> str:
    """Convert URL slug to proper team name"""
    # Replace hyphens with spaces and title case
    name = slug.replace('-', ' ').title()
    
    # Common team name corrections
    corrections = {
        "La Lakers": "Los Angeles Lakers",
        "La Clippers": "Los Angeles Clippers",
        "Ny Knicks": "New York Knicks",
        "Ny Rangers": "New York Rangers",
        "Okc Thunder": "Oklahoma City Thunder",
        "Gs Warriors": "Golden State Warriors",
        "Nola Pelicans": "New Orleans Pelicans",
        "Sa Spurs": "San Antonio Spurs",
    }
    
    for wrong, correct in corrections.items():
        if wrong.lower() in name.lower():
            return correct
    
    return name

def estimate_game_time(content: str, event_url: str) -> str:
    """Estimate game time from page content"""
    now = datetime.now(timezone.utc)
    
    # Look for time indicators near the event
    time_pattern = r'(\d{1,2}):(\d{2})'
    date_pattern = r'(\d{1,2})\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    
    # Default to tomorrow
    game_time = now + timedelta(days=1)
    
    # Try to find date/time near the event URL in content
    event_section = content[max(0, content.find(event_url)-500):content.find(event_url)+200]
    
    time_match = re.search(time_pattern, event_section)
    date_match = re.search(date_pattern, event_section, re.IGNORECASE)
    
    if time_match:
        hour, minute = int(time_match.group(1)), int(time_match.group(2))
        game_time = game_time.replace(hour=hour, minute=minute)
    
    if date_match:
        day = int(date_match.group(1))
        month_str = date_match.group(2)
        months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        month = months.get(month_str.lower(), now.month)
        year = now.year if month >= now.month else now.year + 1
        game_time = game_time.replace(year=year, month=month, day=day)
    
    return game_time.isoformat()

async def scrape_oddsportal_fallback(sport_key: str) -> List[Dict]:
    """Fallback scraper using httpx when Playwright fails"""
    import httpx
    
    if sport_key not in ODDSPORTAL_SPORTS:
        return []
    
    sport_config = ODDSPORTAL_SPORTS[sport_key]
    url = sport_config["url"]
    events = []
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            response = await client.get(url, headers=headers, timeout=30.0, follow_redirects=True)
            
            if response.status_code == 200:
                content = response.text
                
                # Use same parsing logic as main scraper
                if 'basketball' in sport_key:
                    pattern = r'href="(/basketball/usa/nba/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
                elif 'americanfootball' in sport_key:
                    pattern = r'href="(/american-football/usa/nfl/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
                else:
                    pattern = r'href="(/[^/]+/[^/]+/[^/]+/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
                
                matches = re.findall(pattern, content)
                seen_events = set()
                
                for match in matches[:15]:
                    event_url, team1_slug, team2_slug, event_id = match
                    
                    if event_id in seen_events:
                        continue
                    seen_events.add(event_id)
                    
                    home_team = clean_team_name(team1_slug)
                    away_team = clean_team_name(team2_slug)
                    
                    events.append({
                        "id": event_id,
                        "sport_key": sport_key,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
                        "bookmakers": [
                            {
                                "key": "oddsportal_best",
                                "title": "Best Available",
                                "markets": [
                                    {
                                        "key": "h2h",
                                        "outcomes": [
                                            {"name": home_team, "price": 1.91},
                                            {"name": away_team, "price": 1.91}
                                        ]
                                    }
                                ]
                            }
                        ],
                        "source": "oddsportal_fallback",
                        "source_url": f"https://www.oddsportal.com{event_url}"
                    })
                
                logger.info(f"Fallback scraper found {len(events)} events for {sport_key}")
                        
    except Exception as e:
        logger.error(f"Fallback scraper error: {e}")
    
    return events

async def get_line_movement_data(event_id: str, event_url: str) -> List[Dict]:
    """
    Get historical line movement data for an event
    OddsPortal shows odds history when you hover or click on odds
    """
    movement_data = []
    
    try:
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                executable_path="/pw-browsers/chromium-1200/chrome-linux/chrome",
                args=['--no-sandbox']
            )
            page = await browser.new_page()
            
            await page.goto(event_url, wait_until='networkidle', timeout=30000)
            await asyncio.sleep(2)
            
            content = await page.content()
            
            # Look for odds history data
            # OddsPortal often includes this in JavaScript or data attributes
            history_pattern = r'odds[Hh]istory["\s:]*\[([^\]]+)\]'
            history_match = re.search(history_pattern, content)
            
            if history_match:
                try:
                    history_str = f"[{history_match.group(1)}]"
                    history_data = json.loads(history_str)
                    
                    for item in history_data:
                        movement_data.append({
                            "timestamp": item.get("time", datetime.now(timezone.utc).isoformat()),
                            "home_odds": item.get("home", 1.91),
                            "away_odds": item.get("away", 1.91)
                        })
                except:
                    pass
            
            await browser.close()
            
    except Exception as e:
        logger.error(f"Error getting line movement: {e}")
    
    return movement_data
