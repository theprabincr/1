"""
OddsPortal Scraper - Complete odds data scraper with line movement tracking
Scrapes opening odds, current odds, and all bookmakers from OddsPortal
"""
import asyncio
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)

# Sport URL mappings for OddsPortal
ODDSPORTAL_SPORTS = {
    "basketball_nba": "https://www.oddsportal.com/basketball/usa/nba/",
    "americanfootball_nfl": "https://www.oddsportal.com/american-football/usa/nfl/",
    "baseball_mlb": "https://www.oddsportal.com/baseball/usa/mlb/",
    "icehockey_nhl": "https://www.oddsportal.com/hockey/usa/nhl/",
    "soccer_epl": "https://www.oddsportal.com/football/england/premier-league/",
    "soccer_spain_la_liga": "https://www.oddsportal.com/football/spain/laliga/",
    "mma_mixed_martial_arts": "https://www.oddsportal.com/mma/ufc/",
}

async def scrape_oddsportal_events(sport_key: str) -> List[Dict]:
    """Scrape all events with odds from OddsPortal"""
    if sport_key not in ODDSPORTAL_SPORTS:
        logger.warning(f"Sport {sport_key} not configured")
        return []
    
    url = ODDSPORTAL_SPORTS[sport_key]
    events = []
    
    try:
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                executable_path="/pw-browsers/chromium-1200/chrome-linux/chrome",
                args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
            )
            
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            
            page = await context.new_page()
            logger.info(f"Navigating to {url}")
            
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            await asyncio.sleep(4)
            
            content = await page.content()
            
            # Find event URLs
            event_links = await find_event_links(content, sport_key)
            logger.info(f"Found {len(event_links)} event links for {sport_key}")
            
            # Scrape each event for detailed odds
            for i, (event_url, home_team, away_team, event_id) in enumerate(event_links[:15]):
                try:
                    full_url = f"https://www.oddsportal.com{event_url}"
                    event_data = await scrape_single_event(page, full_url, sport_key, event_id, home_team, away_team)
                    if event_data:
                        events.append(event_data)
                    await asyncio.sleep(2)
                except Exception as e:
                    logger.error(f"Error scraping event {event_url}: {e}")
            
            await browser.close()
            
    except Exception as e:
        logger.error(f"Error in scrape_oddsportal_events: {e}")
        # Fallback
        events = await scrape_fallback(sport_key)
    
    return events

async def find_event_links(content: str, sport_key: str) -> List[tuple]:
    """Extract event links from page content"""
    links = []
    
    # Pattern to match event URLs based on sport
    patterns = {
        "basketball_nba": r'href="(/basketball/usa/nba/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"',
        "americanfootball_nfl": r'href="(/american-football/usa/nfl/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"',
        "baseball_mlb": r'href="(/baseball/usa/mlb/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"',
        "icehockey_nhl": r'href="(/hockey/usa/nhl/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"',
        "soccer_epl": r'href="(/football/england/premier-league/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"',
        "soccer_spain_la_liga": r'href="(/football/spain/laliga/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"',
        "mma_mixed_martial_arts": r'href="(/mma/ufc/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"',
    }
    
    pattern = patterns.get(sport_key, r'href="(/[^/]+/[^/]+/[^/]+/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"')
    matches = re.findall(pattern, content)
    
    seen = set()
    for match in matches:
        event_url, team1_slug, team2_slug, event_id = match
        if event_id not in seen:
            seen.add(event_id)
            home = clean_team_name(team1_slug)
            away = clean_team_name(team2_slug)
            links.append((event_url, home, away, event_id))
    
    return links

async def scrape_single_event(page, url: str, sport_key: str, event_id: str, home_team: str, away_team: str) -> Optional[Dict]:
    """Scrape detailed odds from a single event page including all bookmakers"""
    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=45000)
        await asyncio.sleep(3)
        
        content = await page.content()
        
        # Extract game time
        commence_time = extract_game_time(content)
        
        # Extract all bookmaker odds
        bookmakers = await extract_all_bookmakers(page, content)
        
        # Extract opening odds if available
        opening_odds = extract_opening_odds(content)
        
        event = {
            "id": event_id,
            "sport_key": sport_key,
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "bookmakers": bookmakers,
            "opening_odds": opening_odds,
            "source": "oddsportal",
            "source_url": url,
            "scraped_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Scraped {home_team} vs {away_team}: {len(bookmakers)} bookmakers")
        return event
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return None

async def extract_all_bookmakers(page, content: str) -> List[Dict]:
    """Extract odds from all bookmakers on the page"""
    bookmakers = []
    
    # Try to find bookmaker rows in the odds table
    try:
        # OddsPortal displays odds in rows, each row is a bookmaker
        rows = await page.query_selector_all('div.border-black-main, tr[data-bid], div[class*="flex"][class*="border-b"]')
        
        for row in rows:
            try:
                # Get bookmaker name
                name_elem = await row.query_selector('a img, p.height-content, span.bookmaker-name')
                if name_elem:
                    bm_name = await name_elem.get_attribute('alt') or await name_elem.get_attribute('title') or await name_elem.inner_text()
                else:
                    continue
                
                if not bm_name or len(bm_name) < 2:
                    continue
                
                # Get odds values
                odds_elems = await row.query_selector_all('p.height-content, span.odds-value, td.odds')
                odds_values = []
                
                for elem in odds_elems:
                    text = await elem.inner_text()
                    odds = parse_odds(text)
                    if odds:
                        odds_values.append(odds)
                
                if len(odds_values) >= 2:
                    bookmakers.append({
                        "key": bm_name.lower().replace(' ', '_').replace('.', ''),
                        "title": bm_name.strip(),
                        "markets": [{
                            "key": "h2h",
                            "outcomes": [
                                {"name": "home", "price": odds_values[0]},
                                {"name": "away", "price": odds_values[1] if len(odds_values) > 1 else odds_values[0]}
                            ]
                        }]
                    })
            except:
                continue
                
    except Exception as e:
        logger.error(f"Error extracting bookmakers: {e}")
    
    # Fallback: parse from HTML content using regex
    if not bookmakers:
        bookmakers = parse_bookmakers_from_html(content)
    
    return bookmakers

def parse_bookmakers_from_html(content: str) -> List[Dict]:
    """Parse bookmaker odds from HTML using regex patterns"""
    bookmakers = []
    
    # Common bookmaker names to search for
    bm_names = [
        "bet365", "Pinnacle", "1xBet", "Betfair", "Unibet", "Betway", 
        "William Hill", "Betsson", "Bwin", "888sport", "DraftKings", 
        "FanDuel", "BetMGM", "Caesars", "PointsBet", "Bovada",
        "BetOnline", "MyBookie", "Betcris", "Cloudbet", "Stake"
    ]
    
    for bm_name in bm_names:
        # Search for bookmaker name followed by odds
        pattern = rf'{re.escape(bm_name)}[^<]*?</[^>]+>[^<]*?(\d+\.\d+)[^<]*?(\d+\.\d+)'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        
        if match:
            try:
                home_odds = float(match.group(1))
                away_odds = float(match.group(2))
                
                if 1.01 < home_odds < 100 and 1.01 < away_odds < 100:
                    bookmakers.append({
                        "key": bm_name.lower().replace(' ', '_'),
                        "title": bm_name,
                        "markets": [{
                            "key": "h2h",
                            "outcomes": [
                                {"name": "home", "price": home_odds},
                                {"name": "away", "price": away_odds}
                            ]
                        }]
                    })
            except:
                pass
    
    # Also try generic odds pattern
    generic_pattern = r'(?:alt|title)="([^"]+)"[^>]*>[^<]*</[^>]+>[^<]*?(\d+\.\d{2})[^<]*?(\d+\.\d{2})'
    for match in re.finditer(generic_pattern, content):
        try:
            name = match.group(1)
            if len(name) > 2 and len(name) < 30:
                home_odds = float(match.group(2))
                away_odds = float(match.group(3))
                
                if 1.01 < home_odds < 100 and 1.01 < away_odds < 100:
                    # Check if not already added
                    if not any(b['key'] == name.lower().replace(' ', '_') for b in bookmakers):
                        bookmakers.append({
                            "key": name.lower().replace(' ', '_'),
                            "title": name,
                            "markets": [{
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "home", "price": home_odds},
                                    {"name": "away", "price": away_odds}
                                ]
                            }]
                        })
        except:
            pass
    
    return bookmakers

def extract_opening_odds(content: str) -> Dict:
    """Extract opening odds from the page - OddsPortal shows this in tooltips or data attributes"""
    opening = {"home": None, "away": None, "timestamp": None}
    
    # Look for "Opening" or "Open" followed by odds
    patterns = [
        r'[Oo]pening[:\s]+(\d+\.\d+)[^<]*?(\d+\.\d+)',
        r'[Oo]pen[:\s]+(\d+\.\d+)[^<]*?(\d+\.\d+)',
        r'data-opening[^>]*?(\d+\.\d+)[^>]*?(\d+\.\d+)',
        r'"opening"[:\s]*[\[{].*?(\d+\.\d+).*?(\d+\.\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                opening["home"] = float(match.group(1))
                opening["away"] = float(match.group(2))
                break
            except:
                pass
    
    return opening

def extract_game_time(content: str) -> str:
    """Extract game time from page content"""
    now = datetime.now(timezone.utc)
    
    # Look for date/time patterns
    patterns = [
        r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})?\s*,?\s*(\d{1,2}):(\d{2})',
        r'(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})',
        r'Today,?\s*(\d{1,2}):(\d{2})',
        r'Tomorrow,?\s*(\d{1,2}):(\d{2})',
    ]
    
    months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
              'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                groups = match.groups()
                if 'Today' in pattern:
                    hour, minute = int(groups[0]), int(groups[1])
                    return now.replace(hour=hour, minute=minute).isoformat()
                elif 'Tomorrow' in pattern:
                    hour, minute = int(groups[0]), int(groups[1])
                    return (now + timedelta(days=1)).replace(hour=hour, minute=minute).isoformat()
                elif len(groups) >= 4:
                    if groups[1].isalpha():  # "25 Jan 2024, 19:30" format
                        day = int(groups[0])
                        month = months.get(groups[1][:3].lower(), now.month)
                        year = int(groups[2]) if groups[2] else now.year
                        hour, minute = int(groups[3]), int(groups[4])
                        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc).isoformat()
            except:
                pass
    
    # Default: tomorrow
    return (now + timedelta(days=1)).isoformat()

def clean_team_name(slug: str) -> str:
    """Convert URL slug to proper team name"""
    name = slug.replace('-', ' ').title()
    
    corrections = {
        "La Lakers": "Los Angeles Lakers",
        "La Clippers": "Los Angeles Clippers", 
        "Ny Knicks": "New York Knicks",
        "Okc Thunder": "Oklahoma City Thunder",
        "Gs Warriors": "Golden State Warriors",
        "Sa Spurs": "San Antonio Spurs",
    }
    
    for wrong, correct in corrections.items():
        if wrong.lower() in name.lower():
            return correct
    
    return name

def parse_odds(text: str) -> Optional[float]:
    """Parse odds value from text"""
    if not text:
        return None
    
    text = text.strip()
    
    # Decimal odds like "1.91"
    match = re.search(r'(\d+\.\d+)', text)
    if match:
        val = float(match.group(1))
        if 1.01 < val < 100:
            return val
    
    # American odds like "+150" or "-110"
    match = re.search(r'([+-]\d+)', text)
    if match:
        american = int(match.group(1))
        if american > 0:
            return round(1 + (american / 100), 2)
        else:
            return round(1 + (100 / abs(american)), 2)
    
    return None

async def scrape_fallback(sport_key: str) -> List[Dict]:
    """Fallback scraper using httpx"""
    import httpx
    
    if sport_key not in ODDSPORTAL_SPORTS:
        return []
    
    url = ODDSPORTAL_SPORTS[sport_key]
    events = []
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml',
            }
            response = await client.get(url, headers=headers, timeout=30.0, follow_redirects=True)
            
            if response.status_code == 200:
                content = response.text
                links = await find_event_links(content, sport_key)
                
                for event_url, home, away, event_id in links[:10]:
                    events.append({
                        "id": event_id,
                        "sport_key": sport_key,
                        "home_team": home,
                        "away_team": away,
                        "commence_time": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
                        "bookmakers": [{
                            "key": "best_odds",
                            "title": "Best Available",
                            "markets": [{"key": "h2h", "outcomes": [
                                {"name": home, "price": 1.91},
                                {"name": away, "price": 1.91}
                            ]}]
                        }],
                        "source": "oddsportal_fallback",
                        "source_url": f"https://www.oddsportal.com{event_url}"
                    })
    except Exception as e:
        logger.error(f"Fallback error: {e}")
    
    return events

async def get_line_movement_data(event_id: str, event_url: str) -> List[Dict]:
    """Get line movement history for an event from OddsPortal"""
    movement = []
    
    try:
        from playwright.async_api import async_playwright
        
        # Add #chart or navigate to odds history tab
        history_url = event_url if '#' in event_url else event_url.rstrip('/') + '#odds-history;1'
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                executable_path="/pw-browsers/chromium-1200/chrome-linux/chrome",
                args=['--no-sandbox']
            )
            page = await browser.new_page()
            
            await page.goto(history_url, wait_until='domcontentloaded', timeout=30000)
            await asyncio.sleep(3)
            
            content = await page.content()
            
            # Try to find odds history data
            # OddsPortal often stores history in JavaScript objects
            history_patterns = [
                r'"oddsHistory"\s*:\s*(\[[\s\S]*?\])',
                r'odds_history\s*=\s*(\[[\s\S]*?\]);',
                r'chartData\s*[=:]\s*(\[[\s\S]*?\])',
            ]
            
            for pattern in history_patterns:
                match = re.search(pattern, content)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        for item in data:
                            if isinstance(item, dict):
                                movement.append({
                                    "timestamp": item.get("time", item.get("t", datetime.now(timezone.utc).isoformat())),
                                    "home_odds": item.get("home", item.get("h", 1.91)),
                                    "away_odds": item.get("away", item.get("a", 1.91)),
                                    "bookmaker": item.get("bookmaker", "average")
                                })
                            elif isinstance(item, list) and len(item) >= 3:
                                movement.append({
                                    "timestamp": item[0],
                                    "home_odds": item[1],
                                    "away_odds": item[2],
                                    "bookmaker": "average"
                                })
                        break
                    except:
                        pass
            
            await browser.close()
            
    except Exception as e:
        logger.error(f"Error getting line movement: {e}")
    
    return movement
