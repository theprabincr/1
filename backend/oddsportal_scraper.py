"""
OddsPortal Scraper - Scrapes odds data from oddsportal.com
Uses Playwright for JavaScript rendering
"""
import asyncio
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import uuid

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

async def scrape_oddsportal(sport_key: str) -> List[Dict]:
    """
    Scrape odds from OddsPortal for a given sport
    Returns list of events with odds data
    """
    if sport_key not in ODDSPORTAL_SPORTS:
        logger.warning(f"Sport {sport_key} not configured for OddsPortal")
        return []
    
    url = ODDSPORTAL_SPORTS[sport_key]
    events = []
    
    try:
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                executable_path="/pw-browsers/chromium-1200/chrome-linux/chrome"
            )
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()
            
            # Navigate to the page
            await page.goto(url, wait_until='networkidle', timeout=60000)
            await asyncio.sleep(3)  # Wait for dynamic content
            
            # Get all event rows - OddsPortal uses various selectors
            event_rows = await page.query_selector_all('div[class*="eventRow"], div[class*="flex-col"][class*="border-black"]')
            
            if not event_rows:
                # Try alternative selector
                event_rows = await page.query_selector_all('a[href*="/basketball/usa/nba/"][class*="flex"]')
            
            logger.info(f"Found {len(event_rows)} event rows on OddsPortal for {sport_key}")
            
            for row in event_rows:
                try:
                    event = await parse_event_row(row, sport_key)
                    if event:
                        events.append(event)
                except Exception as e:
                    logger.error(f"Error parsing event row: {e}")
                    continue
            
            await browser.close()
            
    except ImportError:
        logger.error("Playwright not installed. Using fallback method.")
        events = await scrape_oddsportal_fallback(sport_key)
    except Exception as e:
        logger.error(f"Error scraping OddsPortal: {e}")
        events = await scrape_oddsportal_fallback(sport_key)
    
    return events

async def parse_event_row(row, sport_key: str) -> Optional[Dict]:
    """Parse a single event row from OddsPortal"""
    try:
        # Get team names
        teams = await row.query_selector_all('a[class*="participant-name"]')
        if len(teams) < 2:
            teams = await row.query_selector_all('p[class*="participant-name"]')
        
        if len(teams) < 2:
            return None
        
        home_team = await teams[0].inner_text()
        away_team = await teams[1].inner_text()
        
        # Get event time
        time_elem = await row.query_selector('p[class*="date"]')
        event_time = None
        if time_elem:
            time_text = await time_elem.inner_text()
            event_time = parse_event_time(time_text)
        
        # Get odds
        odds_cells = await row.query_selector_all('p[class*="odds-value"], span[class*="odds-value"]')
        
        home_odds = None
        away_odds = None
        draw_odds = None
        
        for i, cell in enumerate(odds_cells):
            try:
                odds_text = await cell.inner_text()
                odds_val = parse_odds_value(odds_text)
                if i == 0:
                    home_odds = odds_val
                elif i == 1:
                    # Could be draw or away depending on sport
                    if sport_key.startswith('soccer'):
                        draw_odds = odds_val
                    else:
                        away_odds = odds_val
                elif i == 2:
                    away_odds = odds_val
            except:
                continue
        
        # Get event URL for more details
        link_elem = await row.query_selector('a[href*="/basketball/"], a[href*="/football/"], a[href*="/american-football/"]')
        event_url = None
        if link_elem:
            event_url = await link_elem.get_attribute('href')
        
        # Create event ID
        event_id = str(uuid.uuid4())
        
        return {
            "id": event_id,
            "sport_key": sport_key,
            "home_team": home_team.strip(),
            "away_team": away_team.strip(),
            "commence_time": event_time or datetime.now(timezone.utc).isoformat(),
            "bookmakers": [
                {
                    "key": "oddsportal_best",
                    "title": "OddsPortal Best Odds",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": home_team.strip(), "price": home_odds or 1.91},
                                {"name": away_team.strip(), "price": away_odds or 1.91}
                            ] + ([{"name": "Draw", "price": draw_odds}] if draw_odds else [])
                        }
                    ]
                }
            ],
            "source": "oddsportal",
            "source_url": event_url
        }
        
    except Exception as e:
        logger.error(f"Error parsing event: {e}")
        return None

def parse_event_time(time_text: str) -> str:
    """Parse time text from OddsPortal into ISO format"""
    try:
        # OddsPortal shows times like "Today, 19:30" or "25 Jan, 14:00"
        now = datetime.now(timezone.utc)
        
        if 'Today' in time_text:
            time_match = re.search(r'(\d{1,2}):(\d{2})', time_text)
            if time_match:
                hour, minute = int(time_match.group(1)), int(time_match.group(2))
                return now.replace(hour=hour, minute=minute, second=0, microsecond=0).isoformat()
        
        # Try to parse date like "25 Jan"
        date_match = re.search(r'(\d{1,2})\s*(\w{3})', time_text)
        time_match = re.search(r'(\d{1,2}):(\d{2})', time_text)
        
        if date_match and time_match:
            day = int(date_match.group(1))
            month_str = date_match.group(2)
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            
            months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            month = months.get(month_str, now.month)
            year = now.year if month >= now.month else now.year + 1
            
            return datetime(year, month, day, hour, minute, tzinfo=timezone.utc).isoformat()
        
        return now.isoformat()
    except:
        return datetime.now(timezone.utc).isoformat()

def parse_odds_value(odds_text: str) -> Optional[float]:
    """Parse odds text into decimal float"""
    try:
        # Remove whitespace
        odds_text = odds_text.strip()
        
        # Skip if empty or dash
        if not odds_text or odds_text == '-':
            return None
        
        # Handle decimal odds like "1.91"
        if '.' in odds_text:
            return float(odds_text)
        
        # Handle American odds like "+150" or "-110"
        if odds_text.startswith('+') or odds_text.startswith('-'):
            american = int(odds_text)
            if american > 0:
                return 1 + (american / 100)
            else:
                return 1 + (100 / abs(american))
        
        # Try direct float conversion
        return float(odds_text)
    except:
        return None

async def scrape_oddsportal_fallback(sport_key: str) -> List[Dict]:
    """
    Fallback method using httpx when Playwright fails
    Parses HTML to extract basic event info and odds
    """
    import httpx
    
    if sport_key not in ODDSPORTAL_SPORTS:
        return []
    
    url = ODDSPORTAL_SPORTS[sport_key]
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
                html = response.text
                
                # Parse fixture links from the page content
                # OddsPortal links look like: /basketball/usa/nba/team1-team2-EVENTID/
                
                # Pattern for NBA games
                if 'basketball' in url:
                    event_pattern = r'href="(/basketball/usa/nba/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
                elif 'football' in url and 'american' not in url:
                    event_pattern = r'href="(/football/[^/]+/[^/]+/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
                elif 'american-football' in url:
                    event_pattern = r'href="(/american-football/usa/nfl/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
                elif 'hockey' in url:
                    event_pattern = r'href="(/hockey/usa/nhl/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
                elif 'baseball' in url:
                    event_pattern = r'href="(/baseball/usa/mlb/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
                else:
                    event_pattern = r'href="(/[^/]+/[^/]+/[^/]+/([^/]+)-([^/]+)-([A-Za-z0-9]+)/)"'
                
                matches = re.findall(event_pattern, html)
                seen_events = set()
                
                for match in matches[:20]:  # Limit to 20 events
                    event_url, team1_slug, team2_slug, event_id = match
                    
                    # Skip if already seen
                    if event_id in seen_events:
                        continue
                    seen_events.add(event_id)
                    
                    # Clean up team names
                    home_team = team1_slug.replace('-', ' ').title()
                    away_team = team2_slug.replace('-', ' ').title()
                    
                    # Try to extract odds from the page (basic pattern)
                    odds_pattern = rf'{re.escape(event_url)}[^<]*?(\d+\.\d+)[^<]*?(\d+\.\d+)'
                    odds_match = re.search(odds_pattern, html)
                    
                    home_odds = 1.91
                    away_odds = 1.91
                    
                    if odds_match:
                        try:
                            home_odds = float(odds_match.group(1))
                            away_odds = float(odds_match.group(2))
                        except:
                            pass
                    
                    events.append({
                        "id": event_id,
                        "sport_key": sport_key,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
                        "bookmakers": [
                            {
                                "key": "oddsportal_best",
                                "title": "OddsPortal Best Odds",
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
                        "source": "oddsportal_fallback",
                        "source_url": f"https://www.oddsportal.com{event_url}"
                    })
                
                logger.info(f"Fallback scraper found {len(events)} events for {sport_key}")
                        
    except Exception as e:
        logger.error(f"Fallback scraper error: {e}")
    
    return events

async def scrape_event_details(event_url: str) -> Dict:
    """
    Scrape detailed odds from a specific event page
    Gets opening odds, current odds, and bookmaker breakdown
    """
    details = {
        "opening_odds": {},
        "current_odds": {},
        "bookmakers": [],
        "line_movement": []
    }
    
    try:
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            await page.goto(event_url, wait_until='networkidle', timeout=30000)
            await asyncio.sleep(2)
            
            # Get odds table rows
            odds_rows = await page.query_selector_all('div[class*="border-black-borders"]')
            
            for row in odds_rows:
                try:
                    # Get bookmaker name
                    bookie_elem = await row.query_selector('a[class*="flex-center"], img[alt]')
                    bookie_name = "Unknown"
                    if bookie_elem:
                        bookie_name = await bookie_elem.get_attribute('title') or await bookie_elem.get_attribute('alt') or "Unknown"
                    
                    # Get current odds
                    odds_elems = await row.query_selector_all('p[class*="height-content"]')
                    
                    if len(odds_elems) >= 2:
                        home_odds = parse_odds_value(await odds_elems[0].inner_text())
                        away_odds = parse_odds_value(await odds_elems[1].inner_text())
                        
                        # Check for opening odds (hover tooltip)
                        opening_home = None
                        opening_away = None
                        
                        # Try to get opening odds from data attributes or tooltips
                        for i, elem in enumerate(odds_elems[:2]):
                            title = await elem.get_attribute('title')
                            if title and 'Opening' in title:
                                opening_match = re.search(r'Opening:\s*([\d.]+)', title)
                                if opening_match:
                                    if i == 0:
                                        opening_home = float(opening_match.group(1))
                                    else:
                                        opening_away = float(opening_match.group(1))
                        
                        details["bookmakers"].append({
                            "name": bookie_name,
                            "home_odds": home_odds,
                            "away_odds": away_odds,
                            "opening_home": opening_home,
                            "opening_away": opening_away
                        })
                        
                except Exception as e:
                    continue
            
            # Calculate best odds
            if details["bookmakers"]:
                best_home = max((b["home_odds"] for b in details["bookmakers"] if b["home_odds"]), default=None)
                best_away = max((b["away_odds"] for b in details["bookmakers"] if b["away_odds"]), default=None)
                details["current_odds"] = {"home": best_home, "away": best_away}
                
                # Opening odds from first bookmaker with opening data
                for b in details["bookmakers"]:
                    if b.get("opening_home"):
                        details["opening_odds"]["home"] = b["opening_home"]
                        details["opening_odds"]["away"] = b.get("opening_away")
                        break
            
            await browser.close()
            
    except Exception as e:
        logger.error(f"Error scraping event details: {e}")
    
    return details
