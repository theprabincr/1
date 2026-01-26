"""
OddsPortal Scraper - Fetches odds from multiple bookmakers
Uses web scraping to get odds data from OddsPortal.com

Sports supported: NBA, NFL, NHL, Soccer
Markets: Moneyline (1X2), Spread, Over/Under (Totals)
"""
import asyncio
import logging
import httpx
import re
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# OddsPortal URLs by sport
ODDSPORTAL_URLS = {
    "basketball_nba": "https://www.oddsportal.com/basketball/usa/nba/",
    "americanfootball_nfl": "https://www.oddsportal.com/american-football/usa/nfl/",
    "icehockey_nhl": "https://www.oddsportal.com/hockey/usa/nhl/",
    "soccer_epl": "https://www.oddsportal.com/soccer/england/premier-league/",
}

# Headers to mimic a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# Known bookmaker names for standardization
BOOKMAKER_NAMES = {
    "bet365": "Bet365",
    "pinnacle": "Pinnacle",
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betmgm": "BetMGM",
    "unibet": "Unibet",
    "betway": "Betway",
    "william hill": "William Hill",
    "williamhill": "William Hill",
    "1xbet": "1xBet",
    "betfair": "Betfair",
    "bwin": "Bwin",
    "888sport": "888Sport",
    "marathonbet": "Marathonbet",
    "betvictor": "BetVictor",
    "betsson": "Betsson",
    "bovada": "Bovada",
}


class OddsPortalScraper:
    """Scraper for OddsPortal multi-bookmaker odds"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    async def get_session(self):
        """Get or create HTTP session"""
        if self.session is None:
            self.session = httpx.AsyncClient(
                headers=HEADERS,
                timeout=30.0,
                follow_redirects=True
            )
        return self.session
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.aclose()
            self.session = None
    
    async def fetch_odds(self, sport_key: str) -> List[Dict]:
        """
        Fetch odds for all upcoming games in a sport from OddsPortal
        Returns list of events with multi-bookmaker odds
        """
        # Check cache first
        cache_key = f"oddsportal_{sport_key}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).total_seconds() < self.cache_duration:
                logger.info(f"Using cached OddsPortal data for {sport_key}")
                return cached_data
        
        url = ODDSPORTAL_URLS.get(sport_key)
        if not url:
            logger.warning(f"Sport {sport_key} not supported by OddsPortal scraper")
            return []
        
        events = []
        
        try:
            session = await self.get_session()
            response = await session.get(url)
            
            if response.status_code == 200:
                events = self._parse_oddsportal_page(response.text, sport_key)
                logger.info(f"Scraped {len(events)} events from OddsPortal for {sport_key}")
            else:
                logger.error(f"OddsPortal returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error scraping OddsPortal for {sport_key}: {e}")
        
        # Cache results
        self.cache[cache_key] = (datetime.now(timezone.utc), events)
        
        return events
    
    def _parse_oddsportal_page(self, html: str, sport_key: str) -> List[Dict]:
        """Parse OddsPortal HTML to extract odds data"""
        events = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find event rows - OddsPortal uses various class patterns
            # Try multiple selectors
            event_rows = soup.select('.eventRow, .table-main tr, [data-event-id], .event__match')
            
            if not event_rows:
                # Try alternative parsing using JSON data embedded in page
                events = self._extract_embedded_json(html, sport_key)
                if events:
                    return events
                
                # Fallback: Look for table structure
                tables = soup.find_all('table', class_=lambda x: x and 'table' in x.lower())
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        event = self._parse_table_row(row, sport_key)
                        if event:
                            events.append(event)
            else:
                for row in event_rows:
                    event = self._parse_event_row(row, sport_key)
                    if event:
                        events.append(event)
            
        except Exception as e:
            logger.error(f"Error parsing OddsPortal HTML: {e}")
        
        return events
    
    def _extract_embedded_json(self, html: str, sport_key: str) -> List[Dict]:
        """Extract odds data from embedded JSON in the page"""
        events = []
        
        try:
            # OddsPortal often embeds data in script tags
            json_patterns = [
                r'var\s+pageData\s*=\s*({.*?});',
                r'window\.__PRELOADED_STATE__\s*=\s*({.*?});',
                r'"events"\s*:\s*(\[.*?\])',
                r'"matches"\s*:\s*(\[.*?\])',
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, html, re.DOTALL)
                for match in matches:
                    try:
                        data = json.loads(match)
                        if isinstance(data, list):
                            for item in data:
                                event = self._parse_json_event(item, sport_key)
                                if event:
                                    events.append(event)
                        elif isinstance(data, dict):
                            if 'events' in data:
                                for item in data['events']:
                                    event = self._parse_json_event(item, sport_key)
                                    if event:
                                        events.append(event)
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.debug(f"Could not extract embedded JSON: {e}")
        
        return events
    
    def _parse_json_event(self, data: Dict, sport_key: str) -> Optional[Dict]:
        """Parse event from JSON data"""
        try:
            home_team = data.get('homeTeam', data.get('home', data.get('team1', '')))
            away_team = data.get('awayTeam', data.get('away', data.get('team2', '')))
            
            if not home_team or not away_team:
                return None
            
            event_id = data.get('id', data.get('eventId', f"{home_team}_{away_team}".replace(' ', '_').lower())[:20])
            
            # Parse odds
            bookmakers = []
            odds_data = data.get('odds', data.get('bookmakers', {}))
            
            if isinstance(odds_data, dict):
                for bm_key, bm_odds in odds_data.items():
                    bookmaker = self._standardize_bookmaker_name(bm_key)
                    if isinstance(bm_odds, dict):
                        bookmakers.append({
                            "key": bm_key.lower().replace(' ', '_'),
                            "title": bookmaker,
                            "home_odds": bm_odds.get('home', bm_odds.get('1', 0)),
                            "away_odds": bm_odds.get('away', bm_odds.get('2', 0)),
                            "draw_odds": bm_odds.get('draw', bm_odds.get('X', 0)),
                        })
            
            return {
                "id": event_id,
                "sport_key": sport_key,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": data.get('date', data.get('startTime', '')),
                "bookmakers": bookmakers,
                "source": "oddsportal"
            }
            
        except Exception as e:
            logger.debug(f"Error parsing JSON event: {e}")
            return None
    
    def _parse_event_row(self, row, sport_key: str) -> Optional[Dict]:
        """Parse a single event row from HTML"""
        try:
            # Try to find team names
            team_elements = row.select('.participant-name, .team-name, td.name, .event__participant')
            
            if len(team_elements) < 2:
                return None
            
            home_team = team_elements[0].get_text(strip=True)
            away_team = team_elements[1].get_text(strip=True)
            
            if not home_team or not away_team:
                return None
            
            # Find odds
            odds_elements = row.select('.odds-nowrp, .odds, td.odds, [data-odds]')
            
            bookmakers = []
            for i, odds_el in enumerate(odds_elements):
                odds_text = odds_el.get_text(strip=True)
                try:
                    odds_value = float(odds_text)
                    if odds_value > 1:
                        bookmaker = odds_el.get('data-bookmaker', f'book_{i}')
                        bookmakers.append({
                            "key": bookmaker,
                            "title": self._standardize_bookmaker_name(bookmaker),
                            "odds": odds_value
                        })
                except ValueError:
                    continue
            
            event_id = row.get('data-event-id', f"{home_team}_{away_team}".replace(' ', '_').lower()[:20])
            
            return {
                "id": event_id,
                "sport_key": sport_key,
                "home_team": home_team,
                "away_team": away_team,
                "bookmakers": bookmakers,
                "source": "oddsportal"
            }
            
        except Exception as e:
            logger.debug(f"Error parsing event row: {e}")
            return None
    
    def _parse_table_row(self, row, sport_key: str) -> Optional[Dict]:
        """Parse a table row for odds data"""
        return self._parse_event_row(row, sport_key)
    
    def _standardize_bookmaker_name(self, name: str) -> str:
        """Standardize bookmaker name"""
        name_lower = name.lower().replace('-', '').replace('_', '').strip()
        return BOOKMAKER_NAMES.get(name_lower, name.title())


# Alternative: Use a free odds aggregator API that doesn't require keys
class FreeOddsAggregator:
    """
    Aggregates odds from multiple free sources:
    1. ESPN (DraftKings odds)
    2. OddsPortal scraping
    3. Generate synthetic comparison data from ESPN variance
    """
    
    def __init__(self):
        self.oddsportal_scraper = OddsPortalScraper()
        self.odds_cache = {}
    
    async def get_multi_book_odds(self, sport_key: str, event_id: str, espn_event: Dict = None) -> Dict:
        """
        Get odds from multiple sources for an event
        """
        result = {
            "event_id": event_id,
            "sport_key": sport_key,
            "sources": [],
            "bookmakers": [],
            "best_odds": {},
            "odds_movement": [],
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }
        
        # 1. Get ESPN odds (primary source - always available)
        if espn_event:
            espn_bookmakers = self._extract_espn_odds(espn_event)
            result["bookmakers"].extend(espn_bookmakers)
            result["sources"].append("espn")
        
        # 2. Try to get OddsPortal odds
        try:
            op_events = await self.oddsportal_scraper.fetch_odds(sport_key)
            matched_event = self._find_matching_event(op_events, espn_event)
            if matched_event:
                for bm in matched_event.get("bookmakers", []):
                    result["bookmakers"].append({
                        "key": bm.get("key", "unknown"),
                        "title": bm.get("title", "Unknown"),
                        "source": "oddsportal",
                        "markets": [{
                            "key": "h2h",
                            "outcomes": [
                                {"name": espn_event.get("home_team", "Home"), "price": bm.get("home_odds", 1.91)},
                                {"name": espn_event.get("away_team", "Away"), "price": bm.get("away_odds", 1.91)}
                            ]
                        }]
                    })
                result["sources"].append("oddsportal")
        except Exception as e:
            logger.debug(f"Could not fetch OddsPortal odds: {e}")
        
        # 3. Generate synthetic book variance for comparison (simulates market movement)
        if espn_event:
            synthetic_books = self._generate_synthetic_odds(espn_event)
            result["bookmakers"].extend(synthetic_books)
            result["sources"].append("synthetic_variance")
        
        # Calculate best odds
        result["best_odds"] = self._calculate_best_odds(result["bookmakers"], espn_event)
        
        return result
    
    def _extract_espn_odds(self, espn_event: Dict) -> List[Dict]:
        """Extract odds from ESPN event data"""
        bookmakers = []
        
        odds = espn_event.get("odds", {})
        if odds:
            bookmakers.append({
                "key": odds.get("provider_key", "draftkings"),
                "title": odds.get("provider_name", "DraftKings"),
                "source": "espn",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": espn_event.get("home_team", "Home"), "price": odds.get("home_ml_decimal", 1.91)},
                            {"name": espn_event.get("away_team", "Away"), "price": odds.get("away_ml_decimal", 1.91)}
                        ]
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": espn_event.get("home_team", "Home"), "price": 1.91, "point": odds.get("spread", 0)},
                            {"name": espn_event.get("away_team", "Away"), "price": 1.91, "point": -odds.get("spread", 0)}
                        ]
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.91, "point": odds.get("total", 220)},
                            {"name": "Under", "price": 1.91, "point": odds.get("total", 220)}
                        ]
                    }
                ]
            })
        
        return bookmakers
    
    def _generate_synthetic_odds(self, espn_event: Dict) -> List[Dict]:
        """
        Generate synthetic odds variance to simulate multiple bookmakers.
        This helps the AI understand typical market variance.
        """
        import random
        
        odds = espn_event.get("odds", {})
        home_ml = odds.get("home_ml_decimal", 1.91)
        away_ml = odds.get("away_ml_decimal", 1.91)
        spread = odds.get("spread", 0)
        total = odds.get("total", 220)
        
        synthetic_books = []
        
        # Simulate Pinnacle (sharp book - tighter lines)
        variance = random.uniform(-0.02, 0.02)
        synthetic_books.append({
            "key": "pinnacle_est",
            "title": "Pinnacle (estimated)",
            "source": "synthetic",
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": espn_event.get("home_team", "Home"), "price": round(home_ml * (1 + variance), 2)},
                    {"name": espn_event.get("away_team", "Away"), "price": round(away_ml * (1 - variance), 2)}
                ]
            }]
        })
        
        # Simulate Bet365
        variance = random.uniform(-0.03, 0.03)
        synthetic_books.append({
            "key": "bet365_est",
            "title": "Bet365 (estimated)",
            "source": "synthetic",
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": espn_event.get("home_team", "Home"), "price": round(home_ml * (1 + variance), 2)},
                    {"name": espn_event.get("away_team", "Away"), "price": round(away_ml * (1 - variance), 2)}
                ]
            }]
        })
        
        # Simulate FanDuel
        variance = random.uniform(-0.02, 0.04)
        synthetic_books.append({
            "key": "fanduel_est",
            "title": "FanDuel (estimated)",
            "source": "synthetic",
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": espn_event.get("home_team", "Home"), "price": round(home_ml * (1 + variance), 2)},
                    {"name": espn_event.get("away_team", "Away"), "price": round(away_ml * (1 - variance), 2)}
                ]
            }]
        })
        
        return synthetic_books
    
    def _find_matching_event(self, op_events: List[Dict], espn_event: Dict) -> Optional[Dict]:
        """Find matching OddsPortal event for an ESPN event"""
        if not op_events or not espn_event:
            return None
        
        home_team = espn_event.get("home_team", "").lower()
        away_team = espn_event.get("away_team", "").lower()
        
        for op_event in op_events:
            op_home = op_event.get("home_team", "").lower()
            op_away = op_event.get("away_team", "").lower()
            
            # Check for team name matches (partial matching)
            home_match = any(word in op_home for word in home_team.split()) or any(word in home_team for word in op_home.split())
            away_match = any(word in op_away for word in away_team.split()) or any(word in away_team for word in op_away.split())
            
            if home_match and away_match:
                return op_event
        
        return None
    
    def _calculate_best_odds(self, bookmakers: List[Dict], event: Dict) -> Dict:
        """Calculate best available odds across all bookmakers"""
        home_team = event.get("home_team", "Home") if event else "Home"
        away_team = event.get("away_team", "Away") if event else "Away"
        
        home_odds = []
        away_odds = []
        
        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        price = outcome.get("price", 0)
                        if price > 1:
                            if outcome.get("name") == home_team:
                                home_odds.append({"book": bm.get("key"), "odds": price})
                            else:
                                away_odds.append({"book": bm.get("key"), "odds": price})
        
        best_home = max(home_odds, key=lambda x: x["odds"]) if home_odds else {"book": "unknown", "odds": 1.91}
        best_away = max(away_odds, key=lambda x: x["odds"]) if away_odds else {"book": "unknown", "odds": 1.91}
        
        return {
            "home": best_home,
            "away": best_away,
            "avg_home": sum(o["odds"] for o in home_odds) / len(home_odds) if home_odds else 1.91,
            "avg_away": sum(o["odds"] for o in away_odds) / len(away_odds) if away_odds else 1.91
        }
    
    async def close(self):
        """Close all scrapers"""
        await self.oddsportal_scraper.close()


# Singleton instance
_aggregator = None


def get_odds_aggregator() -> FreeOddsAggregator:
    """Get or create odds aggregator"""
    global _aggregator
    if _aggregator is None:
        _aggregator = FreeOddsAggregator()
    return _aggregator


async def fetch_aggregated_odds(sport_key: str, event_id: str, espn_event: Dict = None) -> Dict:
    """Convenience function to fetch aggregated odds"""
    aggregator = get_odds_aggregator()
    return await aggregator.get_multi_book_odds(sport_key, event_id, espn_event)
