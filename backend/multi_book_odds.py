"""
Multi-Bookmaker Odds Provider
Fetches odds from multiple sources:
1. ESPN (DraftKings) - Free, always available
2. The Odds API - Multiple bookmakers (requires API key)

This allows comparing odds across bookmakers to find value.
"""
import asyncio
import logging
import httpx
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# The Odds API configuration
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Sport key mapping (our keys -> Odds API keys)
SPORT_KEY_MAP = {
    "basketball_nba": "basketball_nba",
    "americanfootball_nfl": "americanfootball_nfl",
    "baseball_mlb": "baseball_mlb",
    "icehockey_nhl": "icehockey_nhl",
    "soccer_epl": "soccer_epl"
}

# Bookmakers to track (prioritized by reliability)
PRIORITY_BOOKMAKERS = [
    "pinnacle",      # Sharpest book - best indicator of true odds
    "draftkings",
    "fanduel",
    "betmgm",
    "bet365",
    "bovada",
    "williamhill_us",
    "pointsbetus",
    "betonlineag"
]


class MultiBookOddsProvider:
    """Provider for fetching odds from multiple bookmakers"""
    
    def __init__(self, odds_api_key: str = None):
        self.odds_api_key = odds_api_key
        self.cache = {}  # Cache to avoid repeated API calls
        self.cache_duration = 300  # 5 minutes
    
    async def get_multi_book_odds(self, sport_key: str, event_id: str = None) -> Dict:
        """
        Fetch odds from multiple bookmakers for a sport.
        If event_id provided, filters to that event.
        """
        # Check cache first
        cache_key = f"{sport_key}_{event_id or 'all'}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).total_seconds() < self.cache_duration:
                return cached_data
        
        result = {
            "sport_key": sport_key,
            "bookmakers": [],
            "sources": [],
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Always try ESPN (free)
        espn_odds = await self._fetch_espn_odds(sport_key, event_id)
        if espn_odds:
            result["bookmakers"].append(espn_odds)
            result["sources"].append("espn")
        
        # If we have Odds API key, fetch from multiple bookmakers
        if self.odds_api_key:
            odds_api_result = await self._fetch_odds_api(sport_key, event_id)
            if odds_api_result:
                result["bookmakers"].extend(odds_api_result)
                result["sources"].append("the_odds_api")
        
        # Cache result
        self.cache[cache_key] = (datetime.now(timezone.utc), result)
        
        return result
    
    async def _fetch_espn_odds(self, sport_key: str, event_id: str = None) -> Optional[Dict]:
        """Fetch odds from ESPN (DraftKings)"""
        # ESPN odds are already fetched in espn_data_provider.py
        # This is a placeholder that returns the ESPN bookmaker format
        
        return {
            "key": "draftkings",
            "title": "DraftKings (via ESPN)",
            "markets": []  # Filled in from ESPN data
        }
    
    async def _fetch_odds_api(self, sport_key: str, event_id: str = None) -> List[Dict]:
        """Fetch odds from The Odds API"""
        if not self.odds_api_key:
            return []
        
        api_sport_key = SPORT_KEY_MAP.get(sport_key)
        if not api_sport_key:
            logger.warning(f"Sport {sport_key} not mapped for Odds API")
            return []
        
        url = f"{ODDS_API_BASE}/sports/{api_sport_key}/odds"
        params = {
            "apiKey": self.odds_api_key,
            "regions": "us",  # US bookmakers
            "markets": "h2h,spreads,totals",
            "oddsFormat": "decimal",
            "bookmakers": ",".join(PRIORITY_BOOKMAKERS)
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Filter to specific event if provided
                    if event_id:
                        data = [e for e in data if e.get("id") == event_id]
                    
                    return self._parse_odds_api_response(data)
                    
                elif response.status_code == 401:
                    logger.error("Invalid Odds API key")
                elif response.status_code == 429:
                    logger.warning("Odds API rate limit reached")
                else:
                    logger.error(f"Odds API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error fetching from Odds API: {e}")
        
        return []
    
    def _parse_odds_api_response(self, data: List[Dict]) -> List[Dict]:
        """Parse The Odds API response into our format"""
        bookmakers_data = {}
        
        for event in data:
            for bm in event.get("bookmakers", []):
                bm_key = bm.get("key", "")
                
                if bm_key not in bookmakers_data:
                    bookmakers_data[bm_key] = {
                        "key": bm_key,
                        "title": bm.get("title", bm_key),
                        "last_update": bm.get("last_update", ""),
                        "markets": []
                    }
                
                # Add markets
                for market in bm.get("markets", []):
                    bookmakers_data[bm_key]["markets"].append({
                        "key": market.get("key", ""),
                        "last_update": market.get("last_update", ""),
                        "outcomes": market.get("outcomes", [])
                    })
        
        return list(bookmakers_data.values())
    
    def get_sharp_book_odds(self, multi_book_odds: Dict, team_name: str) -> Optional[Dict]:
        """
        Get odds from the sharpest book (Pinnacle).
        Sharp books have the most accurate odds.
        """
        for bm in multi_book_odds.get("bookmakers", []):
            if bm.get("key") == "pinnacle":
                for market in bm.get("markets", []):
                    if market.get("key") == "h2h":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == team_name:
                                return {
                                    "bookmaker": "pinnacle",
                                    "odds": outcome.get("price"),
                                    "implied_prob": 1 / outcome.get("price", 2) if outcome.get("price", 2) > 0 else 0.5
                                }
        return None
    
    def find_best_odds(self, multi_book_odds: Dict, team_name: str) -> Dict:
        """Find the best available odds for a team across all bookmakers"""
        best_odds = None
        best_book = None
        all_odds = []
        
        for bm in multi_book_odds.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == team_name:
                            price = outcome.get("price", 0)
                            all_odds.append({
                                "book": bm.get("key"),
                                "odds": price
                            })
                            
                            if best_odds is None or price > best_odds:
                                best_odds = price
                                best_book = bm.get("key")
        
        return {
            "best_odds": best_odds,
            "best_book": best_book,
            "all_odds": all_odds,
            "odds_count": len(all_odds)
        }
    
    def calculate_odds_discrepancy(self, multi_book_odds: Dict, team_name: str) -> Dict:
        """
        Calculate odds discrepancy across books.
        Large discrepancies can indicate value.
        """
        odds_info = self.find_best_odds(multi_book_odds, team_name)
        all_odds = [o["odds"] for o in odds_info.get("all_odds", []) if o.get("odds", 0) > 0]
        
        if len(all_odds) < 2:
            return {"discrepancy": 0, "is_significant": False}
        
        max_odds = max(all_odds)
        min_odds = min(all_odds)
        avg_odds = sum(all_odds) / len(all_odds)
        
        discrepancy_pct = (max_odds - min_odds) / avg_odds * 100
        
        return {
            "max_odds": max_odds,
            "min_odds": min_odds,
            "avg_odds": avg_odds,
            "discrepancy": discrepancy_pct,
            "is_significant": discrepancy_pct > 5,  # 5%+ is significant
            "books_count": len(all_odds)
        }


# Singleton instance
_provider = None


def get_multi_book_provider(odds_api_key: str = None) -> MultiBookOddsProvider:
    """Get or create multi-book odds provider"""
    global _provider
    if _provider is None or (odds_api_key and _provider.odds_api_key != odds_api_key):
        _provider = MultiBookOddsProvider(odds_api_key)
    return _provider


async def fetch_multi_book_odds(sport_key: str, event_id: str = None, odds_api_key: str = None) -> Dict:
    """Convenience function to fetch multi-book odds"""
    provider = get_multi_book_provider(odds_api_key)
    return await provider.get_multi_book_odds(sport_key, event_id)
