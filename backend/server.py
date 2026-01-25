from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import httpx
import asyncio
import hashlib
import csv
import io
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Emergent LLM Key
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')

# Events cache to reduce scraping (1 hour cache)
events_cache = {}
CACHE_DURATION_MINUTES = 60

# Last scrape timestamp for status
last_scrape_time = None

# Notification queue for line movement alerts
notification_queue = []

# Bookmakers tracked from OddsPortal
BOOKMAKERS = {
    'bet365': 'bet365',
    'pinnacle': 'Pinnacle',
    'draftkings': 'DraftKings',
    'fanduel': 'FanDuel',
    'betmgm': 'BetMGM',
    'unibet': 'Unibet',
    'betway': 'Betway',
    'betfair': 'Betfair',
    'william_hill': 'William Hill',
    '1xbet': '1xBet',
    'betonlineag': 'BetOnline'
}

# Sportsbook display names (for UI)
SPORTSBOOK_NAMES = {
    'bet365': 'bet365',
    'pinnacle': 'Pinnacle',
    'draftkings': 'DraftKings',
    'fanduel': 'FanDuel',
    'betmgm': 'BetMGM',
    'unibet': 'Unibet',
    'betway': 'Betway',
    'betfair': 'Betfair',
    'william_hill': 'William Hill',
    '1xbet': '1xBet',
    'betonlineag': 'BetOnline',
    'oddsportal_best': 'Best Available'
}

# Market types to analyze
MARKET_TYPES = ['h2h', 'spreads', 'totals']

app = FastAPI()
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic Models
class Sport(BaseModel):
    key: str
    group: str
    title: str
    description: str
    active: bool
    has_outrights: bool

class Outcome(BaseModel):
    name: str
    price: float
    point: Optional[float] = None

class Market(BaseModel):
    key: str
    last_update: Optional[str] = None
    outcomes: List[Outcome]

class Bookmaker(BaseModel):
    key: str
    title: str
    last_update: Optional[str] = None
    markets: List[Market]

class Event(BaseModel):
    id: str
    sport_key: str
    sport_title: str
    commence_time: str
    home_team: str
    away_team: str
    bookmakers: Optional[List[Bookmaker]] = []

class LineMovement(BaseModel):
    event_id: str
    timestamp: str
    bookmaker: str
    market: str
    home_price: float
    away_price: float
    home_point: Optional[float] = None
    away_point: Optional[float] = None

class Prediction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str
    sport_key: str
    home_team: str
    away_team: str
    commence_time: str
    prediction_type: str  # 'moneyline', 'spread', 'total'
    predicted_outcome: str
    confidence: float
    analysis: str
    ai_model: str
    odds_at_prediction: float
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    result: Optional[str] = None  # 'win', 'loss', 'push', 'pending'

class PredictionCreate(BaseModel):
    event_id: str
    sport_key: str
    home_team: str
    away_team: str
    commence_time: str
    prediction_type: str
    predicted_outcome: str
    confidence: float
    analysis: str
    ai_model: str
    odds_at_prediction: float

class AnalysisRequest(BaseModel):
    event_id: str
    home_team: str
    away_team: str
    sport_key: str
    odds_data: Dict[str, Any]
    line_movement: Optional[List[Dict]] = None

class ResultUpdate(BaseModel):
    prediction_id: str
    result: str  # 'win', 'loss', 'push'

# Notification Models
class NotificationPreferences(BaseModel):
    line_movement_alerts: bool = True
    line_movement_threshold: float = 5.0  # Percentage change to trigger alert
    result_alerts: bool = True
    daily_summary: bool = True

class Notification(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # 'line_movement', 'result', 'recommendation'
    title: str
    message: str
    data: Optional[Dict] = None
    read: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# Settings Model
class AppSettings(BaseModel):
    cache_duration_minutes: int = 60
    priority_sports: List[str] = []
    notification_preferences: NotificationPreferences = NotificationPreferences()

# Helper function to create notifications
async def create_notification(notif_type: str, title: str, message: str, data: Dict = None):
    """Create a notification and store in database"""
    notification = Notification(
        type=notif_type,
        title=title,
        message=message,
        data=data or {}
    )
    await db.notifications.insert_one(notification.model_dump())
    notification_queue.append(notification.model_dump())
    return notification

# ==================== SCRAPER STATUS ENDPOINT ====================

@api_router.get("/scraper-status")
async def get_scraper_status():
    """Get OddsPortal scraper status"""
    global last_scrape_time
    
    # Get cached events count
    total_cached = sum(len(events) for events, _ in events_cache.values())
    
    return {
        "source": "oddsportal",
        "status": "active",
        "lastUpdate": last_scrape_time,
        "cachedEvents": total_cached,
        "cacheDuration": CACHE_DURATION_MINUTES,
        "sports": list(events_cache.keys())
    }

# ==================== MANUAL SCRAPE ENDPOINT ====================

@api_router.post("/scrape-odds")
async def manual_scrape_odds(sport_key: str = "basketball_nba"):
    """Manually trigger OddsPortal scraping for a sport"""
    global events_cache, last_scrape_time
    
    try:
        from oddsportal_scraper import scrape_oddsportal_events
        events = await scrape_oddsportal_events(sport_key)
        
        if events:
            # Store in cache
            cache_key = f"{sport_key}_h2h,spreads,totals"
            events_cache[cache_key] = (events, datetime.now(timezone.utc))
            last_scrape_time = datetime.now(timezone.utc).isoformat()
            
            # Store odds snapshots for line movement tracking
            for event in events:
                await store_odds_snapshot(event)
            
            return {
                "message": f"Successfully scraped {len(events)} events from OddsPortal",
                "events": events
            }
        else:
            return {"message": "No events found", "events": []}
            
    except Exception as e:
        logger.error(f"Manual scrape failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

# ==================== NOTIFICATIONS ENDPOINTS ====================

@api_router.get("/notifications")
async def get_notifications(unread_only: bool = False, limit: int = 50):
    """Get notifications"""
    query = {"read": False} if unread_only else {}
    notifications = await db.notifications.find(query, {"_id": 0}).sort("created_at", -1).limit(limit).to_list(limit)
    unread_count = await db.notifications.count_documents({"read": False})
    return {"notifications": notifications, "unread_count": unread_count}

@api_router.put("/notifications/{notif_id}/read")
async def mark_notification_read(notif_id: str):
    """Mark a notification as read"""
    await db.notifications.update_one({"id": notif_id}, {"$set": {"read": True}})
    return {"message": "Notification marked as read"}

@api_router.put("/notifications/read-all")
async def mark_all_notifications_read():
    """Mark all notifications as read"""
    await db.notifications.update_many({}, {"$set": {"read": True}})
    return {"message": "All notifications marked as read"}

@api_router.delete("/notifications/{notif_id}")
async def delete_notification(notif_id: str):
    """Delete a notification"""
    await db.notifications.delete_one({"id": notif_id})
    return {"message": "Notification deleted"}

# ==================== SETTINGS ENDPOINTS ====================

@api_router.get("/settings")
async def get_settings():
    """Get app settings"""
    settings = await db.settings.find_one({}, {"_id": 0})
    if not settings:
        settings = AppSettings().model_dump()
    return settings

@api_router.put("/settings")
async def update_settings(settings: AppSettings):
    """Update app settings"""
    global CACHE_DURATION_MINUTES
    CACHE_DURATION_MINUTES = settings.cache_duration_minutes
    
    await db.settings.update_one(
        {},
        {"$set": settings.model_dump()},
        upsert=True
    )
    return {"message": "Settings updated"}

# ==================== EXPORT ENDPOINTS ====================

@api_router.get("/export/predictions")
async def export_predictions(format: str = "csv"):
    """Export predictions history"""
    predictions = await db.predictions.find({}, {"_id": 0}).to_list(10000)
    
    if format == "json":
        return predictions
    
    # CSV export
    if not predictions:
        return {"message": "No predictions to export"}
    
    output = io.StringIO()
    fieldnames = ['id', 'event_id', 'sport_key', 'home_team', 'away_team', 'commence_time',
                  'prediction_type', 'predicted_outcome', 'confidence', 'odds_at_prediction',
                  'result', 'created_at', 'ai_model']
    
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for pred in predictions:
        writer.writerow(pred)
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions_export.csv"}
    )

@api_router.get("/export/performance-report")
async def export_performance_report():
    """Generate comprehensive performance report"""
    performance = await get_performance()
    scraper_status = await get_scraper_status()
    
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "performance": performance,
        "scraper_status": scraper_status
    }
    
    return report

# AI Analysis function
async def get_ai_analysis(prompt: str, model: str = "gpt-5.2") -> str:
    if not EMERGENT_LLM_KEY:
        return "AI analysis unavailable - no API key configured"
    
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        
        session_id = str(uuid.uuid4())
        system_message = """You are an expert sports betting analyst. Provide concise, actionable betting recommendations.
        Format your response as:
        PICK: [Team/Side]
        CONFIDENCE: [1-10]
        REASONING: [2-3 sentences max]
        
        Focus on: line value, sharp money indicators, and key matchup factors."""
        
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=session_id,
            system_message=system_message
        )
        
        if model == "claude":
            chat.with_model("anthropic", "claude-sonnet-4-5-20250929")
        else:
            chat.with_model("openai", "gpt-5.2")
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        return response
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return f"AI analysis error: {str(e)}"

# Routes
@api_router.get("/")
async def root():
    return {"message": "BetPredictor API v1.0", "status": "running"}

@api_router.get("/sports", response_model=List[Sport])
async def list_sports():
    """Get list of available sports"""
    # Return hardcoded sports list (OddsPortal supported)
    return [
        {"key": "americanfootball_nfl", "group": "American Football", "title": "NFL", "description": "US Football", "active": True, "has_outrights": False},
        {"key": "basketball_nba", "group": "Basketball", "title": "NBA", "description": "US Basketball", "active": True, "has_outrights": False},
        {"key": "baseball_mlb", "group": "Baseball", "title": "MLB", "description": "US Baseball", "active": True, "has_outrights": False},
        {"key": "icehockey_nhl", "group": "Ice Hockey", "title": "NHL", "description": "US Ice Hockey", "active": True, "has_outrights": False},
        {"key": "soccer_epl", "group": "Soccer", "title": "EPL", "description": "English Premier League", "active": True, "has_outrights": False},
        {"key": "soccer_spain_la_liga", "group": "Soccer", "title": "La Liga", "description": "Spanish La Liga", "active": True, "has_outrights": False},
        {"key": "mma_mixed_martial_arts", "group": "MMA", "title": "MMA", "description": "Mixed Martial Arts", "active": True, "has_outrights": False},
    ]

@api_router.get("/events/{sport_key}")
async def get_events(sport_key: str, markets: str = "h2h,spreads,totals", force_refresh: bool = False):
    """Get events with odds for a specific sport from OddsPortal (free scraping)"""
    global events_cache, last_scrape_time
    
    cache_key = f"{sport_key}_{markets}"
    now = datetime.now(timezone.utc)
    
    # Check cache first (unless force refresh)
    if not force_refresh and cache_key in events_cache:
        cached_data, cached_time = events_cache[cache_key]
        cache_age_minutes = (now - cached_time).total_seconds() / 60
        if cache_age_minutes < CACHE_DURATION_MINUTES:
            logger.info(f"Using cached data for {sport_key} (age: {cache_age_minutes:.1f} min)")
            return cached_data
    
    # Scrape from OddsPortal
    try:
        from oddsportal_scraper import scrape_oddsportal_events
        events = await scrape_oddsportal_events(sport_key)
        
        if events:
            # Store odds snapshots for line movement tracking
            for event in events:
                await store_odds_snapshot(event)
            
            # Cache the results
            events_cache[cache_key] = (events, now)
            last_scrape_time = now.isoformat()
            logger.info(f"Cached {len(events)} events from OddsPortal for {sport_key}")
            return events
        else:
            logger.warning(f"OddsPortal returned no events for {sport_key}")
            # Return cached if available
            if cache_key in events_cache:
                return events_cache[cache_key][0]
            return []
            
    except Exception as e:
        logger.error(f"OddsPortal scraping failed: {e}")
        # Return cached if available
        if cache_key in events_cache:
            return events_cache[cache_key][0]
        return []

@api_router.get("/event/{event_id}")
async def get_event_details(event_id: str, sport_key: str = "basketball_nba"):
    """Get detailed odds for a specific event"""
    # Try multiple sports if not found
    sports_to_try = [sport_key, "basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl", "soccer_epl"]
    
    for sk in sports_to_try:
        events = await get_events(sk)
        for event in events:
            if event.get('id') == event_id:
                return event
    
    raise HTTPException(status_code=404, detail="Event not found")

@api_router.get("/line-movement/{event_id}")
async def get_line_movement(event_id: str, sport_key: str = "basketball_nba"):
    """Get line movement history for an event including opening odds and hourly snapshots"""
    
    # Get opening odds
    opening = await db.opening_odds.find_one({"event_id": event_id}, {"_id": 0})
    
    # Get all hourly snapshots sorted by timestamp
    snapshots = await db.odds_history.find(
        {"event_id": event_id},
        {"_id": 0}
    ).sort("timestamp", 1).to_list(500)
    
    # Group snapshots by bookmaker for better visualization
    by_bookmaker = {}
    all_timestamps = set()
    
    for snap in snapshots:
        bm = snap.get("bookmaker", "average")
        if bm not in by_bookmaker:
            by_bookmaker[bm] = {
                "bookmaker": bm,
                "bookmaker_title": snap.get("bookmaker_title", bm),
                "snapshots": []
            }
        
        by_bookmaker[bm]["snapshots"].append({
            "timestamp": snap.get("timestamp"),
            "home_odds": snap.get("home_odds"),
            "away_odds": snap.get("away_odds")
        })
        all_timestamps.add(snap.get("timestamp"))
    
    # Calculate average odds across all bookmakers at each timestamp
    chart_data = []
    for snap in snapshots:
        ts = snap.get("timestamp")
        home = snap.get("home_odds")
        away = snap.get("away_odds")
        
        if home and away:
            # Check if we already have this timestamp
            existing = next((c for c in chart_data if c["timestamp"] == ts), None)
            if existing:
                # Average with existing
                existing["home_odds"] = (existing["home_odds"] + home) / 2
                existing["away_odds"] = (existing["away_odds"] + away) / 2
            else:
                chart_data.append({
                    "timestamp": ts,
                    "home_odds": home,
                    "away_odds": away
                })
    
    # Sort chart data by timestamp
    chart_data.sort(key=lambda x: x["timestamp"])
    
    # Get event info
    event_info = None
    events = await get_events(sport_key)
    for event in events:
        if event.get("id") == event_id:
            event_info = {
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "commence_time": event.get("commence_time")
            }
            break
    
    return {
        "event_id": event_id,
        "event_info": event_info,
        "opening_odds": opening,
        "bookmakers": list(by_bookmaker.values()),
        "chart_data": chart_data,
        "total_snapshots": len(snapshots)
    }

@api_router.post("/analyze")
async def analyze_game(request: AnalysisRequest):
    """Get AI analysis for a game"""
    prompt = f"""Analyze this {request.sport_key} game:
    
{request.home_team} vs {request.away_team}

Current Odds Data:
{format_odds_for_analysis(request.odds_data)}

{"Line Movement History:" + str(request.line_movement) if request.line_movement else ""}

Provide analysis on:
1. Which side has value based on odds comparison
2. Why lines may be moving (if applicable)
3. Sharp vs public money indicators
4. Your recommended bet with reasoning
5. Confidence level (1-10)"""

    # Get analysis from both models
    gpt_analysis = await get_ai_analysis(prompt, "gpt-5.2")
    claude_analysis = await get_ai_analysis(prompt, "claude")
    
    return {
        "event_id": request.event_id,
        "home_team": request.home_team,
        "away_team": request.away_team,
        "gpt_analysis": gpt_analysis,
        "claude_analysis": claude_analysis,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@api_router.get("/recommendations")
async def get_recommendations(sport_key: Optional[str] = None, limit: int = 10, min_odds: float = 1.5):
    """Get AI-generated bet recommendations - sorted by confidence, filtered by min odds"""
    query = {
        "result": "pending",
        "odds_at_prediction": {"$gte": min_odds, "$lte": 20}
    }
    if sport_key:
        query["sport_key"] = sport_key
    
    # Sort by confidence (highest first)
    predictions = await db.predictions.find(
        query,
        {"_id": 0}
    ).sort("confidence", -1).limit(limit).to_list(limit)
    
    return predictions

@api_router.post("/recommendations")
async def create_recommendation(prediction: PredictionCreate):
    """Create a new bet recommendation"""
    pred_dict = prediction.model_dump()
    pred_obj = Prediction(**pred_dict)
    pred_obj.result = "pending"
    
    await db.predictions.insert_one(pred_obj.model_dump())
    return pred_obj

@api_router.post("/generate-recommendations")
async def generate_recommendations(sport_key: str):
    """Generate AI recommendations for upcoming events"""
    events = await get_events(sport_key)
    
    if not events:
        return {"message": "No events found", "recommendations": []}
    
    recommendations = []
    
    # Analyze top 3 events
    for event in events[:3]:
        odds_data = {
            "bookmakers": event.get("bookmakers", []),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team")
        }
        
        analysis_request = AnalysisRequest(
            event_id=event.get("id"),
            home_team=event.get("home_team"),
            away_team=event.get("away_team"),
            sport_key=sport_key,
            odds_data=odds_data
        )
        
        analysis = await analyze_game(analysis_request)
        
        # Parse AI recommendation and create prediction
        # This is simplified - in production you'd parse the AI response more carefully
        best_odds = get_best_odds(event.get("bookmakers", []))
        
        prediction = PredictionCreate(
            event_id=event.get("id"),
            sport_key=sport_key,
            home_team=event.get("home_team"),
            away_team=event.get("away_team"),
            commence_time=event.get("commence_time"),
            prediction_type="moneyline",
            predicted_outcome=event.get("home_team"),  # Simplified
            confidence=0.65,
            analysis=analysis.get("gpt_analysis", ""),
            ai_model="gpt-5.2",
            odds_at_prediction=best_odds.get("home_price", 0)
        )
        
        saved = await create_recommendation(prediction)
        recommendations.append(saved)
    
    return {"message": f"Generated {len(recommendations)} recommendations", "recommendations": recommendations}

# Auto-generate recommendations for dashboard
async def auto_generate_recommendations():
    """Automatically generate recommendations for top events across sports"""
    logger.info("Starting auto-recommendation generation...")
    sports_to_analyze = ["basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl", "soccer_epl"]
    
    for sport_key in sports_to_analyze:
        try:
            events = await get_events(sport_key)
            if not events:
                continue
            
            # Analyze top 2 events per sport
            for event in events[:2]:
                event_id = event.get("id")
                
                # Check if we already have a recommendation for this event
                existing = await db.predictions.find_one({"event_id": event_id, "result": "pending"})
                if existing:
                    logger.info(f"Skipping {event_id} - already has prediction")
                    continue
                
                # Generate AI analysis
                odds_data = {"bookmakers": event.get("bookmakers", [])}
                analysis = await generate_smart_recommendation(event, sport_key, odds_data)
                
                if analysis:
                    logger.info(f"Generated recommendation for {event.get('home_team')} vs {event.get('away_team')}")
                    
        except Exception as e:
            logger.error(f"Error generating recommendations for {sport_key}: {e}")
    
    logger.info("Auto-recommendation generation complete")

async def generate_smart_recommendation(event: dict, sport_key: str, odds_data: dict) -> Optional[dict]:
    """Generate a smart recommendation analyzing ALL markets (moneyline, spreads, totals)"""
    try:
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        bookmakers = odds_data.get("bookmakers", [])
        
        # Collect all market data
        all_markets = {
            'h2h': {'home': [], 'away': []},
            'spreads': {'home': [], 'away': []},
            'totals': {'over': [], 'under': []}
        }
        
        for bm in bookmakers:
            bm_name = SPORTSBOOK_NAMES.get(bm.get("key"), bm.get("title"))
            for market in bm.get("markets", []):
                market_key = market.get("key")
                
                if market_key == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == home_team:
                            all_markets['h2h']['home'].append({
                                'price': outcome.get("price", 1),
                                'book': bm_name
                            })
                        elif outcome.get("name") == away_team:
                            all_markets['h2h']['away'].append({
                                'price': outcome.get("price", 1),
                                'book': bm_name
                            })
                
                elif market_key == "spreads":
                    for outcome in market.get("outcomes", []):
                        point = outcome.get("point", 0)
                        if outcome.get("name") == home_team:
                            all_markets['spreads']['home'].append({
                                'price': outcome.get("price", 1),
                                'point': point,
                                'book': bm_name
                            })
                        elif outcome.get("name") == away_team:
                            all_markets['spreads']['away'].append({
                                'price': outcome.get("price", 1),
                                'point': point,
                                'book': bm_name
                            })
                
                elif market_key == "totals":
                    for outcome in market.get("outcomes", []):
                        point = outcome.get("point", 0)
                        if outcome.get("name") == "Over":
                            all_markets['totals']['over'].append({
                                'price': outcome.get("price", 1),
                                'point': point,
                                'book': bm_name
                            })
                        elif outcome.get("name") == "Under":
                            all_markets['totals']['under'].append({
                                'price': outcome.get("price", 1),
                                'point': point,
                                'book': bm_name
                            })
        
        # Find best odds for each market
        best_h2h_home = max(all_markets['h2h']['home'], key=lambda x: x['price'], default={'price': 1, 'book': 'N/A'})
        best_h2h_away = max(all_markets['h2h']['away'], key=lambda x: x['price'], default={'price': 1, 'book': 'N/A'})
        best_spread_home = max(all_markets['spreads']['home'], key=lambda x: x['price'], default={'price': 1, 'point': 0, 'book': 'N/A'})
        best_spread_away = max(all_markets['spreads']['away'], key=lambda x: x['price'], default={'price': 1, 'point': 0, 'book': 'N/A'})
        best_over = max(all_markets['totals']['over'], key=lambda x: x['price'], default={'price': 1, 'point': 0, 'book': 'N/A'})
        best_under = max(all_markets['totals']['under'], key=lambda x: x['price'], default={'price': 1, 'point': 0, 'book': 'N/A'})
        
        # Build comprehensive prompt for AI analyzing ALL markets
        prompt = f"""Analyze this {sport_key.replace('_', ' ')} matchup and recommend the BEST VALUE bet across all markets:

{home_team} vs {away_team}

MONEYLINE:
- {home_team}: {best_h2h_home['price']:.2f} ({best_h2h_home['book']})
- {away_team}: {best_h2h_away['price']:.2f} ({best_h2h_away['book']})

SPREAD:
- {home_team} {best_spread_home.get('point', 0):+.1f}: {best_spread_home['price']:.2f} ({best_spread_home['book']})
- {away_team} {best_spread_away.get('point', 0):+.1f}: {best_spread_away['price']:.2f} ({best_spread_away['book']})

TOTALS:
- Over {best_over.get('point', 0)}: {best_over['price']:.2f} ({best_over['book']})
- Under {best_under.get('point', 0)}: {best_under['price']:.2f} ({best_under['book']})

Analyze ALL markets and provide:
MARKET: [moneyline/spread/total]
PICK: [Exact selection - e.g., "Team Name" or "Over 45.5"]
ODDS: [Best odds available]
CONFIDENCE: [1-10]
REASONING: [2-3 sentences on why this is the best value bet]"""

        # Get AI analysis
        analysis_text = await get_ai_analysis(prompt, "gpt-5.2")
        
        # Parse the response
        confidence = 0.6
        predicted_outcome = home_team
        prediction_type = "moneyline"
        odds_at_prediction = best_h2h_home['price']
        
        analysis_lower = analysis_text.lower()
        
        # Extract confidence
        try:
            import re
            conf_match = re.search(r'confidence[:\s]*(\d+)', analysis_lower)
            if conf_match:
                confidence = int(conf_match.group(1)) / 10
        except Exception:
            pass
        
        # Determine market type and pick
        if "market:" in analysis_lower:
            market_section = analysis_lower.split("market:")[1][:50]
            if "spread" in market_section:
                prediction_type = "spread"
            elif "total" in market_section or "over" in market_section or "under" in market_section:
                prediction_type = "total"
        
        # Extract pick
        if "pick:" in analysis_lower:
            pick_section = analysis_lower.split("pick:")[1][:100]
            
            if prediction_type == "total":
                if "over" in pick_section:
                    predicted_outcome = f"Over {best_over.get('point', 0)}"
                    odds_at_prediction = best_over['price']
                elif "under" in pick_section:
                    predicted_outcome = f"Under {best_under.get('point', 0)}"
                    odds_at_prediction = best_under['price']
            elif prediction_type == "spread":
                if away_team.lower() in pick_section:
                    predicted_outcome = f"{away_team} {best_spread_away.get('point', 0):+.1f}"
                    odds_at_prediction = best_spread_away['price']
                else:
                    predicted_outcome = f"{home_team} {best_spread_home.get('point', 0):+.1f}"
                    odds_at_prediction = best_spread_home['price']
            else:  # moneyline
                if away_team.lower() in pick_section:
                    predicted_outcome = away_team
                    odds_at_prediction = best_h2h_away['price']
                else:
                    predicted_outcome = home_team
                    odds_at_prediction = best_h2h_home['price']
        
        # Skip if odds below 1.5 or above 20 (unreasonable)
        if odds_at_prediction < 1.5 or odds_at_prediction > 20:
            logger.info(f"Skipping {home_team} vs {away_team} - odds outside 1.5-20 range: {odds_at_prediction}")
            return None
        
        # Create prediction
        prediction = PredictionCreate(
            event_id=event.get("id"),
            sport_key=sport_key,
            home_team=home_team,
            away_team=away_team,
            commence_time=event.get("commence_time"),
            prediction_type=prediction_type,
            predicted_outcome=predicted_outcome,
            confidence=min(max(confidence, 0.3), 0.95),  # Between 30-95%
            analysis=analysis_text,
            ai_model="gpt-5.2",
            odds_at_prediction=odds_at_prediction
        )
        
        # Save to database
        await create_recommendation(prediction)
        return prediction.model_dump()
        
    except Exception as e:
        logger.error(f"Error in generate_smart_recommendation: {e}")
        return None

# Update recommendations based on line movement
async def update_recommendations_on_line_movement():
    """Check line movement and update recommendations if significant change"""
    logger.info("Checking line movements and updating recommendations...")
    
    try:
        # Get settings for threshold
        settings = await db.settings.find_one({})
        threshold = 5.0
        if settings and settings.get('notification_preferences'):
            threshold = settings['notification_preferences'].get('line_movement_threshold', 5.0)
        
        # Get pending predictions
        pending = await db.predictions.find({"result": "pending"}, {"_id": 0}).to_list(50)
        
        for prediction in pending:
            event_id = prediction.get("event_id")
            sport_key = prediction.get("sport_key")
            original_odds = prediction.get("odds_at_prediction", 0)
            
            if not event_id or not sport_key:
                continue
            
            # Fetch current odds
            events = await get_events(sport_key)
            current_event = None
            for e in events:
                if e.get("id") == event_id:
                    current_event = e
                    break
            
            if not current_event:
                continue
            
            # Get current best odds for predicted outcome
            predicted_team = prediction.get("predicted_outcome")
            current_best = 1.0
            
            for bm in current_event.get("bookmakers", []):
                for market in bm.get("markets", []):
                    if market.get("key") == "h2h":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == predicted_team and outcome.get("price", 1) > current_best:
                                current_best = outcome.get("price")
            
            # Check for significant line movement
            if original_odds > 0:
                change_pct = abs(current_best - original_odds) / original_odds * 100
                
                if change_pct >= threshold:
                    # Line moved significantly - add note to analysis
                    new_analysis = prediction.get("analysis", "") + f"\n\n[LINE MOVEMENT UPDATE: Odds moved from {original_odds:.2f} to {current_best:.2f} ({change_pct:.1f}% change)]"
                    
                    # Adjust confidence based on movement direction
                    new_confidence = prediction.get("confidence", 0.6)
                    movement_direction = "favorable" if current_best > original_odds else "adverse"
                    
                    if current_best > original_odds:
                        # Line moving in our favor (better value)
                        new_confidence = min(new_confidence + 0.05, 0.95)
                        new_analysis += " [Favorable movement - increased confidence]"
                    else:
                        # Line moving against us
                        new_confidence = max(new_confidence - 0.1, 0.3)
                        new_analysis += " [Adverse movement - decreased confidence]"
                    
                    await db.predictions.update_one(
                        {"id": prediction.get("id")},
                        {"$set": {
                            "analysis": new_analysis,
                            "confidence": new_confidence,
                            "current_odds": current_best,
                            "line_movement_pct": change_pct,
                            "last_updated": datetime.now(timezone.utc).isoformat()
                        }}
                    )
                    logger.info(f"Updated prediction {prediction.get('id')} - line moved {change_pct:.1f}%")
                    
                    # Create notification for significant line movement
                    if settings and settings.get('notification_preferences', {}).get('line_movement_alerts', True):
                        await create_notification(
                            "line_movement",
                            f"Line Movement Alert: {prediction.get('home_team')} vs {prediction.get('away_team')}",
                            f"Odds moved {change_pct:.1f}% ({movement_direction}): {original_odds:.2f} → {current_best:.2f}",
                            {
                                "prediction_id": prediction.get("id"),
                                "event_id": event_id,
                                "original_odds": original_odds,
                                "current_odds": current_best,
                                "change_pct": change_pct,
                                "direction": movement_direction
                            }
                        )
                    
    except Exception as e:
        logger.error(f"Error in update_recommendations_on_line_movement: {e}")

@api_router.put("/result")
async def update_result(update: ResultUpdate):
    """Update the result of a prediction"""
    result = await db.predictions.update_one(
        {"id": update.prediction_id},
        {"$set": {"result": update.result}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return {"message": "Result updated successfully"}

@api_router.get("/performance")
async def get_performance(sport_key: Optional[str] = None, days: int = 30):
    """Get performance statistics"""
    query = {"result": {"$ne": "pending"}}
    if sport_key:
        query["sport_key"] = sport_key
    
    predictions = await db.predictions.find(query, {"_id": 0}).to_list(1000)
    
    total = len(predictions)
    wins = len([p for p in predictions if p.get("result") == "win"])
    losses = len([p for p in predictions if p.get("result") == "loss"])
    pushes = len([p for p in predictions if p.get("result") == "push"])
    
    win_rate = (wins / total * 100) if total > 0 else 0
    
    # Calculate ROI (simplified)
    roi = calculate_roi(predictions)
    
    # Get by sport breakdown
    by_sport = {}
    for p in predictions:
        sk = p.get("sport_key", "unknown")
        if sk not in by_sport:
            by_sport[sk] = {"wins": 0, "losses": 0, "pushes": 0, "total": 0}
        by_sport[sk]["total"] += 1
        if p.get("result") == "win":
            by_sport[sk]["wins"] += 1
        elif p.get("result") == "loss":
            by_sport[sk]["losses"] += 1
        else:
            by_sport[sk]["pushes"] += 1
    
    # Recent predictions
    recent = sorted(predictions, key=lambda x: x.get("created_at", ""), reverse=True)[:20]
    
    return {
        "total_predictions": total,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(win_rate, 2),
        "roi": round(roi, 2),
        "by_sport": by_sport,
        "recent_predictions": recent
    }

# Auto-tracking system for results
@api_router.post("/check-results")
async def check_and_update_results(background_tasks: BackgroundTasks):
    """Check for completed events and update prediction results"""
    background_tasks.add_task(auto_check_results)
    return {"message": "Result checking started in background"}

async def auto_check_results():
    """Background task to check event results and update predictions"""
    try:
        # Get all pending predictions
        pending = await db.predictions.find(
            {"result": "pending"},
            {"_id": 0}
        ).to_list(100)
        
        for prediction in pending:
            event_id = prediction.get("event_id")
            commence_time_str = prediction.get("commence_time")
            
            if not commence_time_str:
                continue
            
            try:
                commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                
                # Check if 5 hours have passed since event start
                hours_since_start = (now - commence_time).total_seconds() / 3600
                
                if hours_since_start >= 5:
                    # Try to fetch result from scores API
                    result = await fetch_event_result(
                        prediction.get("sport_key"),
                        event_id,
                        prediction.get("home_team"),
                        prediction.get("away_team"),
                        prediction.get("predicted_outcome"),
                        prediction.get("prediction_type")
                    )
                    
                    if result:
                        await db.predictions.update_one(
                            {"id": prediction.get("id")},
                            {"$set": {"result": result, "result_updated_at": now.isoformat()}}
                        )
                        logger.info(f"Updated prediction {prediction.get('id')} with result: {result}")
            except Exception as e:
                logger.error(f"Error processing prediction {prediction.get('id')}: {e}")
                
    except Exception as e:
        logger.error(f"Error in auto_check_results: {e}")

async def fetch_event_result(sport_key: str, event_id: str, home_team: str, away_team: str, predicted_outcome: str, prediction_type: str) -> Optional[str]:
    """Fetch event result - currently requires manual update as OddsPortal doesn't provide scores"""
    # OddsPortal doesn't provide live scores, results need to be updated manually
    # or we could integrate a free scores API in the future
    return None

@api_router.get("/scores/{sport_key}")
async def get_scores(sport_key: str, days_from: int = 3):
    """Get recent scores - placeholder for future integration"""
    return {
        "message": "Scores tracking coming soon. Use the predictions page to manually update results.",
        "scores": []
    }
@api_router.get("/odds-comparison/{event_id}")
async def get_odds_comparison(event_id: str, sport_key: str = "basketball_nba"):
    """Get odds comparison across all sportsbooks for an event"""
    # Try multiple sports if not found
    sports_to_try = [sport_key, "basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl", "soccer_epl"]
    
    for sk in sports_to_try:
        events = await get_events(sk)
        for event in events:
            if event.get("id") == event_id:
                comparison = format_odds_comparison(event)
                return comparison
    
    raise HTTPException(status_code=404, detail="Event not found")

# Helper functions
async def store_odds_snapshot(event: dict):
    """Store odds snapshot for line movement tracking - stores opening odds and hourly snapshots"""
    event_id = event.get("id")
    timestamp = datetime.now(timezone.utc).isoformat()
    home_team = event.get("home_team", "home")
    away_team = event.get("away_team", "away")
    
    # Check if we have opening odds stored for this event
    existing_opening = await db.opening_odds.find_one({"event_id": event_id})
    
    for bookmaker in event.get("bookmakers", []):
        bm_key = bookmaker.get("key")
        bm_title = bookmaker.get("title", bm_key)
        
        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "h2h")
            outcomes = market.get("outcomes", [])
            
            # Extract home and away odds
            home_odds = None
            away_odds = None
            
            for outcome in outcomes:
                name = outcome.get("name", "").lower()
                price = outcome.get("price")
                
                if name == "home" or name == home_team.lower():
                    home_odds = price
                elif name == "away" or name == away_team.lower():
                    away_odds = price
            
            # If not found by name, use position
            if home_odds is None and len(outcomes) >= 1:
                home_odds = outcomes[0].get("price")
            if away_odds is None and len(outcomes) >= 2:
                away_odds = outcomes[1].get("price")
            
            # Store opening odds if this is first time seeing this event
            if not existing_opening and home_odds and away_odds:
                await db.opening_odds.insert_one({
                    "event_id": event_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "bookmaker": bm_key,
                    "bookmaker_title": bm_title,
                    "market": market_key,
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "timestamp": timestamp
                })
                logger.info(f"Stored opening odds for {home_team} vs {away_team}: {home_odds}/{away_odds}")
            
            # Store hourly snapshot for line movement tracking
            snapshot = {
                "event_id": event_id,
                "home_team": home_team,
                "away_team": away_team,
                "timestamp": timestamp,
                "bookmaker": bm_key,
                "bookmaker_title": bm_title,
                "market": market_key,
                "home_odds": home_odds,
                "away_odds": away_odds,
                "outcomes": outcomes
            }
            await db.odds_history.insert_one(snapshot)

def format_odds_for_analysis(odds_data: dict) -> str:
    """Format odds data for AI analysis (decimal format)"""
    output = []
    for bookmaker in odds_data.get("bookmakers", [])[:5]:
        output.append(f"\n{bookmaker.get('title', 'Unknown')}:")
        for market in bookmaker.get("markets", []):
            market_name = market.get("key", "")
            outcomes = market.get("outcomes", [])
            for outcome in outcomes:
                price = outcome.get("price", 0)
                point = outcome.get("point", "")
                point_str = f" ({point})" if point else ""
                # Format as decimal odds
                output.append(f"  {market_name}: {outcome.get('name')} @ {price:.2f}{point_str}")
    return "\n".join(output)

def get_best_odds(bookmakers: list) -> dict:
    """Get best odds from all bookmakers"""
    best_home = {"price": -99999, "bookmaker": ""}
    best_away = {"price": -99999, "bookmaker": ""}
    
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market.get("key") == "h2h":
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price", 0)
                    if price > best_home["price"]:
                        best_home = {"price": price, "bookmaker": bm.get("title")}
    
    return {"home_price": best_home["price"], "away_price": best_away["price"]}

def format_odds_comparison(event: dict) -> dict:
    """Format odds comparison across sportsbooks"""
    comparison = {
        "event_id": event.get("id"),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
        "commence_time": event.get("commence_time"),
        "h2h": [],
        "spreads": [],
        "totals": []
    }
    
    for bm in event.get("bookmakers", []):
        bm_data = {
            "bookmaker": bm.get("key"),
            "title": SPORTSBOOK_NAMES.get(bm.get("key"), bm.get("title")),
            "last_update": bm.get("last_update")
        }
        
        for market in bm.get("markets", []):
            market_key = market.get("key")
            outcomes = market.get("outcomes", [])
            
            if market_key == "h2h":
                comparison["h2h"].append({
                    **bm_data,
                    "outcomes": outcomes
                })
            elif market_key == "spreads":
                comparison["spreads"].append({
                    **bm_data,
                    "outcomes": outcomes
                })
            elif market_key == "totals":
                comparison["totals"].append({
                    **bm_data,
                    "outcomes": outcomes
                })
    
    return comparison

def calculate_roi(predictions: list) -> float:
    """Calculate ROI from predictions (decimal odds format)"""
    if not predictions:
        return 0.0
    
    total_wagered = len(predictions) * 100  # Assume $100 per bet
    total_returned = 0
    
    for p in predictions:
        if p.get("result") == "win":
            odds = p.get("odds_at_prediction", 1.91)  # Default decimal odds
            # Decimal odds: profit = stake * (odds - 1), return = stake * odds
            if odds >= 1:  # Decimal format
                total_returned += 100 * odds
            else:
                # Handle legacy American odds format
                if odds > 0:
                    profit = 100 * (odds / 100)
                else:
                    profit = 100 * (100 / abs(odds))
                total_returned += 100 + profit
        elif p.get("result") == "push":
            total_returned += 100
    
    roi = ((total_returned - total_wagered) / total_wagered) * 100 if total_wagered > 0 else 0
    return roi

async def get_mock_events(sport_key: str):
    """Generate mock events when API is unavailable - with caching for consistent IDs"""
    global mock_events_cache
    import random
    
    # Check cache first (cache for 5 minutes)
    cache_key = sport_key
    if cache_key in mock_events_cache:
        cached_data, cached_time = mock_events_cache[cache_key]
        if (datetime.now(timezone.utc) - cached_time).total_seconds() < 300:
            return cached_data
    
    teams = {
        "americanfootball_nfl": [
            ("Kansas City Chiefs", "Buffalo Bills"),
            ("Philadelphia Eagles", "Dallas Cowboys"),
            ("San Francisco 49ers", "Detroit Lions"),
            ("Baltimore Ravens", "Cincinnati Bengals"),
        ],
        "basketball_nba": [
            ("Boston Celtics", "Miami Heat"),
            ("Denver Nuggets", "Los Angeles Lakers"),
            ("Milwaukee Bucks", "Philadelphia 76ers"),
            ("Golden State Warriors", "Phoenix Suns"),
        ],
        "baseball_mlb": [
            ("New York Yankees", "Boston Red Sox"),
            ("Los Angeles Dodgers", "San Francisco Giants"),
            ("Houston Astros", "Texas Rangers"),
            ("Atlanta Braves", "Philadelphia Phillies"),
        ],
        "icehockey_nhl": [
            ("Edmonton Oilers", "Florida Panthers"),
            ("Colorado Avalanche", "Dallas Stars"),
            ("Vegas Golden Knights", "Los Angeles Kings"),
            ("Toronto Maple Leafs", "Boston Bruins"),
        ],
        "soccer_epl": [
            ("Manchester City", "Liverpool"),
            ("Arsenal", "Chelsea"),
            ("Manchester United", "Tottenham"),
            ("Newcastle", "Brighton"),
        ],
    }
    
    sport_teams = teams.get(sport_key, teams["basketball_nba"])
    events = []
    
    # Use deterministic seed based on sport key for consistent IDs
    random.seed(hash(sport_key) % 10000)
    
    for i, (home, away) in enumerate(sport_teams):
        # Create deterministic event ID based on sport and teams
        event_hash = hashlib.md5(f"{sport_key}_{home}_{away}".encode()).hexdigest()[:8]
        event_id = f"mock_{sport_key}_{i}_{event_hash}"
        
        # Generate mock odds
        bookmakers = []
        for bm_key, bm_name in SPORTSBOOK_NAMES.items():
            base_home = random.randint(-200, 200)
            base_away = -base_home + random.randint(-20, 20)
            spread = round(random.uniform(-10, 10) * 2) / 2
            total = random.randint(40, 55) + 0.5
            
            bookmakers.append({
                "key": bm_key,
                "title": bm_name,
                "last_update": datetime.now(timezone.utc).isoformat(),
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": datetime.now(timezone.utc).isoformat(),
                        "outcomes": [
                            {"name": home, "price": base_home + random.randint(-15, 15)},
                            {"name": away, "price": base_away + random.randint(-15, 15)}
                        ]
                    },
                    {
                        "key": "spreads",
                        "last_update": datetime.now(timezone.utc).isoformat(),
                        "outcomes": [
                            {"name": home, "price": -110 + random.randint(-5, 5), "point": spread},
                            {"name": away, "price": -110 + random.randint(-5, 5), "point": -spread}
                        ]
                    },
                    {
                        "key": "totals",
                        "last_update": datetime.now(timezone.utc).isoformat(),
                        "outcomes": [
                            {"name": "Over", "price": -110 + random.randint(-5, 5), "point": total},
                            {"name": "Under", "price": -110 + random.randint(-5, 5), "point": total}
                        ]
                    }
                ]
            })
        
        events.append({
            "id": event_id,
            "sport_key": sport_key,
            "sport_title": sport_key.replace("_", " ").title(),
            "commence_time": (datetime.now(timezone.utc).replace(hour=19, minute=0, second=0)).isoformat(),
            "home_team": home,
            "away_team": away,
            "bookmakers": bookmakers
        })
    
    # Reset random seed
    random.seed()
    
    # Cache the result
    mock_events_cache[cache_key] = (events, datetime.now(timezone.utc))
    
    return events

def generate_mock_line_movement(event_id: str):
    """Generate mock line movement data"""
    import random
    
    movements = []
    base_time = datetime.now(timezone.utc)
    base_home_price = random.randint(-200, 200)
    
    for i in range(24):
        hour_offset = 24 - i
        # Use timedelta for proper hour calculation
        timestamp = (base_time - timedelta(hours=hour_offset)).isoformat()
        
        # Simulate line movement
        home_drift = random.randint(-20, 20)
        
        for bm_key in list(SPORTSBOOK_NAMES.keys())[:5]:
            movements.append({
                "event_id": event_id,
                "timestamp": timestamp,
                "bookmaker": bm_key,
                "bookmaker_title": SPORTSBOOK_NAMES.get(bm_key, bm_key),
                "market": "h2h",
                "outcomes": [
                    {"name": "Home", "price": base_home_price + home_drift + random.randint(-10, 10)},
                    {"name": "Away", "price": -(base_home_price + home_drift) + random.randint(-10, 10)}
                ]
            })
        
        base_home_price += random.randint(-10, 10)
    
    return movements

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background task scheduler for auto-checking results
async def scheduled_result_checker():
    """Background task that runs every 2 hours to check for completed events (to save API calls)"""
    while True:
        try:
            await asyncio.sleep(7200)  # Run every 2 hours
            logger.info("Running scheduled result check...")
            await auto_check_results()
        except Exception as e:
            logger.error(f"Scheduled result checker error: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying

# Background task for line movement checking and recommendation updates
async def scheduled_line_movement_checker():
    """Background task that runs every hour to check line movements"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            logger.info("Running scheduled line movement check...")
            await update_recommendations_on_line_movement()
        except Exception as e:
            logger.error(f"Scheduled line movement checker error: {e}")
            await asyncio.sleep(60)

# Background task for auto-generating recommendations (reduced frequency to save API calls)
async def scheduled_recommendation_generator():
    """Background task that generates recommendations every 4 hours"""
    # Wait 60 seconds on startup to let services initialize
    await asyncio.sleep(60)
    
    while True:
        try:
            logger.info("Running scheduled recommendation generation...")
            await auto_generate_recommendations()
            await asyncio.sleep(14400)  # Run every 4 hours
        except Exception as e:
            logger.error(f"Scheduled recommendation generator error: {e}")
            await asyncio.sleep(300)

# Background task for hourly OddsPortal odds scraping
async def scheduled_oddsportal_scraper():
    """Background task that scrapes OddsPortal every hour for odds updates and line movement tracking"""
    global last_scrape_time
    
    # Wait 2 minutes on startup
    await asyncio.sleep(120)
    
    sports_to_scrape = ["basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl", "soccer_epl"]
    
    while True:
        try:
            logger.info("Running hourly OddsPortal scraping...")
            
            for sport_key in sports_to_scrape:
                try:
                    from oddsportal_scraper import scrape_oddsportal_events
                    events = await scrape_oddsportal_events(sport_key)
                    
                    if events:
                        # Store odds snapshots for line movement tracking
                        for event in events:
                            await store_odds_snapshot(event)
                        
                        # Update cache
                        cache_key = f"{sport_key}_h2h,spreads,totals"
                        events_cache[cache_key] = (events, datetime.now(timezone.utc))
                        last_scrape_time = datetime.now(timezone.utc).isoformat()
                        
                        logger.info(f"Scraped {len(events)} events for {sport_key}")
                    
                    await asyncio.sleep(10)  # Small delay between sports
                    
                except Exception as e:
                    logger.error(f"Error scraping {sport_key}: {e}")
            
            logger.info("Hourly OddsPortal scraping complete")
            
            await asyncio.sleep(3600)  # Run every hour
            
        except Exception as e:
            logger.error(f"Scheduled OddsPortal scraper error: {e}")
            await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup"""
    # Start the scheduled result checker
    asyncio.create_task(scheduled_result_checker())
    logger.info("Started background result checker - runs every 2 hours")
    
    # Start line movement checker  
    asyncio.create_task(scheduled_line_movement_checker())
    logger.info("Started line movement checker - runs every hour")
    
    # Start recommendation generator
    asyncio.create_task(scheduled_recommendation_generator())
    logger.info("Started recommendation generator - runs every 6 hours")
    
    # Start OddsPortal scraper
    asyncio.create_task(scheduled_oddsportal_scraper())
    logger.info("Started OddsPortal scraper - runs every hour")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
