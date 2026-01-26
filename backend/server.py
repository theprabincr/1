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
import re

# Import ESPN data provider for REAL odds
from espn_data_provider import (
    fetch_espn_events_with_odds,
    get_comprehensive_matchup_data
)

# Import ESPN scores integration
from espn_scores import (
    fetch_espn_scores, 
    find_matching_game, 
    determine_bet_result,
    get_live_games,
    get_final_games
)

# Import custom betting algorithm (V2 - legacy)
from betting_algorithm import calculate_pick

# Import ENHANCED betting algorithm V3
from enhanced_betting_algorithm import calculate_enhanced_pick, EnhancedBettingAlgorithm

# Import multi-bookmaker odds provider
from multi_book_odds import fetch_multi_book_odds, get_multi_book_provider

# Import odds aggregator (OddsPortal + ESPN)
from odds_aggregator import fetch_aggregated_odds, get_odds_aggregator

# Import SMART prediction engine V4 (no LLM required)
from smart_prediction_engine import generate_smart_prediction, SmartPredictionEngine

# Import lineup/roster scraper
from lineup_scraper import get_matchup_context, fetch_team_roster

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
mock_events_cache = {}  # Cache for mock events
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
    ]

@api_router.get("/events/{sport_key}")
async def get_events(sport_key: str, markets: str = "h2h,spreads,totals", force_refresh: bool = False, pre_match_only: bool = True):
    """Get events with REAL odds from ESPN API
    
    Args:
        sport_key: Sport identifier
        markets: Comma-separated market types
        force_refresh: Force refresh from ESPN
        pre_match_only: Only return events that haven't started yet (default True)
    """
    global events_cache, last_scrape_time
    
    cache_key = f"{sport_key}_{markets}"
    now = datetime.now(timezone.utc)
    
    # Check cache first (unless force refresh)
    if not force_refresh and cache_key in events_cache:
        cached_data, cached_time = events_cache[cache_key]
        cache_age_minutes = (now - cached_time).total_seconds() / 60
        if cache_age_minutes < CACHE_DURATION_MINUTES:
            logger.info(f"Using cached data for {sport_key} (age: {cache_age_minutes:.1f} min)")
            events = cached_data
            # Filter to pre-match only if requested
            if pre_match_only:
                events = filter_prematch_events(events, now)
            return events
    
    # Fetch from ESPN API (REAL ODDS from DraftKings)
    try:
        events = await fetch_espn_events_with_odds(sport_key, days_ahead=3)
        
        if events:
            # Store odds snapshots for line movement tracking (only for pre-match events)
            for event in events:
                try:
                    commence_str = event.get("commence_time", "")
                    if commence_str:
                        commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                        # Only store odds for events that haven't started
                        if commence_time > now:
                            await store_odds_snapshot(event)
                except Exception:
                    pass
            
            # Cache the results
            events_cache[cache_key] = (events, now)
            last_scrape_time = now.isoformat()
            logger.info(f"Cached {len(events)} events from ESPN for {sport_key}")
            
            # Filter to pre-match only if requested
            if pre_match_only:
                events = filter_prematch_events(events, now)
            return events
        else:
            logger.warning(f"ESPN returned no events for {sport_key}")
            # Return cached if available
            if cache_key in events_cache:
                events = events_cache[cache_key][0]
                if pre_match_only:
                    events = filter_prematch_events(events, now)
                return events
            return []
            
    except Exception as e:
        logger.error(f"ESPN API fetch failed: {e}")
        # Return cached if available
        if cache_key in events_cache:
            events = events_cache[cache_key][0]
            if pre_match_only:
                events = filter_prematch_events(events, now)
            return events
        return []


def filter_prematch_events(events: List[dict], now: datetime) -> List[dict]:
    """Filter events to only include pre-match (not started yet)"""
    prematch = []
    for event in events:
        try:
            commence_str = event.get("commence_time", "")
            if commence_str:
                commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                if commence_time > now:
                    prematch.append(event)
        except Exception:
            # If we can't parse time, include the event
            prematch.append(event)
    return prematch

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
    
    # Get event info first
    event_info = None
    current_odds = None
    events = await get_events(sport_key)
    for event in events:
        if event.get("id") == event_id:
            event_info = {
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "commence_time": event.get("commence_time")
            }
            # Get current odds from bookmakers
            bookmakers = event.get("bookmakers", [])
            if bookmakers:
                home_prices = []
                away_prices = []
                for bm in bookmakers:
                    for market in bm.get("markets", []):
                        if market.get("key") == "h2h":
                            outcomes = market.get("outcomes", [])
                            if len(outcomes) >= 2:
                                home_prices.append(outcomes[0].get("price", 0))
                                away_prices.append(outcomes[1].get("price", 0))
                if home_prices and away_prices:
                    current_odds = {
                        "home": round(sum(home_prices) / len(home_prices), 2),
                        "away": round(sum(away_prices) / len(away_prices), 2)
                    }
            break
    
    # Build chart data from snapshots
    chart_data = []
    seen_hours = set()
    
    for snap in snapshots:
        ts = snap.get("timestamp")
        hour_key = snap.get("hour_key", ts[:13] if ts else None)  # YYYY-MM-DD-HH
        
        if hour_key and hour_key not in seen_hours:
            seen_hours.add(hour_key)
            home = snap.get("home_odds")
            away = snap.get("away_odds")
            
            if home and away:
                chart_data.append({
                    "timestamp": ts,
                    "home_odds": round(home, 2),
                    "away_odds": round(away, 2),
                    "num_bookmakers": snap.get("num_bookmakers", 1)
                })
    
    # If we have opening odds but no chart data, create initial point
    if opening and not chart_data:
        chart_data.append({
            "timestamp": opening.get("timestamp"),
            "home_odds": opening.get("home_odds"),
            "away_odds": opening.get("away_odds"),
            "num_bookmakers": len(opening.get("bookmakers", []))
        })
    
    # Add current odds as latest point if different from last snapshot
    if current_odds and chart_data:
        last = chart_data[-1]
        if abs(last["home_odds"] - current_odds["home"]) > 0.01 or abs(last["away_odds"] - current_odds["away"]) > 0.01:
            chart_data.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "home_odds": current_odds["home"],
                "away_odds": current_odds["away"],
                "num_bookmakers": 0  # Current snapshot
            })
    elif current_odds and not chart_data:
        chart_data.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "home_odds": current_odds["home"],
            "away_odds": current_odds["away"],
            "num_bookmakers": 0
        })
    
    # Group by bookmaker for detailed view
    by_bookmaker = {}
    for snap in snapshots:
        for bm_snap in snap.get("bookmakers", []):
            bm_key = bm_snap.get("bookmaker", "unknown")
            if bm_key not in by_bookmaker:
                by_bookmaker[bm_key] = {
                    "bookmaker": bm_key,
                    "bookmaker_title": bm_snap.get("bookmaker_title", bm_key),
                    "snapshots": []
                }
            by_bookmaker[bm_key]["snapshots"].append({
                "timestamp": snap.get("timestamp"),
                "home_odds": bm_snap.get("home_odds"),
                "away_odds": bm_snap.get("away_odds")
            })
    
    # Sort chart data by timestamp
    chart_data.sort(key=lambda x: x["timestamp"] if x.get("timestamp") else "")
    
    return {
        "event_id": event_id,
        "event_info": event_info,
        "opening_odds": opening,
        "current_odds": current_odds,
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
async def get_recommendations(
    sport_key: Optional[str] = None, 
    limit: int = 10, 
    min_odds: float = 1.5,
    min_confidence: float = 0.70,
    include_all: bool = False
):
    """Get AI-generated bet recommendations - filtered by 70%+ confidence and time window"""
    now = datetime.now(timezone.utc)
    
    # Calculate time window: later today through 3 days from now
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    three_days_later = today_start + timedelta(days=4)  # End of day 3
    
    # Base query - pending predictions with valid odds
    query = {
        "result": "pending",
        "odds_at_prediction": {"$gte": min_odds, "$lte": 20}
    }
    
    # Apply 70%+ confidence filter unless include_all is True
    if not include_all:
        query["confidence"] = {"$gte": min_confidence}
    
    if sport_key:
        query["sport_key"] = sport_key
    
    # Get predictions sorted by confidence
    predictions = await db.predictions.find(
        query,
        {"_id": 0}
    ).sort("confidence", -1).limit(limit * 3).to_list(limit * 3)  # Get extra to filter by time
    
    # Filter by time window: only events starting later today through 3 days
    filtered_predictions = []
    for pred in predictions:
        try:
            commence_time_str = pred.get("commence_time", "")
            if commence_time_str:
                commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
                
                # Must be in the future (not already started) and within 3 days
                if now < commence_time <= three_days_later:
                    filtered_predictions.append(pred)
        except Exception:
            # If we can't parse time, skip this prediction
            continue
    
    return filtered_predictions[:limit]

@api_router.post("/recommendations")
async def create_recommendation(prediction: PredictionCreate):
    """Create a new bet recommendation"""
    pred_dict = prediction.model_dump()
    pred_obj = Prediction(**pred_dict)
    pred_obj.result = "pending"
    
    await db.predictions.insert_one(pred_obj.model_dump())
    return pred_obj

@api_router.post("/generate-recommendations")
async def generate_recommendations(sport_key: str, background_tasks: BackgroundTasks):
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

@api_router.post("/force-generate-picks")
async def force_generate_picks(background_tasks: BackgroundTasks):
    """Force immediate generation of picks using custom algorithm"""
    background_tasks.add_task(auto_generate_recommendations)
    return {"message": "Pick generation started in background using custom algorithm - picks will appear shortly"}

# Auto-generate recommendations using CUSTOM ALGORITHM (no AI)
async def auto_generate_recommendations():
    """Automatically generate picks using custom betting algorithm - only 70%+ confidence within 3 day window"""
    logger.info("Starting ALGORITHMIC pick generation (no AI)...")
    sports_to_analyze = ["basketball_nba", "americanfootball_nfl", "icehockey_nhl", "soccer_epl"]
    
    now = datetime.now(timezone.utc)
    three_days_later = now + timedelta(days=3)
    picks_generated = 0
    
    for sport_key in sports_to_analyze:
        try:
            # Fetch events with REAL odds from ESPN
            events = await fetch_espn_events_with_odds(sport_key, days_ahead=3)
            if not events:
                logger.info(f"No events found for {sport_key}")
                continue
            
            logger.info(f"Found {len(events)} events for {sport_key}")
            
            # Filter events within time window (later today to 3 days from now)
            valid_events = []
            for event in events:
                try:
                    commence_str = event.get("commence_time", "")
                    if commence_str:
                        commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                        # Must start in the future and within 3 days
                        if now < commence_time <= three_days_later:
                            valid_events.append(event)
                except Exception:
                    continue
            
            logger.info(f"Found {len(valid_events)} valid pre-match events for {sport_key}")
            
            # Analyze valid events using custom algorithm
            for event in valid_events[:5]:  # Top 5 per sport
                event_id = event.get("id")
                
                # Check if we already have a recommendation for this event
                existing = await db.predictions.find_one({"event_id": event_id, "result": "pending"})
                if existing:
                    logger.debug(f"Skipping {event_id} - already has prediction")
                    continue
                
                # Get comprehensive matchup data for algorithm
                try:
                    logger.info(f"Analyzing: {event.get('home_team')} vs {event.get('away_team')}")
                    matchup_data = await get_comprehensive_matchup_data(event, sport_key)
                    
                    # Get line movement data from our database
                    line_movement = await get_line_movement_data(event_id)
                    
                    # Run custom betting algorithm
                    pick_result = calculate_pick(matchup_data, line_movement)
                    
                    if pick_result:
                        logger.info(f"Algorithm result: {pick_result.get('pick')} conf={pick_result.get('confidence', 0)}")
                        
                        if pick_result.get("confidence", 0) >= 0.70:
                            # Create prediction from algorithm result
                            prediction = PredictionCreate(
                                event_id=event_id,
                                sport_key=sport_key,
                                home_team=event.get("home_team"),
                                away_team=event.get("away_team"),
                                commence_time=event.get("commence_time"),
                                prediction_type=pick_result.get("pick_type", "moneyline"),
                                predicted_outcome=pick_result.get("pick", ""),
                                confidence=pick_result.get("confidence", 0.70),
                                analysis=pick_result.get("reasoning", ""),
                                ai_model="custom_algorithm_v1",
                                odds_at_prediction=pick_result.get("odds", 1.91)
                            )
                            
                            await create_recommendation(prediction)
                            picks_generated += 1
                            logger.info(f"ALGORITHM PICK: {event.get('home_team')} vs {event.get('away_team')} - "
                                      f"{pick_result.get('pick')} @ {pick_result.get('confidence')*100:.0f}% conf, "
                                      f"{pick_result.get('edge', 0):.1f}% edge")
                        else:
                            logger.info(f"Confidence too low: {pick_result.get('confidence', 0)*100:.0f}%")
                    else:
                        logger.info(f"No pick returned for {event.get('home_team')} vs {event.get('away_team')}")
                        
                except Exception as e:
                    logger.error(f"Error analyzing {event_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
        except Exception as e:
            logger.error(f"Error generating recommendations for {sport_key}: {e}")
    
    logger.info(f"Algorithmic pick generation complete - {picks_generated} picks created")


async def get_line_movement_data(event_id: str) -> Dict:
    """Get line movement data for an event from our database"""
    try:
        opening = await db.opening_odds.find_one({"event_id": event_id}, {"_id": 0})
        if opening:
            return {
                "opening_home_odds": opening.get("home_odds"),
                "opening_away_odds": opening.get("away_odds"),
                "opening_time": opening.get("timestamp")
            }
    except Exception as e:
        logger.error(f"Error getting line movement for {event_id}: {e}")
    return {}


# Keep the old function for backward compatibility but it's not used
async def generate_smart_recommendation(event: dict, sport_key: str, odds_data: dict) -> Optional[dict]:
    """DEPRECATED - Use auto_generate_recommendations with custom algorithm instead"""
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
    """Background task to check event results using ESPN API and update predictions"""
    logger.info("Starting automatic result checking with ESPN API...")
    
    try:
        # Get all pending predictions
        pending = await db.predictions.find(
            {"result": "pending"},
            {"_id": 0}
        ).to_list(200)
        
        if not pending:
            logger.info("No pending predictions to check")
            return
        
        logger.info(f"Checking {len(pending)} pending predictions...")
        
        # Group predictions by sport for efficient API calls
        by_sport = {}
        for pred in pending:
            sport_key = pred.get("sport_key", "")
            if sport_key not in by_sport:
                by_sport[sport_key] = []
            by_sport[sport_key].append(pred)
        
        # Fetch scores for each sport and process predictions
        results_updated = 0
        for sport_key, predictions in by_sport.items():
            try:
                # Fetch recent game scores from ESPN
                games = await fetch_espn_scores(sport_key, days_back=5)
                
                if not games:
                    logger.warning(f"No ESPN scores available for {sport_key}")
                    continue
                
                logger.info(f"Fetched {len(games)} games from ESPN for {sport_key}")
                
                for prediction in predictions:
                    try:
                        commence_time_str = prediction.get("commence_time")
                        if not commence_time_str:
                            continue
                        
                        commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
                        now = datetime.now(timezone.utc)
                        
                        # Only check if at least 2 hours have passed since event start (game should be over)
                        hours_since_start = (now - commence_time).total_seconds() / 3600
                        
                        if hours_since_start < 2:
                            continue  # Game probably not finished yet
                        
                        # Find matching game in ESPN results
                        matching_game = await find_matching_game(
                            sport_key,
                            prediction.get("home_team"),
                            prediction.get("away_team"),
                            commence_time_str,
                            games
                        )
                        
                        if matching_game and matching_game.get("status") == "final":
                            # Determine bet result
                            result = determine_bet_result(
                                prediction.get("prediction_type", "moneyline"),
                                prediction.get("predicted_outcome", ""),
                                matching_game.get("home_team"),
                                matching_game.get("away_team"),
                                matching_game.get("home_score", 0),
                                matching_game.get("away_score", 0),
                                matching_game.get("total_score", 0)
                            )
                            
                            if result in ["win", "loss", "push"]:
                                # Update prediction with result
                                await db.predictions.update_one(
                                    {"id": prediction.get("id")},
                                    {"$set": {
                                        "result": result,
                                        "result_updated_at": now.isoformat(),
                                        "final_score": {
                                            "home": matching_game.get("home_score"),
                                            "away": matching_game.get("away_score"),
                                            "total": matching_game.get("total_score")
                                        }
                                    }}
                                )
                                
                                results_updated += 1
                                logger.info(f"Updated prediction {prediction.get('id')}: {result} "
                                          f"({matching_game.get('home_team')} {matching_game.get('home_score')} - "
                                          f"{matching_game.get('away_score')} {matching_game.get('away_team')})")
                                
                                # Create notification for result
                                await create_notification(
                                    "result",
                                    f"Bet Result: {result.upper()}",
                                    f"{prediction.get('home_team')} vs {prediction.get('away_team')} - "
                                    f"Your pick: {prediction.get('predicted_outcome')} "
                                    f"(Final: {matching_game.get('home_score')}-{matching_game.get('away_score')})",
                                    {
                                        "prediction_id": prediction.get("id"),
                                        "result": result,
                                        "final_score": {
                                            "home": matching_game.get("home_score"),
                                            "away": matching_game.get("away_score")
                                        }
                                    }
                                )
                        
                    except Exception as e:
                        logger.error(f"Error processing prediction {prediction.get('id')}: {e}")
                
                await asyncio.sleep(0.5)  # Rate limiting between sports
                
            except Exception as e:
                logger.error(f"Error fetching ESPN scores for {sport_key}: {e}")
        
        logger.info(f"Result checking complete. Updated {results_updated} predictions.")
                
    except Exception as e:
        logger.error(f"Error in auto_check_results: {e}")

async def fetch_event_result(sport_key: str, event_id: str, home_team: str, away_team: str, predicted_outcome: str, prediction_type: str) -> Optional[str]:
    """Fetch event result using ESPN API"""
    try:
        # Find matching game
        game = await find_matching_game(sport_key, home_team, away_team, None)
        
        if game and game.get("status") == "final":
            return determine_bet_result(
                prediction_type,
                predicted_outcome,
                game.get("home_team"),
                game.get("away_team"),
                game.get("home_score", 0),
                game.get("away_score", 0),
                game.get("total_score", 0)
            )
    except Exception as e:
        logger.error(f"Error fetching event result: {e}")
    
    return None

@api_router.get("/scores/{sport_key}")
async def get_scores(sport_key: str, days_back: int = 3, status: Optional[str] = None):
    """Get live and recent scores from ESPN API"""
    try:
        games = await fetch_espn_scores(sport_key, days_back=days_back)
        
        # Filter by status if specified
        if status:
            if status == "live":
                games = [g for g in games if g.get("status") == "in_progress"]
            elif status == "final":
                games = [g for g in games if g.get("status") == "final"]
            elif status == "scheduled":
                games = [g for g in games if g.get("status") == "scheduled"]
        
        return {
            "sport_key": sport_key,
            "games_count": len(games),
            "games": games
        }
    except Exception as e:
        logger.error(f"Error fetching scores for {sport_key}: {e}")
        return {
            "sport_key": sport_key,
            "games_count": 0,
            "games": [],
            "error": str(e)
        }

@api_router.get("/live-scores")
async def get_all_live_scores():
    """Get all currently in-progress games across all sports"""
    sports = ["basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl", "soccer_epl"]
    all_live = []
    
    for sport_key in sports:
        try:
            games = await get_live_games(sport_key)
            for game in games:
                game["sport_key"] = sport_key
            all_live.extend(games)
        except Exception as e:
            logger.error(f"Error fetching live scores for {sport_key}: {e}")
    
    return {
        "live_games_count": len(all_live),
        "games": all_live
    }

@api_router.get("/pending-results")
async def get_pending_results():
    """Get pending predictions that are waiting for results"""
    pending = await db.predictions.find(
        {"result": "pending"},
        {"_id": 0}
    ).sort("commence_time", 1).to_list(100)
    
    now = datetime.now(timezone.utc)
    
    # Categorize by status
    awaiting_start = []
    in_progress = []
    awaiting_result = []
    
    for pred in pending:
        try:
            commence_time_str = pred.get("commence_time", "")
            if commence_time_str:
                commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
                hours_since_start = (now - commence_time).total_seconds() / 3600
                
                if hours_since_start < 0:
                    awaiting_start.append(pred)
                elif hours_since_start < 3:
                    in_progress.append(pred)
                else:
                    awaiting_result.append(pred)
            else:
                awaiting_start.append(pred)
        except Exception:
            awaiting_start.append(pred)
    
    return {
        "total_pending": len(pending),
        "awaiting_start": awaiting_start,
        "in_progress": in_progress,
        "awaiting_result": awaiting_result
    }

@api_router.post("/cleanup-line-movement")
async def cleanup_line_movement_data():
    """Clean up line movement data for events that have already started (live or finished)"""
    now = datetime.now(timezone.utc)
    deleted_count = 0
    
    # Get all events with line movement data
    event_ids = await db.odds_history.distinct("event_id")
    
    for event_id in event_ids:
        # Get opening odds to check commence time
        opening = await db.opening_odds.find_one({"event_id": event_id})
        
        if opening:
            # Check commence time from any prediction for this event
            prediction = await db.predictions.find_one({"event_id": event_id}, {"commence_time": 1})
            
            if prediction:
                commence_str = prediction.get("commence_time", "")
                if commence_str:
                    try:
                        commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                        
                        # If event has started, delete line movement data
                        if commence_time <= now:
                            # Delete odds history for this event
                            result = await db.odds_history.delete_many({"event_id": event_id})
                            deleted_count += result.deleted_count
                            
                            # Delete opening odds (optional - keep for historical reference)
                            # await db.opening_odds.delete_one({"event_id": event_id})
                            
                            logger.info(f"Cleaned up line movement for event {event_id} (started)")
                    except Exception:
                        pass
    
    return {
        "message": f"Cleaned up line movement data for {deleted_count} records",
        "deleted_count": deleted_count
    }

# NEW: Endpoint to manually trigger Smart V4 prediction for a specific event
@api_router.post("/analyze-pregame/{event_id}")
async def analyze_pregame(event_id: str, sport_key: str = "basketball_nba"):
    """
    Manually trigger Smart V4 analysis for a specific event.
    Uses comprehensive statistical analysis - NO LLM REQUIRED.
    Returns diverse predictions (ML/Spread/Total) based on:
    - Squad data & player stats
    - Recent form & margins
    - Line movement
    - Multi-book odds comparison
    """
    try:
        # Fetch event
        events = await fetch_espn_events_with_odds(sport_key, days_ahead=3)
        event = None
        for e in events:
            if e.get("id") == event_id:
                event = e
                break
        
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Check if game has already started
        commence_str = event.get("commence_time", "")
        if commence_str:
            commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
            if commence_time <= datetime.now(timezone.utc):
                raise HTTPException(status_code=400, detail="Event has already started")
        
        logger.info(f"📊 Manual Smart Analysis: {event.get('home_team')} vs {event.get('away_team')}")
        
        # 1. Get comprehensive matchup data
        matchup_data = await get_comprehensive_matchup_data(event, sport_key)
        
        # 2. Get full squad data for both teams
        squad_data = await get_matchup_context(
            event.get("home_team"), 
            event.get("away_team"), 
            sport_key
        )
        
        # 3. Get line movement history
        line_history = await get_line_movement_history(event_id)
        
        # 4. Get multi-bookmaker odds (ESPN + aggregated)
        multi_book_odds = await fetch_aggregated_odds(sport_key, event_id, event)
        
        # 5. Run SMART prediction engine
        smart_prediction = await generate_smart_prediction(
            event=event,
            sport_key=sport_key,
            squad_data=squad_data,
            matchup_data=matchup_data,
            line_movement=line_history,
            multi_book_odds=multi_book_odds
        )
        
        if smart_prediction and smart_prediction.get("has_pick") and smart_prediction.get("confidence", 0) >= 0.70:
            # Create prediction
            prediction = PredictionCreate(
                event_id=event_id,
                sport_key=sport_key,
                home_team=event.get("home_team"),
                away_team=event.get("away_team"),
                commence_time=event.get("commence_time"),
                prediction_type=smart_prediction.get("pick_type", "moneyline"),
                predicted_outcome=smart_prediction.get("pick", ""),
                confidence=smart_prediction.get("confidence", 0.70),
                analysis=smart_prediction.get("reasoning", ""),
                ai_model="smart_v4",
                odds_at_prediction=smart_prediction.get("odds", 1.91)
            )
            
            await create_recommendation(prediction)
            
            return {
                "status": "prediction_created",
                "algorithm": "smart_v4",
                "event": f"{event.get('home_team')} vs {event.get('away_team')}",
                "prediction": smart_prediction,
                "data_sources": {
                    "matchup_data": bool(matchup_data),
                    "squad_data": bool(squad_data),
                    "line_movement_snapshots": len(line_history),
                    "multi_book_sources": multi_book_odds.get("sources", []) if multi_book_odds else []
                }
            }
        else:
            return {
                "status": "no_pick",
                "algorithm": "smart_v4",
                "event": f"{event.get('home_team')} vs {event.get('away_team')}",
                "reason": smart_prediction.get("reasoning", "No value found") if smart_prediction else "Analysis failed",
                "closest_value": smart_prediction.get("closest_value") if smart_prediction else None,
                "data_sources": {
                    "matchup_data": bool(matchup_data),
                    "squad_data": bool(squad_data),
                    "line_movement_snapshots": len(line_history),
                    "multi_book_sources": multi_book_odds.get("sources", []) if multi_book_odds else []
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in manual smart pregame analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Get V3 predictions specifically
@api_router.get("/predictions/v3")
async def get_v3_predictions(limit: int = 50, result: str = None):
    """Get predictions made by Enhanced V3 algorithm"""
    query = {"ai_model": "enhanced_v3"}
    if result:
        query["result"] = result
    
    predictions = await db.predictions.find(
        query, 
        {"_id": 0}
    ).sort("created_at", -1).limit(limit).to_list(limit)
    
    # Calculate V3 stats
    all_v3 = await db.predictions.find({"ai_model": "enhanced_v3"}).to_list(10000)
    
    wins = len([p for p in all_v3 if p.get("result") == "win"])
    losses = len([p for p in all_v3 if p.get("result") == "loss"])
    pending = len([p for p in all_v3 if p.get("result") == "pending"])
    
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    
    return {
        "predictions": predictions,
        "stats": {
            "total": len(all_v3),
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": round(win_rate, 1)
        },
        "algorithm": "enhanced_v3",
        "description": "Predictions made 1-2 hours before game start with deep analysis"
    }

# NEW: Get Smart V4 predictions specifically
@api_router.get("/predictions/smart-v4")
async def get_smart_v4_predictions(limit: int = 50, result: str = None):
    """Get predictions made by Smart V4 algorithm (no LLM required)"""
    query = {"ai_model": "smart_v4"}
    if result:
        query["result"] = result
    
    predictions = await db.predictions.find(
        query, 
        {"_id": 0}
    ).sort("created_at", -1).limit(limit).to_list(limit)
    
    # Calculate V4 stats
    all_v4 = await db.predictions.find({"ai_model": "smart_v4"}).to_list(10000)
    
    wins = len([p for p in all_v4 if p.get("result") == "win"])
    losses = len([p for p in all_v4 if p.get("result") == "loss"])
    pending = len([p for p in all_v4 if p.get("result") == "pending"])
    
    # Count by pick type
    ml_picks = len([p for p in all_v4 if p.get("prediction_type") == "moneyline"])
    spread_picks = len([p for p in all_v4 if p.get("prediction_type") == "spread"])
    total_picks = len([p for p in all_v4 if p.get("prediction_type") == "total"])
    
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    
    return {
        "predictions": predictions,
        "stats": {
            "total": len(all_v4),
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": round(win_rate, 1),
            "pick_types": {
                "moneyline": ml_picks,
                "spread": spread_picks,
                "total": total_picks
            }
        },
        "algorithm": "smart_v4",
        "description": "Smart predictions 1 hour before game - analyzes squads, form, line movement. NO LLM required."
    }

# NEW: Compare V2 vs V3 vs Smart V4 algorithm performance
@api_router.get("/predictions/comparison")
async def compare_algorithms():
    """Compare performance of V2 (legacy) vs V3 (enhanced) vs Smart V4 algorithms"""
    
    # Get all predictions
    all_predictions = await db.predictions.find({}).to_list(10000)
    
    v2_predictions = [p for p in all_predictions if p.get("ai_model") in ["custom_algorithm_v1", "gpt-5.2", "claude"]]
    v3_predictions = [p for p in all_predictions if p.get("ai_model") == "enhanced_v3"]
    v4_predictions = [p for p in all_predictions if p.get("ai_model") == "smart_v4"]
    
    def calculate_stats(predictions):
        completed = [p for p in predictions if p.get("result") in ["win", "loss"]]
        wins = len([p for p in completed if p.get("result") == "win"])
        losses = len([p for p in completed if p.get("result") == "loss"])
        pending = len([p for p in predictions if p.get("result") == "pending"])
        
        # Count by pick type
        ml_picks = len([p for p in predictions if p.get("prediction_type") == "moneyline"])
        spread_picks = len([p for p in predictions if p.get("prediction_type") == "spread"])
        total_picks = len([p for p in predictions if p.get("prediction_type") == "total"])
        
        win_rate = wins / len(completed) * 100 if completed else 0
        avg_confidence = sum(p.get("confidence", 0) for p in predictions) / len(predictions) * 100 if predictions else 0
        
        return {
            "total": len(predictions),
            "completed": len(completed),
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": round(win_rate, 1),
            "avg_confidence": round(avg_confidence, 1),
            "pick_types": {
                "moneyline": ml_picks,
                "spread": spread_picks,
                "total": total_picks
            }
        }
    
    return {
        "v2_legacy": calculate_stats(v2_predictions),
        "v3_enhanced": calculate_stats(v3_predictions),
        "smart_v4": calculate_stats(v4_predictions),
        "description": {
            "v2_legacy": "Original algorithm with basic factors",
            "v3_enhanced": "Enhanced factors without LLM",
            "smart_v4": "Smart algorithm - diverse predictions (ML/Spread/Total), no LLM required"
        }
    }

# NEW: View upcoming games in prediction window
@api_router.get("/upcoming-predictions-window")
async def get_upcoming_prediction_window():
    """View games that are in the 1-2 hour prediction window"""
    now = datetime.now(timezone.utc)
    window_start = now + timedelta(hours=1)
    window_end = now + timedelta(hours=2)
    
    sports = ["basketball_nba", "americanfootball_nfl", "icehockey_nhl", "soccer_epl"]
    games_in_window = []
    upcoming_games = []
    
    for sport_key in sports:
        try:
            events = await fetch_espn_events_with_odds(sport_key, days_ahead=1)
            
            for event in events:
                commence_str = event.get("commence_time", "")
                if not commence_str:
                    continue
                
                try:
                    commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                    time_to_start = (commence_time - now).total_seconds() / 60  # in minutes
                    
                    # Check if already has V3 prediction
                    has_v3_prediction = await db.predictions.find_one({
                        "event_id": event.get("id"),
                        "ai_model": "enhanced_v3"
                    }) is not None
                    
                    game_info = {
                        "event_id": event.get("id"),
                        "sport": sport_key,
                        "home_team": event.get("home_team"),
                        "away_team": event.get("away_team"),
                        "commence_time": commence_str,
                        "minutes_to_start": round(time_to_start),
                        "has_v3_prediction": has_v3_prediction
                    }
                    
                    if window_start <= commence_time <= window_end:
                        game_info["status"] = "IN_PREDICTION_WINDOW"
                        games_in_window.append(game_info)
                    elif now < commence_time < window_start:
                        game_info["status"] = "UPCOMING"
                        upcoming_games.append(game_info)
                    
                except Exception:
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching events for {sport_key}: {e}")
    
    return {
        "current_time": now.isoformat(),
        "prediction_window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat()
        },
        "games_in_window": sorted(games_in_window, key=lambda x: x["minutes_to_start"]),
        "upcoming_games": sorted(upcoming_games, key=lambda x: x["minutes_to_start"])[:10],
        "total_in_window": len(games_in_window),
        "message": "Games in prediction window will be automatically analyzed by V3 algorithm"
    }

@api_router.get("/odds-comparison/{event_id}")
async def get_odds_comparison(event_id: str, sport_key: str = "basketball_nba"):
    """Get odds comparison for an event - uses ESPN data in European/decimal format"""
    # Try multiple sports if not found
    sports_to_try = [sport_key, "basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl", "soccer_epl"]
    
    for sk in sports_to_try:
        events = await get_events(sk)
        for event in events:
            if event.get("id") == event_id:
                # Format ESPN odds for comparison view
                comparison = format_espn_odds_comparison(event)
                return comparison
    
    raise HTTPException(status_code=404, detail="Event not found")


def format_espn_odds_comparison(event: dict) -> dict:
    """Format ESPN odds for comparison view - all in European/decimal format"""
    odds = event.get("odds", {})
    home_team = event.get("home_team", "Home")
    away_team = event.get("away_team", "Away")
    
    # Get odds from ESPN (already in decimal format)
    home_ml = odds.get("home_ml_decimal", 1.91)
    away_ml = odds.get("away_ml_decimal", 1.91)
    spread = odds.get("spread", 0)
    total = odds.get("total", 220)
    provider = odds.get("provider_name", "DraftKings")
    
    comparison = {
        "event_id": event.get("id"),
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": event.get("commence_time"),
        "venue": event.get("venue", {}),
        "source": "ESPN/DraftKings",
        "h2h": [
            {
                "bookmaker": "draftkings",
                "title": provider,
                "last_update": event.get("scraped_at"),
                "outcomes": [
                    {"name": home_team, "price": round(home_ml, 2)},
                    {"name": away_team, "price": round(away_ml, 2)}
                ]
            }
        ],
        "spreads": [
            {
                "bookmaker": "draftkings",
                "title": provider,
                "last_update": event.get("scraped_at"),
                "outcomes": [
                    {"name": home_team, "price": 1.91, "point": spread},
                    {"name": away_team, "price": 1.91, "point": -spread if spread else 0}
                ]
            }
        ],
        "totals": [
            {
                "bookmaker": "draftkings",
                "title": provider,
                "last_update": event.get("scraped_at"),
                "outcomes": [
                    {"name": "Over", "price": 1.91, "point": total},
                    {"name": "Under", "price": 1.91, "point": total}
                ]
            }
        ],
        "best_odds": {
            "home_ml": round(home_ml, 2),
            "away_ml": round(away_ml, 2),
            "spread": spread,
            "total": total,
            "home_favorite": odds.get("home_favorite", spread < 0 if spread else False)
        }
    }
    
    return comparison

# Helper functions
async def store_odds_snapshot(event: dict):
    """Store odds snapshot for line movement tracking - stores opening odds and hourly snapshots"""
    event_id = event.get("id")
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    # Round to current hour for grouping snapshots
    hour_key = now.strftime("%Y-%m-%d-%H")
    
    home_team = event.get("home_team", "home")
    away_team = event.get("away_team", "away")
    
    # Check if we have opening odds stored for this event
    existing_opening = await db.opening_odds.find_one({"event_id": event_id})
    
    # Check if we already have a snapshot for this hour (to avoid duplicates)
    existing_snapshot = await db.odds_history.find_one({
        "event_id": event_id,
        "hour_key": hour_key
    })
    
    # Collect all bookmaker odds for this snapshot
    all_home_odds = []
    all_away_odds = []
    bookmaker_snapshots = []
    
    for bookmaker in event.get("bookmakers", []):
        bm_key = bookmaker.get("key")
        bm_title = bookmaker.get("title", bm_key)
        
        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "h2h")
            if market_key != "h2h":
                continue  # Only track moneyline for line movement
                
            outcomes = market.get("outcomes", [])
            
            # Extract home and away odds
            home_odds = None
            away_odds = None
            
            for outcome in outcomes:
                name = outcome.get("name", "").lower()
                price = outcome.get("price")
                
                if name == "home" or home_team.lower() in name:
                    home_odds = price
                elif name == "away" or away_team.lower() in name:
                    away_odds = price
            
            # If not found by name, use position (first = home, second = away)
            if home_odds is None and len(outcomes) >= 1:
                home_odds = outcomes[0].get("price")
            if away_odds is None and len(outcomes) >= 2:
                away_odds = outcomes[1].get("price")
            
            if home_odds and away_odds:
                all_home_odds.append(home_odds)
                all_away_odds.append(away_odds)
                bookmaker_snapshots.append({
                    "bookmaker": bm_key,
                    "bookmaker_title": bm_title,
                    "home_odds": home_odds,
                    "away_odds": away_odds
                })
    
    # Calculate average odds across all bookmakers
    avg_home = sum(all_home_odds) / len(all_home_odds) if all_home_odds else None
    avg_away = sum(all_away_odds) / len(all_away_odds) if all_away_odds else None
    
    # Store opening odds if this is first time seeing this event
    if not existing_opening and avg_home and avg_away:
        await db.opening_odds.insert_one({
            "event_id": event_id,
            "home_team": home_team,
            "away_team": away_team,
            "home_odds": round(avg_home, 2),
            "away_odds": round(avg_away, 2),
            "bookmakers": bookmaker_snapshots,
            "timestamp": timestamp
        })
        logger.info(f"Stored opening odds for {home_team} vs {away_team}: {avg_home:.2f}/{avg_away:.2f}")
    
    # Store hourly snapshot for line movement tracking (avoid duplicates within same hour)
    if not existing_snapshot and avg_home and avg_away:
        snapshot = {
            "event_id": event_id,
            "home_team": home_team,
            "away_team": away_team,
            "timestamp": timestamp,
            "hour_key": hour_key,
            "home_odds": round(avg_home, 2),
            "away_odds": round(avg_away, 2),
            "bookmakers": bookmaker_snapshots,
            "num_bookmakers": len(bookmaker_snapshots)
        }
        await db.odds_history.insert_one(snapshot)
        logger.info(f"Stored hourly snapshot for {home_team} vs {away_team}: H={avg_home:.2f} A={avg_away:.2f}")

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

@api_router.delete("/clear-all-data")
async def clear_all_data():
    """Clear all predictions, odds history, and opening odds - FRESH START"""
    try:
        pred_result = await db.predictions.delete_many({})
        odds_result = await db.odds_history.delete_many({})
        opening_result = await db.opening_odds.delete_many({})
        
        return {
            "message": "All data cleared - fresh start!",
            "deleted": {
                "predictions": pred_result.deleted_count,
                "odds_history": odds_result.deleted_count,
                "opening_odds": opening_result.deleted_count
            }
        }
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

# Background task scheduler for INSTANT live score sync - runs every 10 SECONDS for live games
async def scheduled_result_checker():
    """Background task that runs every 10 SECONDS for instant live score sync using ESPN API"""
    # Initial delay to let services start
    await asyncio.sleep(30)
    
    while True:
        try:
            # Check if there are any live games or pending predictions awaiting results
            pending_count = await db.predictions.count_documents({"result": "pending"})
            
            if pending_count > 0:
                logger.info(f"Running instant live score sync ({pending_count} pending predictions)...")
                await auto_check_results()
            
            await asyncio.sleep(10)  # Run every 10 SECONDS for instant updates
        except Exception as e:
            logger.error(f"Scheduled result checker error: {e}")
            await asyncio.sleep(5)  # Wait 5 seconds before retrying

# Background task for line movement checking and recommendation updates - NOW EVERY 15 MINUTES
async def scheduled_line_movement_checker():
    """Background task that runs every 15 MINUTES to check line movements"""
    while True:
        try:
            await asyncio.sleep(900)  # Run every 15 MINUTES (was 3600)
            logger.info("Running scheduled line movement check (every 15 min)...")
            await update_recommendations_on_line_movement()
        except Exception as e:
            logger.error(f"Scheduled line movement checker error: {e}")
            await asyncio.sleep(60)

# Background task for auto-generating recommendations - LEGACY (kept for backward compatibility)
async def scheduled_recommendation_generator():
    """DEPRECATED: Legacy recommendation generator. Use scheduled_pregame_predictor instead."""
    # Run immediately on startup to generate picks
    await asyncio.sleep(30)
    
    # Initial generation (legacy)
    logger.info("Running LEGACY recommendation generation on startup...")
    await auto_generate_recommendations()
    
    while True:
        try:
            # Run every 4 hours (reduced frequency since pregame predictor is primary)
            await asyncio.sleep(14400)  # Every 4 hours instead of 2
            logger.info("Running legacy recommendation generation...")
            await auto_generate_recommendations()
        except Exception as e:
            logger.error(f"Scheduled recommendation generator error: {e}")
            await asyncio.sleep(300)

# NEW: Pre-game predictor - runs predictions 1-2 hours before game start
async def scheduled_pregame_predictor():
    """
    SMART PREDICTION ENGINE V4: Generates predictions 1 hour before game start.
    Uses comprehensive statistical analysis - NO LLM REQUIRED!
    
    Features:
    - Pulls full squad data 1 hour before game
    - Analyzes player stats, H2H, venue, injuries
    - Studies line movement across multiple bookmakers (ESPN + aggregated)
    - Makes DIVERSE predictions: Moneyline, Spread, Totals
    - Only 70%+ confidence predictions
    - Considers odds as low as 1.5x
    - FREE - no API keys needed!
    """
    # Wait 1 minute on startup
    await asyncio.sleep(60)
    
    logger.info("📊 Started SMART PRE-GAME PREDICTOR V4 - analyzes games 1 hour before start, runs every 10 min")
    
    sports_to_analyze = ["basketball_nba", "americanfootball_nfl", "icehockey_nhl", "soccer_epl"]
    
    while True:
        try:
            now = datetime.now(timezone.utc)
            # Changed to 1 hour window (45 min to 75 min before game)
            window_start = now + timedelta(minutes=45)
            window_end = now + timedelta(minutes=75)
            
            predictions_made = 0
            
            for sport_key in sports_to_analyze:
                try:
                    # Fetch events
                    events = await fetch_espn_events_with_odds(sport_key, days_ahead=1)
                    
                    if not events:
                        continue
                    
                    # Filter to games starting in ~1 hour
                    for event in events:
                        try:
                            commence_str = event.get("commence_time", "")
                            if not commence_str:
                                continue
                            
                            commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                            
                            # Check if game is in our target window (~1 hour from now)
                            if window_start <= commence_time <= window_end:
                                event_id = event.get("id")
                                
                                # Check if we already have a Smart V4 prediction for this event
                                existing = await db.predictions.find_one({
                                    "event_id": event_id, 
                                    "result": "pending",
                                    "ai_model": "smart_v4"
                                })
                                
                                if existing:
                                    logger.debug(f"Skipping {event_id} - already has Smart V4 prediction")
                                    continue
                                
                                # This game is ~1 hour away - time for smart analysis!
                                logger.info(f"📊 SMART PRE-GAME ANALYSIS: {event.get('home_team')} vs {event.get('away_team')} "
                                          f"(starts in {(commence_time - now).total_seconds() / 60:.0f} min)")
                                
                                # 1. Get comprehensive matchup data
                                matchup_data = await get_comprehensive_matchup_data(event, sport_key)
                                
                                # 2. Get full squad data for both teams
                                squad_data = await get_matchup_context(
                                    event.get("home_team"), 
                                    event.get("away_team"), 
                                    sport_key
                                )
                                
                                # 3. Get line movement history from database
                                line_history = await get_line_movement_history(event_id)
                                
                                # 4. Get multi-bookmaker odds (ESPN + aggregated)
                                multi_book_odds = await fetch_aggregated_odds(sport_key, event_id, event)
                                
                                # 5. Run SMART prediction engine (no LLM required)
                                smart_prediction = await generate_smart_prediction(
                                    event=event,
                                    sport_key=sport_key,
                                    squad_data=squad_data,
                                    matchup_data=matchup_data,
                                    line_movement=line_history,
                                    multi_book_odds=multi_book_odds
                                )
                                
                                if smart_prediction and smart_prediction.get("has_pick") and smart_prediction.get("confidence", 0) >= 0.70:
                                    # Create prediction from smart algorithm result
                                    prediction = PredictionCreate(
                                        event_id=event_id,
                                        sport_key=sport_key,
                                        home_team=event.get("home_team"),
                                        away_team=event.get("away_team"),
                                        commence_time=event.get("commence_time"),
                                        prediction_type=smart_prediction.get("pick_type", "moneyline"),
                                        predicted_outcome=smart_prediction.get("pick", ""),
                                        confidence=smart_prediction.get("confidence", 0.70),
                                        analysis=smart_prediction.get("reasoning", ""),
                                        ai_model="smart_v4",
                                        odds_at_prediction=smart_prediction.get("odds", 1.91)
                                    )
                                    
                                    await create_recommendation(prediction)
                                    predictions_made += 1
                                    
                                    logger.info(f"✅ SMART V4 PREDICTION: {event.get('home_team')} vs {event.get('away_team')} - "
                                              f"{smart_prediction.get('pick_type')}: {smart_prediction.get('pick')} "
                                              f"@ {smart_prediction.get('confidence')*100:.0f}% conf, "
                                              f"EV: {smart_prediction.get('expected_value', 0):.1f}%")
                                else:
                                    reason = smart_prediction.get("reasoning", "No value found") if smart_prediction else "Analysis failed"
                                    logger.info(f"⏭️ NO PICK: {event.get('home_team')} vs {event.get('away_team')} - {reason[:100]}")
                                
                                # Store odds snapshot for line movement tracking
                                await store_odds_snapshot(event)
                                
                                # Small delay between analyses
                                await asyncio.sleep(3)
                                
                        except Exception as e:
                            logger.error(f"Error analyzing event: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    
                except Exception as e:
                    logger.error(f"Error in smart pregame predictor for {sport_key}: {e}")
            
            if predictions_made > 0:
                logger.info(f"📊 Smart pre-game predictor complete - {predictions_made} Smart V4 predictions created")
            
            # Run every 10 minutes to catch all games in the 1 hour window
            await asyncio.sleep(600)
            
        except Exception as e:
            logger.error(f"Scheduled smart pregame predictor error: {e}")
            await asyncio.sleep(120)


async def get_line_movement_history(event_id: str) -> List[Dict]:
    """Get complete line movement history for an event"""
    history = []
    
    try:
        # Get opening odds
        opening = await db.opening_odds.find_one({"event_id": event_id}, {"_id": 0})
        if opening:
            history.append({
                "timestamp": opening.get("timestamp"),
                "home_odds": opening.get("home_odds"),
                "away_odds": opening.get("away_odds"),
                "is_opening": True
            })
        
        # Get hourly snapshots
        snapshots = await db.odds_history.find(
            {"event_id": event_id},
            {"_id": 0}
        ).sort("timestamp", 1).to_list(100)
        
        for snapshot in snapshots:
            history.append({
                "timestamp": snapshot.get("timestamp"),
                "home_odds": snapshot.get("home_odds"),
                "away_odds": snapshot.get("away_odds"),
                "is_opening": False
            })
        
    except Exception as e:
        logger.error(f"Error getting line movement history for {event_id}: {e}")
    
    return history

# Background task for line movement data cleanup
async def scheduled_line_movement_cleanup():
    """Background task that cleans up line movement data for events that have started"""
    # Wait 5 minutes on startup
    await asyncio.sleep(300)
    
    while True:
        try:
            logger.info("Running line movement cleanup for live/finished events...")
            now = datetime.now(timezone.utc)
            deleted_count = 0
            
            # Find events that have started and delete their line movement data
            event_ids = await db.odds_history.distinct("event_id")
            
            for event_id in event_ids:
                # Get opening odds to check commence time
                opening = await db.opening_odds.find_one({"event_id": event_id}, {"_id": 0})
                
                if opening:
                    # Check commence time
                    prediction = await db.predictions.find_one({"event_id": event_id}, {"commence_time": 1})
                    if prediction:
                        commence_str = prediction.get("commence_time", "")
                        if commence_str:
                            try:
                                commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                                if commence_time <= now:
                                    result = await db.odds_history.delete_many({"event_id": event_id})
                                    deleted_count += result.deleted_count
                            except Exception:
                                pass
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} line movement records for started events")
            
            await asyncio.sleep(1800)  # Run every 30 minutes
            
        except Exception as e:
            logger.error(f"Line movement cleanup error: {e}")
            await asyncio.sleep(300)

# Background task for ESPN odds refresh - NOW EVERY 15 MINUTES for better line tracking
async def scheduled_espn_odds_refresh():
    """Background task that refreshes ESPN odds every 15 MINUTES for pre-match events and line movement tracking"""
    global last_scrape_time
    
    # Wait 2 minutes on startup
    await asyncio.sleep(120)
    
    sports_to_refresh = ["basketball_nba", "americanfootball_nfl", "icehockey_nhl", "soccer_epl"]
    
    while True:
        try:
            logger.info("Running ESPN odds refresh every 15 minutes (pre-match only)...")
            now = datetime.now(timezone.utc)
            
            for sport_key in sports_to_refresh:
                try:
                    events = await fetch_espn_events_with_odds(sport_key, days_ahead=3)
                    
                    if events:
                        # Only store odds snapshots for PRE-MATCH events (not started)
                        prematch_count = 0
                        for event in events:
                            try:
                                commence_str = event.get("commence_time", "")
                                if commence_str:
                                    commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                                    if commence_time > now:
                                        await store_odds_snapshot(event)
                                        prematch_count += 1
                            except Exception:
                                pass
                        
                        # Update cache
                        cache_key = f"{sport_key}_h2h,spreads,totals"
                        events_cache[cache_key] = (events, now)
                        last_scrape_time = now.isoformat()
                        
                        logger.info(f"Refreshed {len(events)} events from ESPN for {sport_key} ({prematch_count} pre-match)")
                    
                    await asyncio.sleep(5)  # Small delay between sports
                    
                except Exception as e:
                    logger.error(f"Error refreshing ESPN odds for {sport_key}: {e}")
            
            logger.info("ESPN odds refresh complete (15 min interval)")
            
            await asyncio.sleep(900)  # Run every 15 MINUTES (was 3600)
            
        except Exception as e:
            logger.error(f"Scheduled ESPN refresh error: {e}")
            await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup"""
    # Start the scheduled result checker - INSTANT 10 SECOND sync for live games
    asyncio.create_task(scheduled_result_checker())
    logger.info("Started INSTANT live score sync - runs every 10 SECONDS with ESPN API")
    
    # Start line movement checker - NOW EVERY 15 MINUTES
    asyncio.create_task(scheduled_line_movement_checker())
    logger.info("Started line movement checker - runs every 15 MINUTES")
    
    # Start legacy recommendation generator (runs less frequently now)
    asyncio.create_task(scheduled_recommendation_generator())
    logger.info("Started legacy recommendation generator - runs every 4 hours")
    
    # START SMART PRE-GAME PREDICTOR V4 - PRIMARY PREDICTION SYSTEM (NO LLM)
    asyncio.create_task(scheduled_pregame_predictor())
    logger.info("📊 Started SMART PRE-GAME PREDICTOR V4 - analyzes games 1 hour before start, runs every 10 min, NO LLM REQUIRED")
    
    # Start ESPN odds refresh - NOW EVERY 15 MINUTES
    asyncio.create_task(scheduled_espn_odds_refresh())
    logger.info("Started ESPN odds refresh - runs every 15 MINUTES (pre-match events only)")
    
    # Start line movement data cleanup
    asyncio.create_task(scheduled_line_movement_cleanup())
    logger.info("Started line movement cleanup - runs every 30 minutes (deletes data for started events)")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
