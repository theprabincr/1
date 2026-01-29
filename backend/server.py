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

# Import BetPredictor V5 - Comprehensive Line Movement Analysis Engine
from betpredictor_v5 import generate_v5_prediction, BetPredictorV5
from line_movement_analyzer import analyze_line_movement, LineMovementAnalyzer

# Import BetPredictor V6 - Advanced Algorithm with ML and Ensemble
from betpredictor_v6 import generate_v6_prediction, BetPredictorV6

# Import Unified Predictor - Combines V5 + V6 with V6 weighted heavier
from unified_predictor import generate_unified_prediction, UnifiedBetPredictor

# Import lineup/roster scraper
from lineup_scraper import get_matchup_context, fetch_team_roster

# Import Adaptive Learning System for self-improving ML
from adaptive_learning import (
    AdaptiveLearningSystem,
    AdaptiveLogisticRegression,
    AdaptiveEnsemble,
    create_adaptive_learning_system
)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Adaptive Learning System (initialized on startup)
adaptive_learning: AdaptiveLearningSystem = None

# Events cache to reduce scraping (1 hour cache)
events_cache = {}
mock_events_cache = {}  # Cache for mock events
CACHE_DURATION_MINUTES = 60

# Last scrape timestamp for status
last_scrape_time = None

# Notification queue for line movement alerts
notification_queue = []

# ESPN provides DraftKings odds only
BOOKMAKERS = {
    'draftkings': 'DraftKings (via ESPN)',
}

# Sportsbook display names (for UI)
SPORTSBOOK_NAMES = {
    'draftkings': 'DraftKings (via ESPN)',
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
async def create_notification(notif_type: str, title: str, message: str, data: Dict = None, dedupe_window_minutes: int = 5):
    """Create a notification and store in database with deduplication
    
    Args:
        notif_type: Type of notification (e.g., 'daily_summary', 'result', 'new_pick')
        title: Notification title
        message: Notification message
        data: Optional additional data
        dedupe_window_minutes: Time window to check for duplicates (default 5 minutes)
    """
    # Check for duplicate notification within the dedupe window
    now = datetime.now(timezone.utc)
    dedupe_cutoff = (now - timedelta(minutes=dedupe_window_minutes)).isoformat()
    
    existing = await db.notifications.find_one({
        "type": notif_type,
        "title": title,
        "created_at": {"$gte": dedupe_cutoff}
    })
    
    if existing:
        logger.debug(f"Skipping duplicate notification: {notif_type} - {title}")
        return None
    
    notification = Notification(
        type=notif_type,
        title=title,
        message=message,
        data=data or {}
    )
    await db.notifications.insert_one(notification.model_dump())
    notification_queue.append(notification.model_dump())
    return notification

# ==================== DATA SOURCE STATUS ====================

@api_router.get("/data-source-status")
async def get_data_source_status():
    """Get ESPN data source status and line movement tracking info"""
    global last_scrape_time
    
    # Get cached events count
    total_cached = sum(len(events) for events, _ in events_cache.values())
    
    # Count line movement snapshots
    snapshot_count = await db.odds_history.count_documents({})
    
    return {
        "source": "ESPN/DraftKings",
        "status": "active",
        "lastUpdate": last_scrape_time,
        "cachedEvents": total_cached,
        "lineMovementSnapshots": snapshot_count,
        "refreshInterval": "5 minutes",
        "predictionWindow": "1 hour before game",
        "sports": list(events_cache.keys())
    }

# ==================== ESPN LINE MOVEMENT TRACKING ====================

@api_router.post("/refresh-odds")
async def manual_refresh_odds(sport_key: str = "basketball_nba"):
    """Manually trigger ESPN odds refresh and store snapshot for line movement"""
    global events_cache, last_scrape_time
    
    try:
        events = await fetch_espn_events_with_odds(sport_key, days_ahead=3)
        
        if events:
            # Store in cache
            cache_key = f"{sport_key}_h2h,spreads,totals"
            events_cache[cache_key] = (events, datetime.now(timezone.utc))
            last_scrape_time = datetime.now(timezone.utc).isoformat()
            
            # Store odds snapshots for line movement tracking
            snapshots_stored = 0
            for event in events:
                # Only store for pre-match events
                commence_str = event.get("commence_time", "")
                if commence_str:
                    try:
                        commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                        if commence_time > datetime.now(timezone.utc):
                            await store_odds_snapshot(event)
                            snapshots_stored += 1
                    except Exception:
                        pass
            
            return {
                "message": f"Refreshed {len(events)} events from ESPN",
                "snapshots_stored": snapshots_stored,
                "source": "ESPN/DraftKings"
            }
        else:
            return {"message": "No events found", "events": []}
            
    except Exception as e:
        logger.error(f"Manual refresh failed: {e}")
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")

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

@api_router.post("/notifications/test")
async def create_test_notification():
    """Create a test notification to verify the system works"""
    await create_notification(
        "recommendation",
        "Welcome to BetPredictor!",
        "Notifications are working correctly. You'll receive alerts for line movements and bet results.",
        {"test": True}
    )
    return {"message": "Test notification created"}

@api_router.post("/notifications/daily-summary")
async def trigger_daily_summary():
    """Manually trigger a daily summary notification"""
    try:
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get today's predictions
        predictions = await db.predictions.find({
            "created_at": {"$gte": today_start.isoformat()}
        }).to_list(100)
        
        # Get today's results
        completed = [p for p in predictions if p.get("result") in ["win", "loss", "push"]]
        wins = len([p for p in completed if p.get("result") == "win"])
        losses = len([p for p in completed if p.get("result") == "loss"])
        pending = len([p for p in predictions if p.get("result") == "pending"])
        
        # Calculate profit
        total_profit = 0
        for p in completed:
            odds = p.get("odds_at_prediction", 1.91)
            if p.get("result") == "win":
                total_profit += 100 * (odds - 1)
            elif p.get("result") == "loss":
                total_profit -= 100
        
        # Create summary message
        if len(predictions) > 0 or len(completed) > 0:
            win_rate = (wins / len(completed) * 100) if len(completed) > 0 else 0
            message = f"Today's Results: {wins}W-{losses}L ({win_rate:.0f}% win rate)\n"
            message += f"Profit/Loss: ${total_profit:+.2f}\n"
            message += f"Pending picks: {pending}"
        else:
            message = "No picks were generated today. Check back tomorrow!"
        
        await create_notification(
            "daily_summary",
            "📊 Daily Performance Summary",
            message,
            {
                "date": now.date().isoformat(),
                "wins": wins,
                "losses": losses,
                "pending": pending,
                "profit": total_profit
            }
        )
        
        return {"message": "Daily summary notification created", "data": {"wins": wins, "losses": losses, "profit": total_profit}}
    except Exception as e:
        return {"error": str(e)}

@api_router.post("/notifications/result-test")
async def trigger_result_notification():
    """Manually trigger a result notification for testing"""
    # Get a random completed prediction
    prediction = await db.predictions.find_one({"result": {"$in": ["win", "loss"]}})
    
    if prediction:
        result = prediction.get("result", "win")
        odds = prediction.get("odds_at_prediction", 1.91)
        profit = 100 * (odds - 1) if result == "win" else -100
        
        await create_notification(
            "result",
            f"🎯 Bet Result: {result.upper()}",
            f"{prediction.get('home_team')} vs {prediction.get('away_team')} - "
            f"Your pick: {prediction.get('predicted_outcome')} - {result.upper()}! (${profit:+.2f})",
            {
                "prediction_id": prediction.get("id"),
                "result": result,
                "pick": prediction.get("predicted_outcome"),
                "profit": profit
            }
        )
        return {"message": f"Result notification created for {result}", "profit": profit}
    else:
        # Create a sample result notification
        await create_notification(
            "result",
            "🎯 Bet Result: WIN",
            "Boston Celtics vs Portland Trail Blazers - Your pick: Boston Celtics -12.5 - WIN! (+$91.00)",
            {
                "result": "win",
                "pick": "Boston Celtics -12.5",
                "profit": 91.00
            }
        )
        return {"message": "Sample result notification created"}

@api_router.post("/notifications/new-pick-test")
async def trigger_new_pick_notification():
    """Manually trigger a new pick notification for testing"""
    # Get a random pending prediction
    prediction = await db.predictions.find_one({"result": "pending"})
    
    if prediction:
        await create_notification(
            "new_pick",
            f"🎯 New Pick: {prediction.get('home_team')} vs {prediction.get('away_team')}",
            f"{prediction.get('predicted_outcome')} ({prediction.get('prediction_type')}) "
            f"@ {prediction.get('confidence', 0.7)*100:.0f}% confidence",
            {
                "event_id": prediction.get("event_id"),
                "pick": prediction.get("predicted_outcome"),
                "pick_type": prediction.get("prediction_type"),
                "confidence": prediction.get("confidence", 0.7),
                "odds": prediction.get("odds_at_prediction", 1.91)
            }
        )
        return {"message": "New pick notification created", "pick": prediction.get("predicted_outcome")}
    else:
        # Create a sample new pick notification
        await create_notification(
            "new_pick",
            "🎯 New Pick: Boston Celtics vs Portland Trail Blazers",
            "Boston Celtics -12.5 (spread) @ 75% confidence",
            {
                "pick": "Boston Celtics -12.5",
                "pick_type": "spread",
                "confidence": 0.75,
                "odds": 1.91
            }
        )
        return {"message": "Sample new pick notification created"}

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
    
    # Get data source status
    snapshot_count = await db.odds_history.count_documents({})
    opening_count = await db.opening_odds.count_documents({})
    
    data_source_status = {
        "source": "ESPN/DraftKings",
        "status": "active",
        "line_movement_snapshots": snapshot_count,
        "opening_odds_tracked": opening_count
    }
    
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "performance": performance,
        "data_source_status": data_source_status
    }
    
    return report

# Routes
@api_router.get("/")
async def root():
    return {"message": "BetPredictor API v1.0", "status": "running"}

@api_router.get("/sports", response_model=List[Sport])
async def list_sports():
    """Get list of available sports"""
    # Return hardcoded sports list (ESPN supported)
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
    """Get line movement history for an event including ALL markets (ML, Spread, Totals)"""
    
    # Get opening odds
    opening = await db.opening_odds.find_one({"event_id": event_id}, {"_id": 0})
    
    # Get all snapshots sorted by timestamp
    snapshots = await db.odds_history.find(
        {"event_id": event_id},
        {"_id": 0}
    ).sort("timestamp", 1).to_list(500)
    
    # Get event info
    event_info = None
    current_odds = None
    current_spread = None
    current_total = None
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
                spreads = []
                totals = []
                for bm in bookmakers:
                    for market in bm.get("markets", []):
                        if market.get("key") == "h2h":
                            outcomes = market.get("outcomes", [])
                            if len(outcomes) >= 2:
                                home_prices.append(outcomes[0].get("price", 0))
                                away_prices.append(outcomes[1].get("price", 0))
                        elif market.get("key") == "spreads":
                            for outcome in market.get("outcomes", []):
                                if event.get("home_team", "").lower() in outcome.get("name", "").lower():
                                    spreads.append(outcome.get("point"))
                        elif market.get("key") == "totals":
                            for outcome in market.get("outcomes", []):
                                if "over" in outcome.get("name", "").lower():
                                    totals.append(outcome.get("point"))
                if home_prices and away_prices:
                    current_odds = {
                        "home": round(sum(home_prices) / len(home_prices), 2),
                        "away": round(sum(away_prices) / len(away_prices), 2)
                    }
                if spreads:
                    current_spread = round(sum(spreads) / len(spreads), 1)
                if totals:
                    current_total = round(sum(totals) / len(totals), 1)
            break
    
    # Build chart data for ALL markets
    ml_chart = []
    spread_chart = []
    totals_chart = []
    seen_time_keys = set()
    
    for snap in snapshots:
        ts = snap.get("timestamp")
        time_key = snap.get("time_key", ts[:16] if ts else None)
        
        if time_key and time_key not in seen_time_keys:
            seen_time_keys.add(time_key)
            
            # ML data
            home = snap.get("home_odds")
            away = snap.get("away_odds")
            if home and away:
                ml_chart.append({
                    "timestamp": ts,
                    "home_odds": round(home, 2),
                    "away_odds": round(away, 2),
                    "num_bookmakers": snap.get("num_bookmakers", 1)
                })
            
            # Spread data
            spread = snap.get("spread")
            if spread is not None:
                spread_chart.append({
                    "timestamp": ts,
                    "spread": round(spread, 1),
                    "odds": snap.get("spread_odds", 1.91)
                })
            
            # Totals data
            total = snap.get("total")
            if total is not None:
                totals_chart.append({
                    "timestamp": ts,
                    "total": round(total, 1),
                    "over_odds": snap.get("over_odds", 1.91),
                    "under_odds": snap.get("under_odds", 1.91)
                })
    
    # Add opening data if no chart data
    if opening:
        if not ml_chart and opening.get("home_odds"):
            ml_chart.append({
                "timestamp": opening.get("timestamp"),
                "home_odds": opening.get("home_odds"),
                "away_odds": opening.get("away_odds"),
                "num_bookmakers": len(opening.get("bookmakers", []))
            })
        if not spread_chart and opening.get("spread") is not None:
            spread_chart.append({
                "timestamp": opening.get("timestamp"),
                "spread": opening.get("spread"),
                "odds": opening.get("spread_odds", 1.91)
            })
        if not totals_chart and opening.get("total") is not None:
            totals_chart.append({
                "timestamp": opening.get("timestamp"),
                "total": opening.get("total"),
                "over_odds": opening.get("over_odds", 1.91),
                "under_odds": opening.get("under_odds", 1.91)
            })
    
    # Add current values if different from last
    now_ts = datetime.now(timezone.utc).isoformat()
    
    if current_odds and ml_chart:
        last = ml_chart[-1]
        if abs(last["home_odds"] - current_odds["home"]) > 0.01:
            ml_chart.append({
                "timestamp": now_ts,
                "home_odds": current_odds["home"],
                "away_odds": current_odds["away"],
                "num_bookmakers": 0
            })
    
    if current_spread is not None and spread_chart:
        last = spread_chart[-1]
        if abs(last["spread"] - current_spread) > 0.1:
            spread_chart.append({
                "timestamp": now_ts,
                "spread": current_spread,
                "odds": 1.91
            })
    
    if current_total is not None and totals_chart:
        last = totals_chart[-1]
        if abs(last["total"] - current_total) > 0.1:
            totals_chart.append({
                "timestamp": now_ts,
                "total": current_total,
                "over_odds": 1.91,
                "under_odds": 1.91
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
                "home_ml": bm_snap.get("home_ml"),
                "away_ml": bm_snap.get("away_ml"),
                "spread": bm_snap.get("home_spread"),
                "total": bm_snap.get("total_line")
            })
    
    # Sort charts
    ml_chart.sort(key=lambda x: x.get("timestamp", "") or "")
    spread_chart.sort(key=lambda x: x.get("timestamp", "") or "")
    totals_chart.sort(key=lambda x: x.get("timestamp", "") or "")
    
    return {
        "event_id": event_id,
        "event_info": event_info,
        "opening_odds": {
            "ml": {"home": opening.get("home_odds"), "away": opening.get("away_odds")} if opening else None,
            "spread": opening.get("spread") if opening else None,
            "total": opening.get("total") if opening else None,
            "timestamp": opening.get("timestamp") if opening else None
        },
        "current_odds": {
            "ml": current_odds,
            "spread": current_spread,
            "total": current_total
        },
        "bookmakers": list(by_bookmaker.values()),
        "chart_data": {
            "moneyline": ml_chart,
            "spread": spread_chart,
            "totals": totals_chart
        },
        "total_snapshots": len(snapshots)
    }

@api_router.post("/analyze")
async def analyze_game(request: AnalysisRequest):
    """Get ML-based analysis for a game using V5 and V6 algorithms"""
    # Build event object for analysis
    event = {
        "id": request.event_id,
        "home_team": request.home_team,
        "away_team": request.away_team,
        "odds": request.odds_data,
        "commence_time": datetime.now(timezone.utc).isoformat()
    }
    
    # Get matchup data
    matchup_data = {
        "home_team": {"name": request.home_team, "form": {}, "stats": {}},
        "away_team": {"name": request.away_team, "form": {}, "stats": {}},
        "odds": request.odds_data
    }
    
    # Get line movement history
    line_history = request.line_movement or []
    opening_odds = {}
    current_odds = request.odds_data or {}
    
    # Run V6 analysis (comprehensive ML-based)
    try:
        v6_result = await generate_v6_prediction(
            event=event,
            sport_key=request.sport_key,
            squad_data={"home_team": {"injuries": []}, "away_team": {"injuries": []}},
            matchup_data=matchup_data,
            line_movement_history=line_history,
            opening_odds=opening_odds,
            current_odds=current_odds
        )
    except Exception as e:
        v6_result = {"error": str(e), "has_pick": False}
    
    # Run V5 analysis (line movement focused)
    try:
        v5_result = await generate_v5_prediction(
            event=event,
            sport_key=request.sport_key,
            squad_data={"home_team": {"injuries": []}, "away_team": {"injuries": []}},
            matchup_data=matchup_data,
            line_movement_history=line_history,
            opening_odds=opening_odds,
            current_odds=current_odds
        )
    except Exception as e:
        v5_result = {"error": str(e), "has_pick": False}
    
    return {
        "event_id": request.event_id,
        "home_team": request.home_team,
        "away_team": request.away_team,
        "v5_analysis": v5_result,
        "v6_analysis": v6_result,
        "analysis_type": "ML-based (V5 Line Movement + V6 Ensemble)",
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
    """Get ML-generated bet recommendations - filtered by 70%+ confidence and time window"""
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
@api_router.post("/force-generate-picks")
async def force_generate_picks():
    """
    Force immediate analysis of games in the prediction window.
    Note: Predictions are automatically generated 1 hour before games.
    This endpoint manually triggers analysis for games currently in the 45-75 minute window.
    """
    return {
        "message": "BetPredictor V5 automatically generates picks 1 hour before game start",
        "note": "Use POST /api/analyze-v5/{event_id} to manually analyze a specific event"
    }


async def get_line_movement_data(event_id: str) -> Dict:
    """Get line movement data for an event from our database"""
    try:
        opening = await db.opening_odds.find_one({"event_id": event_id}, {"_id": 0})
        if opening:
            return {
                "opening_home_odds": opening.get("home_odds"),
                "opening_away_odds": opening.get("away_odds"),
                "opening_spread": opening.get("spread"),
                "opening_total": opening.get("total"),
                "opening_time": opening.get("timestamp")
            }
    except Exception as e:
        logger.error(f"Error getting line movement for {event_id}: {e}")
    return {}


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
            prediction_type = prediction.get("prediction_type", "moneyline")
            
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
            
            # Get predicted team/side
            predicted_outcome = prediction.get("predicted_outcome", "")
            home_team = prediction.get("home_team", current_event.get("home_team", ""))
            away_team = prediction.get("away_team", current_event.get("away_team", ""))
            
            # Determine if prediction is for home or away team
            is_home_pick = False
            is_away_pick = False
            
            # Check if predicted outcome matches home team
            if predicted_outcome and home_team:
                if predicted_outcome.lower() in home_team.lower() or home_team.lower() in predicted_outcome.lower():
                    is_home_pick = True
            
            # Check if predicted outcome matches away team
            if predicted_outcome and away_team:
                if predicted_outcome.lower() in away_team.lower() or away_team.lower() in predicted_outcome.lower():
                    is_away_pick = True
            
            # If still not determined, check for "over"/"under" for totals
            if not is_home_pick and not is_away_pick:
                if "over" in predicted_outcome.lower():
                    is_home_pick = True  # Use home_odds position for over
                elif "under" in predicted_outcome.lower():
                    is_away_pick = True  # Use away_odds position for under
            
            # Get current odds for the CORRECT team
            current_odds_value = None
            
            for bm in current_event.get("bookmakers", []):
                for market in bm.get("markets", []):
                    market_key = market.get("key", "h2h")
                    
                    # Match market type to prediction type
                    if prediction_type == "moneyline" and market_key == "h2h":
                        outcomes = market.get("outcomes", [])
                        if len(outcomes) >= 2:
                            # First outcome is typically home, second is away
                            if is_home_pick:
                                current_odds_value = outcomes[0].get("price")
                            elif is_away_pick:
                                current_odds_value = outcomes[1].get("price")
                            else:
                                # Try to match by name
                                for outcome in outcomes:
                                    name = outcome.get("name", "").lower()
                                    if predicted_outcome.lower() in name or name in predicted_outcome.lower():
                                        current_odds_value = outcome.get("price")
                                        break
                    
                    elif prediction_type == "spread" and market_key == "spreads":
                        for outcome in market.get("outcomes", []):
                            name = outcome.get("name", "").lower()
                            if (is_home_pick and (home_team.lower() in name or "home" in name)) or \
                               (is_away_pick and (away_team.lower() in name or "away" in name)):
                                current_odds_value = outcome.get("price")
                                break
                    
                    elif prediction_type == "total" and market_key == "totals":
                        for outcome in market.get("outcomes", []):
                            name = outcome.get("name", "").lower()
                            if ("over" in predicted_outcome.lower() and "over" in name) or \
                               ("under" in predicted_outcome.lower() and "under" in name):
                                current_odds_value = outcome.get("price")
                                break
                    
                    if current_odds_value:
                        break
                if current_odds_value:
                    break
            
            # Skip if we couldn't find current odds
            if not current_odds_value or current_odds_value <= 1.0:
                logger.debug(f"Could not find current odds for {predicted_outcome} in {home_team} vs {away_team}")
                continue
            
            # Check for significant line movement
            if original_odds > 0:
                change_pct = abs(current_odds_value - original_odds) / original_odds * 100
                
                if change_pct >= threshold:
                    # Determine movement direction
                    # Odds DECREASING = market moving TOWARD our pick = FAVORABLE
                    # Odds INCREASING = market moving AGAINST our pick = ADVERSE
                    if current_odds_value < original_odds:
                        movement_direction = "favorable"
                        # Market agrees with our pick - slight confidence boost
                        new_confidence = min(prediction.get("confidence", 0.6) + 0.03, 0.95)
                        movement_note = "Market moving toward our pick"
                    else:
                        movement_direction = "adverse"
                        # Market moving against our pick - decrease confidence
                        new_confidence = max(prediction.get("confidence", 0.6) - 0.08, 0.3)
                        movement_note = "Market moving against our pick"
                    
                    # Line moved significantly - add note to analysis
                    new_analysis = prediction.get("analysis", "") + f"\n\n[LINE MOVEMENT UPDATE: {predicted_outcome} odds moved from {original_odds:.2f} to {current_odds_value:.2f} ({change_pct:.1f}% {movement_direction}). {movement_note}]"
                    
                    await db.predictions.update_one(
                        {"id": prediction.get("id")},
                        {"$set": {
                            "analysis": new_analysis,
                            "confidence": new_confidence,
                            "current_odds": current_odds_value,
                            "line_movement_pct": change_pct,
                            "last_updated": datetime.now(timezone.utc).isoformat()
                        }}
                    )
                    logger.info(f"Updated prediction {prediction.get('id')} - {predicted_outcome} odds moved {change_pct:.1f}% ({movement_direction}): {original_odds:.2f} → {current_odds_value:.2f}")
                    
                    # Create notification for significant line movement
                    if settings and settings.get('notification_preferences', {}).get('line_movement_alerts', True):
                        await create_notification(
                            "line_movement",
                            f"Line Movement Alert: {home_team} vs {away_team}",
                            f"{predicted_outcome} odds moved {change_pct:.1f}% ({movement_direction}): {original_odds:.2f} → {current_odds_value:.2f}. {movement_note}.",
                            {
                                "prediction_id": prediction.get("id"),
                                "event_id": event_id,
                                "predicted_outcome": predicted_outcome,
                                "original_odds": original_odds,
                                "current_odds": current_odds_value,
                                "change_pct": change_pct,
                                "direction": movement_direction
                            }
                        )
                    
    except Exception as e:
        logger.error(f"Error in update_recommendations_on_line_movement: {e}")
        import traceback
        traceback.print_exc()

@api_router.put("/result")
async def update_result(update: ResultUpdate):
    """Update the result of a prediction and remove analysis"""
    result = await db.predictions.update_one(
        {"id": update.prediction_id},
        {
            "$set": {"result": update.result, "result_updated_at": datetime.now(timezone.utc).isoformat()},
            "$unset": {"analysis": "", "reasoning": ""}  # Remove analysis once result is recorded
        }
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
                                # Update prediction with result and remove analysis
                                await db.predictions.update_one(
                                    {"id": prediction.get("id")},
                                    {
                                        "$set": {
                                            "result": result,
                                            "result_updated_at": now.isoformat(),
                                            "final_score": {
                                                "home": matching_game.get("home_score"),
                                                "away": matching_game.get("away_score"),
                                                "total": matching_game.get("total_score")
                                            }
                                        },
                                        "$unset": {"analysis": "", "reasoning": ""}  # Remove analysis once result is recorded
                                    }
                                )
                                
                                results_updated += 1
                                logger.info(f"Updated prediction {prediction.get('id')}: {result} "
                                          f"({matching_game.get('home_team')} {matching_game.get('home_score')} - "
                                          f"{matching_game.get('away_score')} {matching_game.get('away_team')})")
                                
                                # Clean up line movement data for this completed event
                                event_id = prediction.get("event_id")
                                if event_id:
                                    deleted_history = await db.odds_history.delete_many({"event_id": event_id})
                                    deleted_opening = await db.opening_odds.delete_one({"event_id": event_id})
                                    logger.info(f"Cleaned up line movement data for {event_id}: "
                                              f"{deleted_history.deleted_count} snapshots, "
                                              f"{deleted_opening.deleted_count} opening odds")
                                
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
                                
                                # 🧠 ADAPTIVE LEARNING: Update model performance from this result
                                if adaptive_learning:
                                    try:
                                        # Determine the actual winner
                                        home_score = matching_game.get("home_score", 0)
                                        away_score = matching_game.get("away_score", 0)
                                        
                                        if prediction.get("prediction_type") == "moneyline":
                                            actual_winner = prediction.get("home_team") if home_score > away_score else prediction.get("away_team")
                                        else:
                                            # For spread/totals, the "winner" is which side covered
                                            actual_winner = prediction.get("predicted_outcome") if result == "win" else "opponent"
                                        
                                        await adaptive_learning.update_model_performance_from_result(
                                            prediction_id=prediction.get("id"),
                                            actual_winner=actual_winner,
                                            sport_key=sport_key
                                        )
                                        logger.info(f"🧠 Adaptive learning updated for prediction {prediction.get('id')}")
                                    except Exception as learn_err:
                                        logger.error(f"Adaptive learning error: {learn_err}")
                        
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
    deleted_history_count = 0
    deleted_opening_count = 0
    
    # Get all events with line movement data (from odds_history)
    event_ids = await db.odds_history.distinct("event_id")
    
    for event_id in event_ids:
        # Get opening odds to check commence time (stored directly in opening_odds)
        opening = await db.opening_odds.find_one({"event_id": event_id})
        
        commence_str = None
        if opening:
            commence_str = opening.get("commence_time")
        
        # If no commence_time in opening odds, check odds_history
        if not commence_str:
            history_record = await db.odds_history.find_one({"event_id": event_id})
            if history_record:
                commence_str = history_record.get("commence_time")
        
        # If still no commence_time, try predictions
        if not commence_str:
            prediction = await db.predictions.find_one({"event_id": event_id}, {"commence_time": 1})
            if prediction:
                commence_str = prediction.get("commence_time")
        
        if commence_str:
            try:
                commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                
                # If event has started, delete line movement data
                if commence_time <= now:
                    # Delete odds history for this event
                    result = await db.odds_history.delete_many({"event_id": event_id})
                    deleted_history_count += result.deleted_count
                    
                    # Also delete opening odds since the event has started
                    opening_result = await db.opening_odds.delete_one({"event_id": event_id})
                    deleted_opening_count += opening_result.deleted_count
                    
                    logger.info(f"Cleaned up line movement for event {event_id} (started)")
            except Exception as e:
                logger.error(f"Error processing event {event_id}: {e}")
    
    return {
        "message": f"Cleaned up line movement data: {deleted_history_count} history records, {deleted_opening_count} opening odds",
        "deleted_history_count": deleted_history_count,
        "deleted_opening_count": deleted_opening_count,
        "total_deleted": deleted_history_count + deleted_opening_count
    }

# BetPredictor V5 - Get predictions
@api_router.get("/predictions/v5")
async def get_v5_predictions(limit: int = 50, result: str = None):
    """Get predictions made by BetPredictor V5 algorithm (comprehensive line movement analysis)"""
    query = {"ai_model": "betpredictor_v5"}
    if result:
        query["result"] = result
    
    predictions = await db.predictions.find(
        query, 
        {"_id": 0}
    ).sort("created_at", -1).limit(limit).to_list(limit)
    
    # Calculate V5 stats
    all_v5 = await db.predictions.find({"ai_model": "betpredictor_v5"}, {"_id": 0}).to_list(10000)
    
    wins = len([p for p in all_v5 if p.get("result") == "win"])
    losses = len([p for p in all_v5 if p.get("result") == "loss"])
    pending = len([p for p in all_v5 if p.get("result") == "pending"])
    
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg_confidence = sum(p.get("confidence", 0) for p in all_v5) / len(all_v5) if all_v5 else 0
    
    return {
        "predictions": predictions,
        "stats": {
            "total": len(all_v5),
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": round(win_rate, 1),
            "avg_confidence": round(avg_confidence, 1)
        },
        "algorithm": "betpredictor_v5",
        "description": "Comprehensive analysis: Complete line movement study, sharp money detection, squad/H2H/venue analysis. Only recommends when 4+ factors align."
    }


# NEW: Analyze event with V5 engine - comprehensive line movement study
@api_router.post("/analyze-v5/{event_id}")
async def analyze_event_v5(event_id: str, sport_key: str = "basketball_nba"):
    """
    Comprehensive V5 analysis for an event including:
    - Complete line movement study from opening to now
    - Sharp money detection
    - Squad and injury analysis
    - H2H historical analysis
    - Venue factors
    - Confidence based on factor alignment
    """
    try:
        # 1. Fetch event
        events = await fetch_espn_events_with_odds(sport_key, days_ahead=3)
        event = None
        for e in events:
            if e.get("id") == event_id:
                event = e
                break
        
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")
        
        logger.info(f"🎯 V5 Analysis: {home_team} vs {away_team}")
        
        # 2. Get complete line movement history
        opening_odds = await db.opening_odds.find_one({"event_id": event_id}, {"_id": 0})
        line_history = await db.odds_history.find(
            {"event_id": event_id},
            {"_id": 0}
        ).sort("timestamp", 1).to_list(500)
        
        # Current odds from event
        current_odds = {}
        bookmakers = event.get("bookmakers", [])
        if bookmakers:
            for bm in bookmakers:
                for market in bm.get("markets", []):
                    if market.get("key") == "h2h":
                        outcomes = market.get("outcomes", [])
                        if len(outcomes) >= 2:
                            current_odds["home"] = outcomes[0].get("price", 1.91)
                            current_odds["away"] = outcomes[1].get("price", 1.91)
                        break
        
        # 3. Get squad data
        squad_data = await get_matchup_context(home_team, away_team, sport_key)
        
        # 4. Get matchup data
        matchup_data = await get_comprehensive_matchup_data(event, sport_key)
        
        # 5. Run V5 analysis
        prediction = await generate_v5_prediction(
            event,
            sport_key,
            squad_data,
            matchup_data,
            line_history,
            opening_odds or {},
            current_odds
        )
        
        # 6. Also run standalone line movement analysis for detailed view
        line_analysis = await analyze_line_movement(
            line_history,
            opening_odds or {},
            current_odds,
            commence_time,
            home_team,
            away_team
        )
        
        # 7. If has pick, optionally store it
        if prediction.get("has_pick"):
            # Check if prediction already exists
            existing = await db.predictions.find_one({
                "event_id": event_id,
                "ai_model": "betpredictor_v5"
            })
            
            if not existing:
                prediction_record = {
                    "id": str(uuid.uuid4()),
                    "event_id": event_id,
                    "sport_key": sport_key,
                    "home_team": home_team,
                    "away_team": away_team,
                    "prediction": prediction["pick"],
                    "prediction_type": prediction["pick_type"],
                    "odds": prediction["odds"],
                    "confidence": prediction["confidence"] / 100,
                    "edge_percent": prediction["edge_percent"],
                    "reasoning": prediction["reasoning"],
                    "result": "pending",
                    "commence_time": commence_time,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "ai_model": "betpredictor_v5",
                    "factor_count": prediction.get("factor_count", 0),
                    "line_analysis_summary": prediction.get("line_analysis_summary")
                }
                await db.predictions.insert_one(prediction_record)
                logger.info(f"✅ Stored V5 prediction: {prediction['pick']} @ {prediction['confidence']}%")
        
        return {
            "event": {
                "id": event_id,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": commence_time,
                "sport_key": sport_key
            },
            "prediction": prediction,
            "line_movement_analysis": {
                "total_movement_pct": line_analysis.get("total_movement_pct"),
                "movement_direction": line_analysis.get("movement_direction"),
                "sharp_money_side": line_analysis.get("sharp_money_side"),
                "reverse_line_movement": line_analysis.get("reverse_line_movement"),
                "steam_moves": line_analysis.get("steam_moves", []),
                "key_insights": line_analysis.get("key_insights", []),
                "summary": line_analysis.get("summary"),
                "phases": line_analysis.get("movement_phases", [])
            },
            "data_summary": {
                "line_movement_snapshots": len(line_history),
                "has_opening_odds": opening_odds is not None,
                "squad_data_available": bool(squad_data.get("home_team") or squad_data.get("away_team"))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V5 analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Algorithm Performance Stats (V5 only)
@api_router.get("/predictions/stats")
async def get_prediction_stats():
    """Get BetPredictor V5 performance statistics"""
    
    all_predictions = await db.predictions.find({"ai_model": "betpredictor_v5"}, {"_id": 0}).to_list(10000)
    
    completed = [p for p in all_predictions if p.get("result") in ["win", "loss"]]
    wins = len([p for p in completed if p.get("result") == "win"])
    losses = len([p for p in completed if p.get("result") == "loss"])
    pending = len([p for p in all_predictions if p.get("result") == "pending"])
    
    # Count by pick type
    ml_picks = len([p for p in all_predictions if p.get("prediction_type") == "moneyline"])
    spread_picks = len([p for p in all_predictions if p.get("prediction_type") == "spread"])
    total_picks = len([p for p in all_predictions if p.get("prediction_type") == "total"])
    
    win_rate = wins / len(completed) * 100 if completed else 0
    avg_confidence = sum(p.get("confidence", 0) for p in all_predictions) / len(all_predictions) * 100 if all_predictions else 0
    
    return {
        "algorithm": "betpredictor_v5",
        "description": "Comprehensive line movement analysis with sharp money detection, squad/H2H/venue factors",
        "stats": {
            "total": len(all_predictions),
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
    }


# ==================== BETPREDICTOR V6 ENDPOINTS ====================
# Advanced algorithm with Phase 1-3 enhancements

@api_router.get("/predictions/v6")
async def get_v6_predictions(limit: int = 50, result: str = None):
    """Get BetPredictor V6 predictions (advanced algorithm)"""
    query = {"ai_model": "betpredictor_v6"}
    if result:
        query["result"] = result
    
    predictions = await db.predictions.find(query, {"_id": 0}).sort("created_at", -1).limit(limit).to_list(limit)
    
    # Stats
    all_v6 = await db.predictions.find({"ai_model": "betpredictor_v6"}, {"_id": 0}).to_list(10000)
    
    wins = len([p for p in all_v6 if p.get("result") == "win"])
    losses = len([p for p in all_v6 if p.get("result") == "loss"])
    pending = len([p for p in all_v6 if p.get("result") == "pending"])
    
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg_confidence = sum(p.get("confidence", 0) for p in all_v6) / len(all_v6) if all_v6 else 0
    
    return {
        "predictions": predictions,
        "stats": {
            "total": len(all_v6),
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": round(win_rate, 1),
            "avg_confidence": round(avg_confidence, 1)
        },
        "algorithm": "betpredictor_v6",
        "description": "Advanced ML ensemble with ELO, context, simulations, and psychology analysis"
    }


@api_router.post("/analyze-v6/{event_id}")
async def analyze_event_v6(event_id: str, sport_key: str = "basketball_nba"):
    """
    Analyze a specific event using BetPredictor V6 (Advanced Algorithm)
    
    Features:
    - Phase 1: ELO ratings, context (rest/travel/altitude), smart injury weighting, advanced metrics
    - Phase 2: Monte Carlo simulations, Poisson modeling, market psychology, contrarian opportunities
    - Phase 3: Logistic regression, 5-model ensemble, Kelly Criterion, historical tracking
    """
    
    # Fetch event data from ESPN
    events = await fetch_espn_events_with_odds(sport_key, days_ahead=3)
    event = next((e for e in events if e.get("id") == event_id or e.get("espn_id") == event_id), None)
    
    if not event:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found for sport {sport_key}")
    
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    
    logger.info(f"🚀 V6 Manual Analysis: {home_team} vs {away_team}")
    
    # Get comprehensive matchup data
    matchup_data = await get_comprehensive_matchup_data(event, sport_key)
    
    # Get squad data (injuries and rosters)
    squad_data = {
        "home_team": {"injuries": [], "roster": [], "key_players": []},
        "away_team": {"injuries": [], "roster": [], "key_players": []}
    }
    
    try:
        # Fetch rosters using team names
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        
        home_roster = await fetch_team_roster(home_team, sport_key)
        away_roster = await fetch_team_roster(away_team, sport_key)
        
        squad_data["home_team"]["injuries"] = home_roster.get("injuries", [])
        squad_data["home_team"]["roster"] = home_roster.get("players", [])
        squad_data["home_team"]["key_players"] = home_roster.get("key_players", [])
        
        squad_data["away_team"]["injuries"] = away_roster.get("injuries", [])
        squad_data["away_team"]["roster"] = away_roster.get("players", [])
        squad_data["away_team"]["key_players"] = away_roster.get("key_players", [])
        
        logger.info(f"Fetched rosters: {home_team} ({len(home_roster.get('players', []))} players), "
                   f"{away_team} ({len(away_roster.get('players', []))} players)")
    except Exception as e:
        logger.warning(f"Could not fetch roster data: {e}")
    
    # Get line movement history
    line_movement_history = []
    opening_odds = {}
    
    try:
        history_query = {"event_id": event_id}
        history_docs = await db.odds_history.find(history_query).sort("timestamp", 1).to_list(1000)
        line_movement_history = history_docs
        
        if history_docs:
            opening_odds = history_docs[0]
        
        # Also check opening_odds collection
        opening_doc = await db.opening_odds.find_one({"event_id": event_id})
        if opening_doc:
            opening_odds = opening_doc
    except Exception as e:
        logger.warning(f"Could not fetch line movement history: {e}")
    
    # Get current odds
    current_odds = event.get("odds", {})
    
    # Run V6 analysis
    try:
        prediction = await generate_v6_prediction(
            event,
            sport_key,
            squad_data,
            matchup_data,
            line_movement_history,
            opening_odds,
            current_odds
        )
        
        # Save prediction if it has a pick
        if prediction.get("has_pick"):
            prediction_doc = {
                "id": str(uuid.uuid4()),
                "event_id": event_id,
                "sport_key": sport_key,
                "home_team": home_team,
                "away_team": away_team,
                "prediction": prediction.get("pick"),
                "confidence": prediction.get("confidence", 0) / 100,
                "odds": prediction.get("odds", 1.91),
                "ai_model": "betpredictor_v6",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "commence_time": event.get("commence_time", ""),
                "prediction_type": prediction.get("pick_type", "moneyline"),
                "result": "pending",
                "reasoning": prediction.get("reasoning", ""),
                "edge": prediction.get("edge", 0),
                "model_agreement": prediction.get("model_agreement", 0),
                "ensemble_confidence": prediction.get("confidence", 0),
                "predicted_outcome": prediction.get("pick", "")  # For result tracking
            }
            
            await db.predictions.insert_one(prediction_doc)
            logger.info(f"✅ V6 prediction saved: {prediction.get('pick')} at {prediction.get('confidence')}% confidence")
            
            # 🧠 ADAPTIVE LEARNING: Store individual model predictions for tracking
            if adaptive_learning and prediction.get("ensemble_details"):
                try:
                    individual_preds = prediction.get("ensemble_details", {}).get("individual_predictions", {})
                    if individual_preds:
                        await adaptive_learning.record_individual_model_predictions(
                            prediction_id=prediction_doc["id"],
                            event_id=event_id,
                            sport_key=sport_key,
                            model_predictions=individual_preds,
                            final_pick=prediction.get("pick"),
                            home_team=home_team,
                            away_team=away_team
                        )
                        logger.info(f"🧠 Stored individual model predictions for adaptive learning")
                except Exception as learn_err:
                    logger.warning(f"Could not store individual model predictions: {learn_err}")
        
        return {
            "event": {
                "id": event_id,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": event.get("commence_time", ""),
                "sport_key": sport_key
            },
            "prediction": prediction,
            "data_summary": {
                "line_movement_snapshots": len(line_movement_history),
                "has_opening_odds": bool(opening_odds),
                "squad_data_available": bool(squad_data.get("home_team", {}).get("injuries") or squad_data.get("away_team", {}).get("injuries")),
                "matchup_data_available": bool(matchup_data)
            }
        }
    
    except Exception as e:
        logger.error(f"Error in V6 analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"V6 analysis failed: {str(e)}")


@api_router.get("/predictions/comparison")
async def get_algorithm_comparison():
    """Compare performance of different prediction algorithms"""
    
    algorithms = ["betpredictor_v5", "betpredictor_v6", "unified"]
    comparison = {}
    
    for algo in algorithms:
        predictions = await db.predictions.find({"ai_model": algo}, {"_id": 0}).to_list(10000)
        completed = [p for p in predictions if p.get("result") in ["win", "loss"]]
        
        wins = len([p for p in completed if p.get("result") == "win"])
        losses = len([p for p in completed if p.get("result") == "loss"])
        pending = len([p for p in predictions if p.get("result") == "pending"])
        
        win_rate = wins / len(completed) * 100 if completed else 0
        avg_confidence = sum(p.get("confidence", 0) for p in predictions) / len(predictions) if predictions else 0
        
        # Calculate by pick type
        ml_picks = len([p for p in predictions if p.get("prediction_type") == "moneyline"])
        spread_picks = len([p for p in predictions if p.get("prediction_type") == "spread"])
        total_picks = len([p for p in predictions if p.get("prediction_type") == "total"])
        
        description = ""
        if algo == "betpredictor_v5":
            description = "Line movement analysis with sharp money detection"
        elif algo == "betpredictor_v6":
            description = "Advanced ML ensemble: ELO + Context + Simulations + Psychology + 5-model voting"
        elif algo == "unified":
            description = "🔄 UNIFIED: Combines V5 + V6 (V6 weighted 70%, V5 weighted 30%) - Single pick per game"
        
        comparison[algo] = {
            "total": len(predictions),
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": round(win_rate, 1),
            "avg_confidence": round(avg_confidence * 100 if avg_confidence < 1 else avg_confidence, 1),
            "pick_types": {
                "moneyline": ml_picks,
                "spread": spread_picks,
                "total": total_picks
            },
            "description": description
        }
    
    return {
        "algorithms": comparison,
        "recommendation": "UNIFIED algorithm combines best of V5 and V6 - single pick per game with ML-driven decisions"
    }


@api_router.get("/model-performance")
async def get_model_performance():
    """Get performance metrics for individual V6 sub-models"""
    from ml_models import ModelPerformanceTracker
    
    stats = ModelPerformanceTracker.get_model_stats()
    
    return {
        "sub_models": stats,
        "description": "V6 uses 5 sub-models: ELO, Context, Line Movement, Statistical, and Psychology",
        "note": "Weights are automatically adjusted based on historical performance"
    }


# ==================== UNIFIED PREDICTOR ENDPOINTS ====================
# Combines V5 + V6 with V6 weighted 70%

@api_router.get("/predictions/unified")
async def get_unified_predictions(limit: int = 50, result: str = None):
    """Get unified predictions (V5 + V6 combined with V6 weighted 70%)"""
    query = {"ai_model": "unified"}
    if result:
        query["result"] = result
    
    predictions = await db.predictions.find(query, {"_id": 0}).sort("created_at", -1).limit(limit).to_list(limit)
    
    # Stats
    all_unified = await db.predictions.find({"ai_model": "unified"}, {"_id": 0}).to_list(10000)
    
    wins = len([p for p in all_unified if p.get("result") == "win"])
    losses = len([p for p in all_unified if p.get("result") == "loss"])
    pending = len([p for p in all_unified if p.get("result") == "pending"])
    
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg_confidence = sum(p.get("confidence", 0) for p in all_unified) / len(all_unified) if all_unified else 0
    
    return {
        "predictions": predictions,
        "stats": {
            "total": len(all_unified),
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": round(win_rate, 1),
            "avg_confidence": round(avg_confidence * 100 if avg_confidence < 1 else avg_confidence, 1)
        },
        "algorithm": "unified",
        "description": "Combines V5 (line movement) + V6 (ML ensemble) with V6 weighted 70%"
    }


@api_router.post("/analyze-unified/{event_id}")
async def analyze_event_unified(event_id: str, sport_key: str = "basketball_nba"):
    """
    Analyze event using UNIFIED predictor (V5 + V6 combined)
    
    - Runs both V5 (line movement) and V6 (ML ensemble)
    - Combines results with V6 weighted 70%, V5 weighted 30%
    - Agreement bonus when both algorithms align
    - Single pick recommendation (no conflicts)
    """
    
    # Fetch event data
    events = await fetch_espn_events_with_odds(sport_key, days_ahead=3)
    event = next((e for e in events if e.get("id") == event_id or e.get("espn_id") == event_id), None)
    
    if not event:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found for sport {sport_key}")
    
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    
    logger.info(f"🔄 UNIFIED Manual Analysis: {home_team} vs {away_team}")
    
    # Get comprehensive data
    matchup_data = await get_comprehensive_matchup_data(event, sport_key)
    
    squad_data = {
        "home_team": {"injuries": []},
        "away_team": {"injuries": []}
    }
    
    try:
        home_roster = await fetch_team_roster(event.get("home_team_id", ""), sport_key)
        away_roster = await fetch_team_roster(event.get("away_team_id", ""), sport_key)
        squad_data["home_team"]["injuries"] = home_roster.get("injuries", [])
        squad_data["away_team"]["injuries"] = away_roster.get("injuries", [])
    except Exception as e:
        logger.warning(f"Could not fetch roster data: {e}")
    
    # Get line movement history
    line_movement_history = []
    opening_odds = {}
    
    try:
        history_docs = await db.odds_history.find({"event_id": event_id}).sort("timestamp", 1).to_list(1000)
        line_movement_history = history_docs
        
        if history_docs:
            opening_odds = history_docs[0]
        
        opening_doc = await db.opening_odds.find_one({"event_id": event_id})
        if opening_doc:
            opening_odds = opening_doc
    except Exception as e:
        logger.warning(f"Could not fetch line movement history: {e}")
    
    current_odds = event.get("odds", {})
    
    # Run unified analysis
    try:
        prediction = await generate_unified_prediction(
            event,
            sport_key,
            squad_data,
            matchup_data,
            line_movement_history,
            opening_odds,
            current_odds
        )
        
        # Save prediction if it has a pick
        if prediction.get("has_pick"):
            prediction_doc = {
                "id": str(uuid.uuid4()),
                "event_id": event_id,
                "sport_key": sport_key,
                "home_team": home_team,
                "away_team": away_team,
                "prediction": prediction.get("pick"),
                "confidence": prediction.get("confidence", 0) / 100,
                "odds": prediction.get("odds", 1.91),
                "ai_model": "unified",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "commence_time": event.get("commence_time", ""),
                "prediction_type": prediction.get("pick_type", "moneyline"),
                "result": "pending",
                "reasoning": prediction.get("reasoning", ""),
                "edge": prediction.get("edge", 0),
                "consensus_level": prediction.get("consensus_level", "unknown")
            }
            
            await db.predictions.insert_one(prediction_doc)
            logger.info(f"✅ Unified prediction saved: {prediction.get('pick')} at {prediction.get('confidence')}% confidence")
        
        return {
            "event": {
                "id": event_id,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": event.get("commence_time", ""),
                "sport_key": sport_key
            },
            "prediction": prediction,
            "data_summary": {
                "line_movement_snapshots": len(line_movement_history),
                "has_opening_odds": bool(opening_odds),
                "squad_data_available": bool(squad_data.get("home_team", {}).get("injuries") or squad_data.get("away_team", {}).get("injuries")),
                "matchup_data_available": bool(matchup_data)
            }
        }
    
    except Exception as e:
        logger.error(f"Error in unified analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unified analysis failed: {str(e)}")


# View upcoming games in prediction window
@api_router.get("/upcoming-predictions-window")
async def get_upcoming_predictions_window(sport_key: str = "basketball_nba"):
    """Get games in prediction window (45-75 min before start)"""
    try:
        now = datetime.now(timezone.utc)
        window_start = now + timedelta(minutes=45)
        window_end = now + timedelta(minutes=75)
        
        events = await fetch_espn_events_with_odds(sport_key, days_ahead=1)
        
        games_in_window = []
        for event in events:
            try:
                commence_str = event.get("commence_time", "")
                if not commence_str:
                    continue
                commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                if window_start <= commence_time <= window_end:
                    games_in_window.append(event)
            except:
                continue
        
        return {"games": games_in_window, "count": len(games_in_window)}
    except Exception as e:
        logger.error(f"Error fetching prediction window: {e}")
        return {"games": [], "count": 0}


# ==================== MY BETS TRACKING ====================
# User's actual bet performance tracking

@api_router.get("/my-bets")
async def get_my_bets():
    """Get user's tracked bets"""
    try:
        bets = await db.my_bets.find({}, {"_id": 0}).sort("created_at", -1).to_list(1000)
        return {"bets": bets}
    except Exception as e:
        logger.error(f"Error fetching my bets: {e}")
        return {"bets": []}


@api_router.post("/my-bets")
async def add_my_bet(bet: dict):
    """Add a new bet to track"""
    try:
        bet_doc = {
            "id": str(uuid.uuid4()),
            "event_name": bet.get("event_name"),
            "selection": bet.get("selection"),
            "stake": bet.get("stake"),
            "odds": bet.get("odds"),
            "result": bet.get("result", "pending"),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.my_bets.insert_one(bet_doc)
        # Remove _id before returning (MongoDB adds it after insert)
        bet_doc.pop("_id", None)
        return {"success": True, "bet": bet_doc}
    except Exception as e:
        logger.error(f"Error adding bet: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.put("/my-bets/{bet_id}")
async def update_my_bet(bet_id: str, update_data: dict):
    """Update a bet's result"""
    try:
        await db.my_bets.update_one(
            {"id": bet_id},
            {"$set": update_data}
        )
        return {"success": True}
    except Exception as e:
        logger.error(f"Error updating bet: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/my-bets/{bet_id}")
async def delete_my_bet(bet_id: str):
    """Delete a tracked bet"""
    try:
        await db.my_bets.delete_one({"id": bet_id})
        return {"success": True}
    except Exception as e:
        logger.error(f"Error deleting bet: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# View upcoming games in prediction window
@api_router.get("/upcoming-predictions-window")
async def get_upcoming_prediction_window():
    """View games that are in the 1-hour prediction window (45-75 min before start)"""
    now = datetime.now(timezone.utc)
    window_start = now + timedelta(minutes=45)  # 45 min from now
    window_end = now + timedelta(minutes=75)    # 75 min from now (1 hour ± 15 min)
    
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
                    
                    # Check if already has unified prediction
                    has_prediction = await db.predictions.find_one({
                        "event_id": event.get("id"),
                        "ai_model": "unified"
                    }) is not None
                    
                    game_info = {
                        "event_id": event.get("id"),
                        "sport": sport_key,
                        "home_team": event.get("home_team"),
                        "away_team": event.get("away_team"),
                        "commence_time": commence_str,
                        "minutes_to_start": round(time_to_start),
                        "has_prediction": has_prediction
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
        "message": "Games in prediction window will be automatically analyzed by Unified Predictor (V5+V6)"
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
    """
    Store odds snapshot for line movement tracking - ALL markets (ML, Spread, Totals)
    Stores opening odds and snapshots every 5 minutes
    """
    event_id = event.get("id")
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    # Round to 5-minute intervals for grouping snapshots (e.g., 10:00, 10:05, 10:10)
    minute_bucket = (now.minute // 5) * 5
    time_key = now.strftime(f"%Y-%m-%d-%H-{minute_bucket:02d}")
    
    home_team = event.get("home_team", "home")
    away_team = event.get("away_team", "away")
    commence_time = event.get("commence_time")  # Store commence time for cleanup
    
    # Check if we have opening odds stored for this event
    existing_opening = await db.opening_odds.find_one({"event_id": event_id})
    
    # Check if we already have a snapshot for this 5-minute window (to avoid duplicates)
    existing_snapshot = await db.odds_history.find_one({
        "event_id": event_id,
        "time_key": time_key
    })
    
    # Collect ALL market odds for this snapshot
    all_home_ml = []
    all_away_ml = []
    all_spreads = []  # Home spread
    all_spread_odds = []
    all_totals = []  # Total points line
    all_over_odds = []
    all_under_odds = []
    bookmaker_snapshots = []
    
    for bookmaker in event.get("bookmakers", []):
        bm_key = bookmaker.get("key")
        bm_title = bookmaker.get("title", bm_key)
        
        bm_snapshot = {
            "bookmaker": bm_key,
            "bookmaker_title": bm_title
        }
        
        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "h2h")
            outcomes = market.get("outcomes", [])
            
            if market_key == "h2h":  # Moneyline
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
                    all_home_ml.append(home_odds)
                    all_away_ml.append(away_odds)
                    bm_snapshot["home_ml"] = home_odds
                    bm_snapshot["away_ml"] = away_odds
                    
            elif market_key == "spreads":  # Point Spread
                for outcome in outcomes:
                    name = outcome.get("name", "").lower()
                    price = outcome.get("price")
                    point = outcome.get("point", 0)
                    
                    if name == "home" or home_team.lower() in name:
                        all_spreads.append(point)
                        all_spread_odds.append(price)
                        bm_snapshot["home_spread"] = point
                        bm_snapshot["spread_odds"] = price
                        
            elif market_key == "totals":  # Over/Under
                for outcome in outcomes:
                    name = outcome.get("name", "").lower()
                    price = outcome.get("price")
                    point = outcome.get("point", 0)
                    
                    if "over" in name:
                        all_totals.append(point)
                        all_over_odds.append(price)
                        bm_snapshot["total_line"] = point
                        bm_snapshot["over_odds"] = price
                    elif "under" in name:
                        all_under_odds.append(price)
                        bm_snapshot["under_odds"] = price
        
        if bm_snapshot.get("home_ml") or bm_snapshot.get("home_spread") or bm_snapshot.get("total_line"):
            bookmaker_snapshots.append(bm_snapshot)
    
    # Calculate averages across all bookmakers
    avg_home_ml = sum(all_home_ml) / len(all_home_ml) if all_home_ml else None
    avg_away_ml = sum(all_away_ml) / len(all_away_ml) if all_away_ml else None
    avg_spread = sum(all_spreads) / len(all_spreads) if all_spreads else None
    avg_total = sum(all_totals) / len(all_totals) if all_totals else None
    avg_over_odds = sum(all_over_odds) / len(all_over_odds) if all_over_odds else None
    avg_under_odds = sum(all_under_odds) / len(all_under_odds) if all_under_odds else None
    
    # Store opening odds if this is first time seeing this event
    if not existing_opening and (avg_home_ml or avg_spread or avg_total):
        opening_data = {
            "event_id": event_id,
            "home_team": home_team,
            "away_team": away_team,
            "timestamp": timestamp,
            "commence_time": commence_time,
            "bookmakers": bookmaker_snapshots
        }
        
        # ML
        if avg_home_ml and avg_away_ml:
            opening_data["home_odds"] = round(avg_home_ml, 2)
            opening_data["away_odds"] = round(avg_away_ml, 2)
        
        # Spread
        if avg_spread is not None:
            opening_data["spread"] = round(avg_spread, 1)
            opening_data["spread_odds"] = round(sum(all_spread_odds) / len(all_spread_odds), 2) if all_spread_odds else 1.91
        
        # Totals
        if avg_total is not None:
            opening_data["total"] = round(avg_total, 1)
            opening_data["over_odds"] = round(avg_over_odds, 2) if avg_over_odds else 1.91
            opening_data["under_odds"] = round(avg_under_odds, 2) if avg_under_odds else 1.91
        
        await db.opening_odds.insert_one(opening_data)
        logger.info(f"Stored opening odds for {home_team} vs {away_team}: ML={avg_home_ml:.2f}/{avg_away_ml:.2f}, Spread={avg_spread}, Total={avg_total}")
    
    # Store 5-minute snapshot for line movement tracking (avoid duplicates within same 5-min window)
    if not existing_snapshot and (avg_home_ml or avg_spread or avg_total):
        snapshot = {
            "event_id": event_id,
            "home_team": home_team,
            "away_team": away_team,
            "timestamp": timestamp,
            "time_key": time_key,
            "commence_time": commence_time,
            "num_bookmakers": len(bookmaker_snapshots),
            "bookmakers": bookmaker_snapshots
        }
        
        # ML
        if avg_home_ml and avg_away_ml:
            snapshot["home_odds"] = round(avg_home_ml, 2)
            snapshot["away_odds"] = round(avg_away_ml, 2)
        
        # Spread
        if avg_spread is not None:
            snapshot["spread"] = round(avg_spread, 1)
            snapshot["spread_odds"] = round(sum(all_spread_odds) / len(all_spread_odds), 2) if all_spread_odds else 1.91
        
        # Totals
        if avg_total is not None:
            snapshot["total"] = round(avg_total, 1)
            snapshot["over_odds"] = round(avg_over_odds, 2) if avg_over_odds else 1.91
            snapshot["under_odds"] = round(avg_under_odds, 2) if avg_under_odds else 1.91
        
        await db.odds_history.insert_one(snapshot)
        logger.debug(f"Stored 5-min snapshot for {home_team} vs {away_team}: ML={avg_home_ml:.2f}/{avg_away_ml:.2f}, Spread={avg_spread}, Total={avg_total}")

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

# ====================================================================================
# NOTE: Standalone V5 and V6 schedulers have been REMOVED (they were not running)
# 
# The UNIFIED PREDICTOR below is the ONLY production prediction system.
# It combines:
#   - V5 (30% weight): Line movement analysis from betpredictor_v5.py
#   - V6 (70% weight): ML ensemble from betpredictor_v6.py
#
# V5 and V6 can still be accessed directly via their API endpoints:
#   - /api/analyze-v5/{event_id} - Manual V5 analysis
#   - /api/analyze-v6/{event_id} - Manual V6 analysis  
#   - /api/analyze-unified/{event_id} - Manual unified analysis
#   - /api/predictions/v5 - List V5 predictions
#   - /api/predictions/v6 - List V6 predictions
#   - /api/predictions/unified - List unified predictions
# ====================================================================================


# UNIFIED PREDICTOR - Combines V5 + V6 into single prediction (V6 weighted 70%)
async def scheduled_unified_predictor():
    """
    UNIFIED PREDICTOR: Combines V5 (line movement) + V6 (ML ensemble) into single prediction.
    
    Weighting:
    - V6: 70% (ML ensemble is primary decision maker)
    - V5: 30% (Line movement provides confirmation)
    
    Benefits:
    - Single prediction per game (no conflicts)
    - Best of both algorithms
    - Agreement bonus when both align
    - V6-driven with V5 validation
    """
    # Wait 2 minutes on startup
    await asyncio.sleep(120)
    
    logger.info("🔄 Started UNIFIED PREDICTOR - Combining V5 + V6 (V6 weighted 70%)")
    
    sports_to_analyze = ["basketball_nba", "americanfootball_nfl", "icehockey_nhl", "soccer_epl"]
    
    while True:
        try:
            now = datetime.now(timezone.utc)
            # 1 hour window (45 min to 75 min before game)
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
                                home_team = event.get("home_team")
                                away_team = event.get("away_team")
                                
                                # Check if we already have a unified prediction for this event
                                existing = await db.predictions.find_one({
                                    "event_id": event_id, 
                                    "result": "pending",
                                    "ai_model": "unified"
                                })
                                
                                if existing:
                                    logger.debug(f"Skipping {event_id} - already has unified prediction")
                                    continue
                                
                                # This game is ~1 hour away - time for unified analysis!
                                logger.info(f"🔄 UNIFIED ANALYSIS: {home_team} vs {away_team} "
                                          f"(starts in {(commence_time - now).total_seconds() / 60:.0f} min)")
                                
                                # 1. Get comprehensive matchup data
                                matchup_data = await get_comprehensive_matchup_data(event, sport_key)
                                
                                # 2. Get squad data (injuries and rosters)
                                squad_data = {
                                    "home_team": {"injuries": [], "roster": [], "key_players": []},
                                    "away_team": {"injuries": [], "roster": [], "key_players": []}
                                }
                                try:
                                    home_roster = await fetch_team_roster(home_team, sport_key)
                                    away_roster = await fetch_team_roster(away_team, sport_key)
                                    squad_data["home_team"]["injuries"] = home_roster.get("injuries", [])
                                    squad_data["home_team"]["roster"] = home_roster.get("players", [])
                                    squad_data["home_team"]["key_players"] = home_roster.get("key_players", [])
                                    squad_data["away_team"]["injuries"] = away_roster.get("injuries", [])
                                    squad_data["away_team"]["roster"] = away_roster.get("players", [])
                                    squad_data["away_team"]["key_players"] = away_roster.get("key_players", [])
                                    logger.info(f"Fetched rosters: {home_team} ({len(home_roster.get('players', []))} players, "
                                              f"{len(home_roster.get('injuries', []))} injuries), "
                                              f"{away_team} ({len(away_roster.get('players', []))} players, "
                                              f"{len(away_roster.get('injuries', []))} injuries)")
                                except Exception as e:
                                    logger.warning(f"Could not fetch roster data: {e}")
                                
                                # 3. Get line movement history
                                line_history = await db.odds_history.find(
                                    {"event_id": event_id}, {"_id": 0}
                                ).sort("timestamp", 1).to_list(500)
                                
                                # 4. Get opening odds
                                opening_odds = await db.opening_odds.find_one(
                                    {"event_id": event_id}, {"_id": 0}
                                ) or {}
                                
                                # 5. Get current odds
                                current_odds = event.get("odds", {})
                                
                                # 6. Run UNIFIED prediction (combines V5 + V6)
                                unified_prediction = await generate_unified_prediction(
                                    event=event,
                                    sport_key=sport_key,
                                    squad_data=squad_data,
                                    matchup_data=matchup_data,
                                    line_movement_history=line_history,
                                    opening_odds=opening_odds,
                                    current_odds=current_odds
                                )
                                
                                if unified_prediction and unified_prediction.get("has_pick"):
                                    confidence = unified_prediction.get("confidence", 0) / 100
                                    edge = unified_prediction.get("edge", 0) / 100
                                    consensus = unified_prediction.get("consensus_level", "unknown")
                                    
                                    if confidence >= 0.60:  # Unified uses 60% threshold
                                        # Create prediction from unified result
                                        prediction = PredictionCreate(
                                            event_id=event_id,
                                            sport_key=sport_key,
                                            home_team=home_team,
                                            away_team=away_team,
                                            commence_time=commence_str,
                                            prediction_type=unified_prediction.get("pick_type", "moneyline"),
                                            predicted_outcome=unified_prediction.get("pick_display", unified_prediction.get("pick", "")),
                                            confidence=confidence,
                                            analysis=unified_prediction.get("reasoning", ""),
                                            ai_model="unified",
                                            odds_at_prediction=unified_prediction.get("odds", 1.91)
                                        )
                                        
                                        await create_recommendation(prediction)
                                        predictions_made += 1
                                        
                                        # Send notification for new pick
                                        await create_notification(
                                            "new_pick",
                                            f"🎯 New Pick: {home_team} vs {away_team}",
                                            f"{unified_prediction.get('pick_display', unified_prediction.get('pick'))} "
                                            f"({unified_prediction.get('pick_type')}) @ {confidence*100:.0f}% confidence",
                                            {
                                                "event_id": event_id,
                                                "pick": unified_prediction.get('pick_display', unified_prediction.get('pick')),
                                                "pick_type": unified_prediction.get('pick_type'),
                                                "confidence": confidence,
                                                "odds": unified_prediction.get("odds", 1.91)
                                            }
                                        )
                                        
                                        logger.info(f"✅ UNIFIED PICK: {home_team} vs {away_team} - "
                                                  f"{unified_prediction.get('pick_type')}: {unified_prediction.get('pick_display', unified_prediction.get('pick'))} "
                                                  f"@ {confidence*100:.0f}% conf, edge: {edge*100:.1f}%, "
                                                  f"consensus: {consensus}")
                                    else:
                                        logger.info(f"⏭️ LOW CONFIDENCE: {home_team} vs {away_team} - "
                                                  f"{confidence*100:.0f}% < 60% threshold")
                                else:
                                    reason = unified_prediction.get("reasoning", "No consensus") if unified_prediction else "Analysis failed"
                                    logger.info(f"⏭️ NO PICK: {home_team} vs {away_team} - {reason[:150]}")
                                
                                # Store odds snapshot
                                await store_odds_snapshot(event)
                                
                                # Small delay between analyses
                                await asyncio.sleep(3)
                                
                        except Exception as e:
                            logger.error(f"Error in unified analysis for event: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    
                except Exception as e:
                    logger.error(f"Error in unified predictor for {sport_key}: {e}")
            
            if predictions_made > 0:
                logger.info(f"🔄 Unified predictor complete - {predictions_made} predictions created")
            
            # Run every 10 minutes to catch all games in the 1 hour window
            await asyncio.sleep(600)
            
        except Exception as e:
            logger.error(f"Scheduled unified predictor error: {e}")
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
            deleted_history_count = 0
            deleted_opening_count = 0
            
            # Find events that have started and delete their line movement data
            event_ids = await db.odds_history.distinct("event_id")
            
            for event_id in event_ids:
                # Get commence time from opening_odds (now stored directly)
                opening = await db.opening_odds.find_one({"event_id": event_id}, {"_id": 0})
                
                commence_str = None
                if opening:
                    commence_str = opening.get("commence_time")
                
                # Fallback: check odds_history
                if not commence_str:
                    history_record = await db.odds_history.find_one({"event_id": event_id})
                    if history_record:
                        commence_str = history_record.get("commence_time")
                
                # Fallback: check predictions
                if not commence_str:
                    prediction = await db.predictions.find_one({"event_id": event_id}, {"commence_time": 1})
                    if prediction:
                        commence_str = prediction.get("commence_time", "")
                
                if commence_str:
                    try:
                        commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                        if commence_time <= now:
                            # Delete odds history
                            result = await db.odds_history.delete_many({"event_id": event_id})
                            deleted_history_count += result.deleted_count
                            
                            # Delete opening odds
                            opening_result = await db.opening_odds.delete_one({"event_id": event_id})
                            deleted_opening_count += opening_result.deleted_count
                    except Exception:
                        pass
            
            if deleted_history_count > 0 or deleted_opening_count > 0:
                logger.info(f"Cleaned up {deleted_history_count} history records, {deleted_opening_count} opening odds for started events")
            
            await asyncio.sleep(1800)  # Run every 30 minutes
            
        except Exception as e:
            logger.error(f"Line movement cleanup error: {e}")
            await asyncio.sleep(300)

# Background task for daily summary notification
async def scheduled_daily_summary():
    """Background task that sends daily summary notification at 9 PM UTC"""
    # Wait 1 hour on startup to avoid sending summary immediately
    await asyncio.sleep(3600)
    
    while True:
        try:
            now = datetime.now(timezone.utc)
            
            # Calculate time until 9 PM UTC
            target_hour = 21  # 9 PM UTC
            if now.hour >= target_hour:
                # Already past 9 PM, wait until tomorrow 9 PM
                tomorrow = now + timedelta(days=1)
                target_time = tomorrow.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            else:
                # Wait until today 9 PM
                target_time = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            
            wait_seconds = (target_time - now).total_seconds()
            
            if wait_seconds > 0:
                logger.info(f"📊 Daily summary scheduled for {target_time.isoformat()} (in {wait_seconds/3600:.1f} hours)")
                await asyncio.sleep(wait_seconds)
            
            # Check if daily summary is enabled
            settings = await db.settings.find_one({}, {"_id": 0})
            if not settings or settings.get('notification_preferences', {}).get('daily_summary', True):
                await send_daily_summary_notification()
            
            # Wait a bit before next cycle to ensure we're past target time
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Daily summary error: {e}")
            await asyncio.sleep(3600)  # Retry in 1 hour on error

async def send_daily_summary_notification():
    """Generate and send the daily summary notification"""
    try:
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get today's predictions
        predictions = await db.predictions.find({
            "created_at": {"$gte": today_start.isoformat()}
        }).to_list(100)
        
        # Get today's results
        completed = [p for p in predictions if p.get("result") in ["win", "loss", "push"]]
        wins = len([p for p in completed if p.get("result") == "win"])
        losses = len([p for p in completed if p.get("result") == "loss"])
        pending = len([p for p in predictions if p.get("result") == "pending"])
        
        # Calculate profit
        total_profit = 0
        for p in completed:
            odds = p.get("odds_at_prediction", 1.91)
            if p.get("result") == "win":
                total_profit += 100 * (odds - 1)
            elif p.get("result") == "loss":
                total_profit -= 100
        
        # Create summary message
        if len(predictions) > 0:
            win_rate = (wins / len(completed) * 100) if len(completed) > 0 else 0
            message = f"Today's Results: {wins}W-{losses}L ({win_rate:.0f}% win rate)\n"
            message += f"Profit/Loss: ${total_profit:+.2f}\n"
            message += f"Pending picks: {pending}"
        else:
            message = "No picks were generated today. Check back tomorrow!"
        
        result = await create_notification(
            "daily_summary",
            "📊 Daily Performance Summary",
            message,
            {
                "date": now.date().isoformat(),
                "wins": wins,
                "losses": losses,
                "pending": pending,
                "profit": total_profit
            }
        )
        
        if result:
            logger.info(f"Daily summary sent: {wins}W-{losses}L, ${total_profit:+.2f}")
            return {"wins": wins, "losses": losses, "profit": total_profit}
        else:
            logger.debug("Daily summary skipped (duplicate)")
            return None
        
    except Exception as e:
        logger.error(f"Error creating daily summary: {e}")
        return None

@api_router.post("/notifications/daily-summary")
async def trigger_daily_summary():
    """Manually trigger a daily summary notification"""
    result = await send_daily_summary_notification()
    if result:
        return {"message": "Daily summary notification created", "created": True, "data": result}
    else:
        return {"message": "Daily summary notification already exists (within 5 min window)", "created": False, "data": None}

@api_router.post("/notifications/result-test")
async def trigger_result_notification():
    """Manually trigger a result notification for testing"""
    # Get a random completed prediction
    prediction = await db.predictions.find_one({"result": {"$in": ["win", "loss"]}})
    
    if prediction:
        result = prediction.get("result", "win")
        await create_notification(
            "result",
            f"🎯 Bet Result: {result.upper()}",
            f"{prediction.get('home_team')} vs {prediction.get('away_team')} - "
            f"Your pick: {prediction.get('predicted_outcome')} - Result: {result.upper()}!",
            {
                "prediction_id": prediction.get("id"),
                "result": result,
                "pick": prediction.get("predicted_outcome")
            }
        )
        return {"message": f"Result notification created for {result}"}
    else:
        # Create a sample result notification
        await create_notification(
            "result",
            "🎯 Bet Result: WIN",
            "Boston Celtics vs Portland Trail Blazers - Your pick: Boston Celtics -12.5 - Result: WIN! +$91.00",
            {
                "result": "win",
                "pick": "Boston Celtics -12.5",
                "profit": 91.00
            }
        )
        return {"message": "Sample result notification created"}

# Background task for ESPN odds refresh - EVERY 5 MINUTES for accurate line tracking
async def scheduled_espn_odds_refresh():
    """Background task that refreshes ESPN odds every 5 MINUTES for pre-match events and line movement tracking"""
    global last_scrape_time
    
    # Run IMMEDIATELY on startup to get first snapshot
    logger.info("📸 Running INITIAL ESPN odds snapshot on startup...")
    
    sports_to_refresh = ["basketball_nba", "americanfootball_nfl", "icehockey_nhl", "soccer_epl"]
    
    # Initial run immediately
    try:
        now = datetime.now(timezone.utc)
        for sport_key in sports_to_refresh:
            events = await fetch_espn_events_with_odds(sport_key, days_ahead=3)
            if events:
                prematch_count = 0
                for event in events:
                    try:
                        commence_str = event.get("commence_time", "")
                        if commence_str:
                            commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                            # Only store for games MORE than 1 hour away (stop at 1 hour before)
                            if commence_time > now + timedelta(hours=1):
                                await store_odds_snapshot(event)
                                prematch_count += 1
                    except Exception:
                        pass
                cache_key = f"{sport_key}_h2h,spreads,totals"
                events_cache[cache_key] = (events, now)
                last_scrape_time = now.isoformat()
                logger.info(f"Initial snapshot: {prematch_count} pre-match events for {sport_key}")
    except Exception as e:
        logger.error(f"Initial ESPN snapshot error: {e}")
    
    while True:
        try:
            # Wait 5 MINUTES between snapshots
            await asyncio.sleep(300)  # 5 minutes = 300 seconds
            
            logger.info("📸 Running ESPN odds snapshot (every 5 min)...")
            now = datetime.now(timezone.utc)
            total_snapshots = 0
            
            for sport_key in sports_to_refresh:
                try:
                    events = await fetch_espn_events_with_odds(sport_key, days_ahead=3)
                    
                    if events:
                        # Only store odds snapshots for PRE-MATCH events MORE than 1 hour away
                        prematch_count = 0
                        for event in events:
                            try:
                                commence_str = event.get("commence_time", "")
                                if commence_str:
                                    commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                                    # Stop tracking at 1 hour before game (prediction time)
                                    if commence_time > now + timedelta(hours=1):
                                        await store_odds_snapshot(event)
                                        prematch_count += 1
                            except Exception:
                                pass
                        
                        total_snapshots += prematch_count
                        
                        # Update cache
                        cache_key = f"{sport_key}_h2h,spreads,totals"
                        events_cache[cache_key] = (events, now)
                        last_scrape_time = now.isoformat()
                    
                    await asyncio.sleep(2)  # Small delay between sports
                    
                except Exception as e:
                    logger.error(f"Error refreshing ESPN odds for {sport_key}: {e}")
            
            logger.info(f"📸 ESPN snapshot complete: {total_snapshots} events tracked")
            
        except Exception as e:
            logger.error(f"Scheduled ESPN refresh error: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup"""
    global adaptive_learning
    
    # Initialize Adaptive Learning System
    logger.info("🧠 Initializing Adaptive Learning System...")
    adaptive_learning = await create_adaptive_learning_system(db)
    logger.info("✅ Adaptive Learning System ready - models will now learn from results!")
    
    # Run immediate cleanup for completed/started events on startup
    logger.info("🧹 Running startup cleanup for completed events...")
    await startup_cleanup_completed_events()
    
    # Start the scheduled result checker - INSTANT 10 SECOND sync for live games
    asyncio.create_task(scheduled_result_checker())
    logger.info("Started INSTANT live score sync - runs every 10 SECONDS with ESPN API")
    
    # Start line movement checker - updates recommendations on significant line moves
    asyncio.create_task(scheduled_line_movement_checker())
    logger.info("Started line movement checker - monitors for significant line moves")
    
    # START UNIFIED PREDICTOR - Combines V5 (line movement) + V6 (ML ensemble) with V6 weighted 70%
    asyncio.create_task(scheduled_unified_predictor())
    logger.info("🔄 Started UNIFIED PREDICTOR - V5 + V6 combined (V6 weighted 70%)")
    
    # Note: Individual V5 and V6 schedulers disabled in favor of unified approach
    # Endpoints still available for manual analysis: /api/analyze-v5 and /api/analyze-v6
    
    # Start ESPN odds refresh - EVERY 5 MINUTES for accurate line tracking
    asyncio.create_task(scheduled_espn_odds_refresh())
    logger.info("📸 Started ESPN odds snapshots - runs every 5 MINUTES for line movement tracking")
    
    # Start line movement data cleanup
    asyncio.create_task(scheduled_line_movement_cleanup())
    logger.info("Started line movement cleanup - runs every 30 minutes (deletes data for started events)")
    
    # Start daily summary scheduler
    asyncio.create_task(scheduled_daily_summary())
    logger.info("📊 Started daily summary scheduler - sends daily performance recap")


async def startup_cleanup_completed_events():
    """Clean up line movement data for events that have already started or finished"""
    try:
        now = datetime.now(timezone.utc)
        deleted_history = 0
        deleted_opening = 0
        
        # Get all event IDs that have line movement data
        event_ids = await db.odds_history.distinct("event_id")
        
        for event_id in event_ids:
            # Check if event has started
            opening = await db.opening_odds.find_one({"event_id": event_id})
            commence_str = opening.get("commence_time") if opening else None
            
            if not commence_str:
                # Try to get from odds_history
                history_record = await db.odds_history.find_one({"event_id": event_id})
                commence_str = history_record.get("commence_time") if history_record else None
            
            if commence_str:
                try:
                    commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                    
                    # If event has already started, delete its line movement data
                    if commence_time <= now:
                        result = await db.odds_history.delete_many({"event_id": event_id})
                        deleted_history += result.deleted_count
                        
                        opening_result = await db.opening_odds.delete_one({"event_id": event_id})
                        deleted_opening += opening_result.deleted_count
                except Exception:
                    pass
        
        if deleted_history > 0 or deleted_opening > 0:
            logger.info(f"🧹 Startup cleanup: Deleted {deleted_history} snapshots, {deleted_opening} opening odds for started events")
        else:
            logger.info("🧹 Startup cleanup: No stale line movement data to clean")
            
    except Exception as e:
        logger.error(f"Startup cleanup error: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
