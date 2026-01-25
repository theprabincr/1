from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
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

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# The Odds API config
ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '')
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Emergent LLM Key
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')

# Cache for mock events (to keep IDs consistent)
mock_events_cache = {}

# Events cache to reduce API calls (cache for 30 minutes)
events_cache = {}
CACHE_DURATION_MINUTES = 30

# API usage tracking
api_usage = {
    "requests_remaining": None,
    "requests_used": None,
    "last_updated": None,
    "monthly_limit": 500
}

# Active sportsbooks (only ones that return data from API)
SPORTSBOOKS = {
    'draftkings': 'draftkings',
    'fanduel': 'fanduel',
    'betmgm': 'betmgm',
    'pinnacle': 'pinnacle',
    'unibet': 'unibet',
    'betway': 'betway',
    'betonline': 'betonlineag'
}

SPORTSBOOK_NAMES = {
    'draftkings': 'DraftKings',
    'fanduel': 'FanDuel',
    'betmgm': 'BetMGM',
    'pinnacle': 'Pinnacle',
    'unibet': 'Unibet',
    'betway': 'Betway',
    'betonlineag': 'BetOnline'
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

# Helper function to fetch from Odds API
async def fetch_odds_api(endpoint: str, params: dict = None):
    if not ODDS_API_KEY:
        logger.warning("No ODDS_API_KEY configured")
        return None
    
    if params is None:
        params = {}
    params['apiKey'] = ODDS_API_KEY
    
    global api_usage
    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.get(f"{ODDS_API_BASE}{endpoint}", params=params, timeout=30.0)
            response.raise_for_status()
            
            # Track API usage from response headers
            requests_remaining = response.headers.get('x-requests-remaining')
            requests_used = response.headers.get('x-requests-used')
            if requests_remaining:
                api_usage['requests_remaining'] = int(requests_remaining)
            if requests_used:
                api_usage['requests_used'] = int(requests_used)
            api_usage['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Odds API error: {e}")
            return None

# Get API usage endpoint
@api_router.get("/api-usage")
async def get_api_usage():
    """Get current API usage statistics"""
    return api_usage

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
async def get_sports():
    """Get list of available sports"""
    data = await fetch_odds_api("/sports")
    if data is None:
        # Return mock data if API unavailable
        return [
            {"key": "americanfootball_nfl", "group": "American Football", "title": "NFL", "description": "US Football", "active": True, "has_outrights": False},
            {"key": "basketball_nba", "group": "Basketball", "title": "NBA", "description": "US Basketball", "active": True, "has_outrights": False},
            {"key": "baseball_mlb", "group": "Baseball", "title": "MLB", "description": "US Baseball", "active": True, "has_outrights": False},
            {"key": "icehockey_nhl", "group": "Ice Hockey", "title": "NHL", "description": "US Ice Hockey", "active": True, "has_outrights": False},
            {"key": "soccer_epl", "group": "Soccer", "title": "EPL", "description": "English Premier League", "active": True, "has_outrights": False},
            {"key": "soccer_spain_la_liga", "group": "Soccer", "title": "La Liga", "description": "Spanish La Liga", "active": True, "has_outrights": False},
            {"key": "mma_mixed_martial_arts", "group": "MMA", "title": "MMA", "description": "Mixed Martial Arts", "active": True, "has_outrights": False},
            {"key": "tennis_atp_french_open", "group": "Tennis", "title": "ATP French Open", "description": "Tennis", "active": True, "has_outrights": False},
        ]
    return [Sport(**s) for s in data if s.get('active', False)]

@api_router.get("/events/{sport_key}")
async def get_events(sport_key: str, markets: str = "h2h,spreads,totals"):
    """Get events with odds for a specific sport - only upcoming events"""
    bookmaker_keys = ",".join(SPORTSBOOKS.values())
    params = {
        "regions": "us,eu,uk,au",
        "markets": markets,
        "bookmakers": bookmaker_keys,
        "oddsFormat": "decimal"  # European format
    }
    
    data = await fetch_odds_api(f"/sports/{sport_key}/odds", params)
    
    if data is None:
        # Return mock data
        return await get_mock_events(sport_key)
    
    # Filter to only show upcoming events (not past)
    now = datetime.now(timezone.utc)
    upcoming_events = []
    for event in data:
        commence_time_str = event.get('commence_time', '')
        if commence_time_str:
            try:
                commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
                if commence_time > now:
                    upcoming_events.append(event)
                    # Store odds history for line movement tracking
                    await store_odds_snapshot(event)
            except:
                upcoming_events.append(event)
    
    return upcoming_events

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
    """Get line movement history for an event - up to 5 days"""
    # First check database for stored history
    history = await db.odds_history.find(
        {"event_id": event_id},
        {"_id": 0}
    ).sort("timestamp", -1).limit(500).to_list(500)
    
    # If we have history, return it
    if history:
        return history
    
    # Try to fetch historical odds from API (if available)
    if ODDS_API_KEY:
        try:
            # Fetch historical odds - The Odds API provides event odds endpoint
            params = {
                "apiKey": ODDS_API_KEY,
                "regions": "us,eu,uk,au",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "decimal"
            }
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(
                    f"{ODDS_API_BASE}/sports/{sport_key}/events/{event_id}/odds",
                    params=params,
                    timeout=30.0
                )
                if response.status_code == 200:
                    event_data = response.json()
                    # Store and return this data
                    await store_odds_snapshot(event_data)
                    return await db.odds_history.find(
                        {"event_id": event_id},
                        {"_id": 0}
                    ).sort("timestamp", -1).to_list(500)
        except Exception as e:
            logger.error(f"Error fetching historical odds: {e}")
    
    # Fallback to mock data
    return generate_mock_line_movement(event_id)

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
async def get_recommendations(sport_key: Optional[str] = None, limit: int = 10):
    """Get AI-generated bet recommendations"""
    query = {"result": "pending"}
    if sport_key:
        query["sport_key"] = sport_key
    
    predictions = await db.predictions.find(
        query,
        {"_id": 0}
    ).sort("created_at", -1).limit(limit).to_list(limit)
    
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
    """Generate a smart recommendation with AI analysis"""
    try:
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        bookmakers = odds_data.get("bookmakers", [])
        
        # Get best odds
        best_home = 1.0
        best_away = 1.0
        best_home_book = ""
        best_away_book = ""
        
        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == home_team and outcome.get("price", 1) > best_home:
                            best_home = outcome.get("price")
                            best_home_book = SPORTSBOOK_NAMES.get(bm.get("key"), bm.get("title"))
                        elif outcome.get("name") == away_team and outcome.get("price", 1) > best_away:
                            best_away = outcome.get("price")
                            best_away_book = SPORTSBOOK_NAMES.get(bm.get("key"), bm.get("title"))
        
        # Build prompt for AI
        prompt = f"""Analyze this {sport_key.replace('_', ' ')} matchup:
{home_team} (Home) @ {best_home:.2f} vs {away_team} (Away) @ {best_away:.2f}

Best odds: {home_team} @ {best_home:.2f} ({best_home_book}), {away_team} @ {best_away:.2f} ({best_away_book})

Provide your betting recommendation with pick, confidence (1-10), and brief reasoning."""

        # Get AI analysis
        analysis_text = await get_ai_analysis(prompt, "gpt-5.2")
        
        # Parse confidence from analysis (default to 6 if not found)
        confidence = 0.6
        predicted_team = home_team
        
        analysis_lower = analysis_text.lower()
        if "confidence:" in analysis_lower or "confidence" in analysis_lower:
            try:
                # Extract confidence number
                import re
                conf_match = re.search(r'confidence[:\s]*(\d+)', analysis_lower)
                if conf_match:
                    confidence = int(conf_match.group(1)) / 10
            except:
                pass
        
        # Determine predicted team from analysis
        if away_team.lower() in analysis_lower and "pick:" in analysis_lower:
            pick_section = analysis_lower.split("pick:")[1][:100] if "pick:" in analysis_lower else ""
            if away_team.lower() in pick_section:
                predicted_team = away_team
        
        # Determine best odds for predicted team
        odds_at_prediction = best_home if predicted_team == home_team else best_away
        
        # Skip if odds are unreasonable (likely futures/outrights)
        if odds_at_prediction < 1.01 or odds_at_prediction > 50:
            logger.info(f"Skipping {home_team} vs {away_team} - unusual odds: {odds_at_prediction}")
            return None
        
        # Create prediction
        prediction = PredictionCreate(
            event_id=event.get("id"),
            sport_key=sport_key,
            home_team=home_team,
            away_team=away_team,
            commence_time=event.get("commence_time"),
            prediction_type="moneyline",
            predicted_outcome=predicted_team,
            confidence=min(confidence, 0.95),  # Cap at 95%
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
            
            # Check for significant line movement (>5% change)
            if original_odds > 0:
                change_pct = abs(current_best - original_odds) / original_odds * 100
                
                if change_pct > 5:
                    # Line moved significantly - add note to analysis
                    new_analysis = prediction.get("analysis", "") + f"\n\n[LINE MOVEMENT UPDATE: Odds moved from {original_odds:.2f} to {current_best:.2f} ({change_pct:.1f}% change)]"
                    
                    # Adjust confidence based on movement direction
                    new_confidence = prediction.get("confidence", 0.6)
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
    """Fetch event result from The Odds API scores endpoint"""
    if not ODDS_API_KEY:
        return None
    
    try:
        params = {
            "apiKey": ODDS_API_KEY,
            "daysFrom": 3  # Look back 3 days for completed events
        }
        
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(
                f"{ODDS_API_BASE}/sports/{sport_key}/scores",
                params=params,
                timeout=30.0
            )
            
            if response.status_code == 200:
                scores_data = response.json()
                
                # Find the matching event
                for event in scores_data:
                    if event.get("id") == event_id or (
                        event.get("home_team") == home_team and 
                        event.get("away_team") == away_team
                    ):
                        if event.get("completed"):
                            scores = event.get("scores", [])
                            if scores and len(scores) >= 2:
                                home_score = None
                                away_score = None
                                
                                for score in scores:
                                    if score.get("name") == home_team:
                                        home_score = int(score.get("score", 0))
                                    elif score.get("name") == away_team:
                                        away_score = int(score.get("score", 0))
                                
                                if home_score is not None and away_score is not None:
                                    # Determine if prediction was correct
                                    if prediction_type == "moneyline":
                                        winner = home_team if home_score > away_score else away_team
                                        if home_score == away_score:
                                            return "push"
                                        return "win" if predicted_outcome == winner else "loss"
                                    
                                    # For now, mark as needs manual review
                                    return None
                        return None
            return None
    except Exception as e:
        logger.error(f"Error fetching scores: {e}")
        return None

@api_router.get("/scores/{sport_key}")
async def get_scores(sport_key: str, days_from: int = 3):
    """Get recent scores for completed events"""
    if not ODDS_API_KEY:
        return {"error": "API key not configured", "scores": []}
    
    try:
        params = {
            "apiKey": ODDS_API_KEY,
            "daysFrom": days_from
        }
        
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(
                f"{ODDS_API_BASE}/sports/{sport_key}/scores",
                params=params,
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            return {"error": f"API returned {response.status_code}", "scores": []}
    except Exception as e:
        logger.error(f"Error fetching scores: {e}")
        return {"error": str(e), "scores": []}

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
    """Store odds snapshot for line movement tracking"""
    event_id = event.get("id")
    timestamp = datetime.now(timezone.utc).isoformat()
    
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            snapshot = {
                "event_id": event_id,
                "timestamp": timestamp,
                "bookmaker": bookmaker.get("key"),
                "bookmaker_title": bookmaker.get("title"),
                "market": market.get("key"),
                "outcomes": market.get("outcomes", [])
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
                    name = outcome.get("name", "")
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
    from datetime import timedelta
    
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
    """Background task that runs every hour to check for completed events"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
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

# Background task for auto-generating recommendations
async def scheduled_recommendation_generator():
    """Background task that generates recommendations periodically"""
    # Wait 30 seconds on startup to let services initialize
    await asyncio.sleep(30)
    
    while True:
        try:
            logger.info("Running scheduled recommendation generation...")
            await auto_generate_recommendations()
            await asyncio.sleep(7200)  # Run every 2 hours
        except Exception as e:
            logger.error(f"Scheduled recommendation generator error: {e}")
            await asyncio.sleep(120)

@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup"""
    # Start the scheduled result checker
    asyncio.create_task(scheduled_result_checker())
    logger.info("Started background result checker - runs every hour")
    
    # Start line movement checker
    asyncio.create_task(scheduled_line_movement_checker())
    logger.info("Started line movement checker - runs every hour")
    
    # Start recommendation generator
    asyncio.create_task(scheduled_recommendation_generator())
    logger.info("Started recommendation generator - runs every 2 hours")
    logger.info("Started background result checker - will run every hour")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
