#!/usr/bin/env python3
"""
Create sample picks with correct spread format
"""

import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
import uuid

BACKEND_URL = "http://localhost:8001/api"

# Sample picks with correct format
SAMPLE_PICKS = [
    # SPREAD picks with exact spread value
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Boston Celtics",
        "away_team": "Portland Trail Blazers",
        "commence_time": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
        "prediction_type": "spread",
        "predicted_outcome": "Boston Celtics -12.5",  # Correct format: Team + spread
        "confidence": 0.75,
        "analysis": "V6 Analysis: 4/5 models agree. Boston covering -12.5 with strong home court advantage.",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.91
    },
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Denver Nuggets",
        "away_team": "San Antonio Spurs",
        "commence_time": (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat(),
        "prediction_type": "spread",
        "predicted_outcome": "Denver Nuggets -14.5",  # Correct format
        "confidence": 0.78,
        "analysis": "V6 Analysis: Well-rested Nuggets vs fatigued Spurs. Line moved from -11.5 to -14.5.",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.91
    },
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Oklahoma City Thunder",
        "away_team": "Utah Jazz",
        "commence_time": (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat(),
        "prediction_type": "spread",
        "predicted_outcome": "Oklahoma City Thunder -15.5",  # Correct format
        "confidence": 0.72,
        "analysis": "V6 Analysis: OKC at home against struggling Jazz. Sharp money on Thunder spread.",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.91
    },
    # MONEYLINE pick - just team name
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Cleveland Cavaliers",
        "away_team": "Washington Wizards",
        "commence_time": (datetime.now(timezone.utc) + timedelta(hours=2.5)).isoformat(),
        "prediction_type": "moneyline",
        "predicted_outcome": "Cleveland Cavaliers",  # ML is just team name
        "confidence": 0.82,
        "analysis": "V6 Analysis: Heavy favorite at home. Elite team vs worst team in league.",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.15
    },
    # TOTAL pick - Over/Under format
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Golden State Warriors",
        "away_team": "Charlotte Hornets",
        "commence_time": (datetime.now(timezone.utc) + timedelta(hours=3.5)).isoformat(),
        "prediction_type": "total",
        "predicted_outcome": "Over 225.5",  # Total is Over/Under + number
        "confidence": 0.73,
        "analysis": "V6 Analysis: Both teams playing fast. Line has moved up from 222.5.",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.91
    },
    # COMPLETED picks for history
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Miami Heat",
        "away_team": "New York Knicks",
        "commence_time": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        "prediction_type": "spread",
        "predicted_outcome": "Miami Heat -3.5",
        "confidence": 0.71,
        "analysis": "V6 Analysis: Miami covering at home vs struggling Knicks.",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.91,
        "result": "win"
    },
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Los Angeles Lakers",
        "away_team": "Phoenix Suns",
        "commence_time": (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(),
        "prediction_type": "moneyline",
        "predicted_outcome": "Los Angeles Lakers",
        "confidence": 0.70,
        "analysis": "V6 Analysis: Lakers at home with full roster.",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.65,
        "result": "win"
    },
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Milwaukee Bucks",
        "away_team": "Chicago Bulls",
        "commence_time": (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat(),
        "prediction_type": "spread",
        "predicted_outcome": "Milwaukee Bucks -8.5",
        "confidence": 0.74,
        "analysis": "V6 Analysis: Bucks dominant at home. Line value at -8.5.",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.91,
        "result": "win"
    },
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Philadelphia 76ers",
        "away_team": "Brooklyn Nets",
        "commence_time": (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat(),
        "prediction_type": "total",
        "predicted_outcome": "Under 218.5",
        "confidence": 0.72,
        "analysis": "V6 Analysis: Both teams playing slower pace this week.",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.91,
        "result": "loss"
    },
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Atlanta Hawks",
        "away_team": "Detroit Pistons",
        "commence_time": (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat(),
        "prediction_type": "spread",
        "predicted_outcome": "Atlanta Hawks -11.5",
        "confidence": 0.76,
        "analysis": "V6 Analysis: Hawks at home vs worst team in league.",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.91,
        "result": "win"
    }
]

async def create_picks():
    """Create sample picks"""
    print("Creating sample picks with correct format...\n")
    
    async with aiohttp.ClientSession() as session:
        for i, pick_data in enumerate(SAMPLE_PICKS):
            result = pick_data.pop("result", None)
            
            try:
                async with session.post(f"{BACKEND_URL}/recommendations", json=pick_data) as response:
                    if response.status == 200:
                        created = await response.json()
                        pred_id = created.get("id")
                        
                        print(f"✅ Created: {pick_data['predicted_outcome']} ({pick_data['prediction_type']})")
                        
                        # Update result if provided
                        if result:
                            update_data = {"prediction_id": pred_id, "result": result}
                            async with session.put(f"{BACKEND_URL}/result", json=update_data) as update_resp:
                                if update_resp.status == 200:
                                    print(f"   → Result: {result.upper()}")
                    else:
                        text = await response.text()
                        print(f"❌ Failed: {text[:50]}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Get performance summary
        print("\n" + "=" * 50)
        async with session.get(f"{BACKEND_URL}/performance") as response:
            if response.status == 200:
                perf = await response.json()
                print(f"Total: {perf.get('total', 0)} | Wins: {perf.get('wins', 0)} | Losses: {perf.get('losses', 0)}")
                print(f"Win Rate: {perf.get('win_rate', 0)}% | ROI: {perf.get('roi', 0)}%")

if __name__ == "__main__":
    asyncio.run(create_picks())
