#!/usr/bin/env python3
"""
BetPredictor Pre-Game Window Simulation
Simulates the automatic prediction generation that happens 1 hour before games
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timezone, timedelta
import uuid

BACKEND_URL = "http://localhost:8001/api"

# Simulate games that would be in the pre-game window (45-75 min before tip-off)
PREGAME_WINDOW_GAMES = [
    {
        "id": f"PREGAME_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Boston Celtics",
        "away_team": "Portland Trail Blazers",
        "commence_time": (datetime.now(timezone.utc) + timedelta(minutes=60)).isoformat(),
        "prediction": {
            "pick": "Boston Celtics",
            "pick_type": "spread",
            "confidence": 0.72,
            "edge": 8.5,
            "odds": 1.91,
            "reasoning": "V6 Analysis: 4/5 models agree. Strong home team advantage, away team on B2B with key injuries."
        }
    },
    {
        "id": f"PREGAME_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Denver Nuggets",
        "away_team": "San Antonio Spurs",
        "commence_time": (datetime.now(timezone.utc) + timedelta(minutes=55)).isoformat(),
        "prediction": {
            "pick": "Denver Nuggets",
            "pick_type": "spread",
            "confidence": 0.75,
            "edge": 10.2,
            "odds": 1.91,
            "reasoning": "V6 Analysis: Well-rested home team vs fatigued away team. Sharp money moving toward Denver."
        }
    },
    {
        "id": f"PREGAME_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Oklahoma City Thunder",
        "away_team": "Utah Jazz",
        "commence_time": (datetime.now(timezone.utc) + timedelta(minutes=70)).isoformat(),
        "prediction": {
            "pick": "Oklahoma City Thunder",
            "pick_type": "moneyline",
            "confidence": 0.78,
            "edge": 12.0,
            "odds": 1.35,
            "reasoning": "V6 Analysis: Elite team vs tanking team. Line movement shows sharp action on OKC."
        }
    },
    {
        "id": f"PREGAME_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Golden State Warriors",
        "away_team": "Charlotte Hornets",
        "commence_time": (datetime.now(timezone.utc) + timedelta(minutes=65)).isoformat(),
        "prediction": {
            "pick": "Golden State Warriors",
            "pick_type": "spread",
            "confidence": 0.71,
            "edge": 7.8,
            "odds": 1.91,
            "reasoning": "V6 Analysis: Perfect storm - Charlotte missing 3 key players, GSW on win streak at home."
        }
    },
    {
        "id": f"PREGAME_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Cleveland Cavaliers",
        "away_team": "Detroit Pistons",
        "commence_time": (datetime.now(timezone.utc) + timedelta(minutes=50)).isoformat(),
        "prediction": {
            "pick": "Over 218.5",
            "pick_type": "total",
            "confidence": 0.73,
            "edge": 9.1,
            "odds": 1.91,
            "reasoning": "V6 Analysis: Both teams playing at fast pace lately. Line has moved up from 215.5."
        }
    }
]

async def simulate_pregame_predictions():
    """Simulate the pre-game prediction window and store predictions"""
    print("\n" + "=" * 80)
    print("🕐 SIMULATING PRE-GAME PREDICTION WINDOW")
    print("=" * 80)
    print(f"Current time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Simulating {len(PREGAME_WINDOW_GAMES)} games in the 45-75 minute prediction window\n")
    
    predictions_created = 0
    
    async with aiohttp.ClientSession() as session:
        for i, game in enumerate(PREGAME_WINDOW_GAMES):
            print(f"\n{'='*60}")
            print(f"📊 GAME {i+1}/{len(PREGAME_WINDOW_GAMES)}")
            print(f"{'='*60}")
            print(f"   {game['home_team']} vs {game['away_team']}")
            print(f"   Tip-off: {game['commence_time']}")
            
            pred = game["prediction"]
            print(f"\n   🎯 V6 PICK: {pred['pick']} ({pred['pick_type']})")
            print(f"   Confidence: {pred['confidence']*100:.0f}%")
            print(f"   Edge: {pred['edge']}%")
            print(f"   Odds: {pred['odds']}")
            print(f"   Reasoning: {pred['reasoning'][:60]}...")
            
            # Create prediction via API
            prediction_data = {
                "event_id": game["id"],
                "sport_key": game["sport_key"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "commence_time": game["commence_time"],
                "prediction_type": pred["pick_type"],
                "predicted_outcome": pred["pick"],
                "confidence": pred["confidence"],
                "analysis": pred["reasoning"],
                "ai_model": "betpredictor_v6",
                "odds_at_prediction": pred["odds"]
            }
            
            try:
                async with session.post(f"{BACKEND_URL}/recommendations", json=prediction_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        predictions_created += 1
                        print(f"\n   ✅ Prediction stored (ID: {result.get('id', 'unknown')[:8]}...)")
                    else:
                        text = await response.text()
                        print(f"\n   ⚠️ Failed to store: {response.status} - {text[:50]}")
            except Exception as e:
                print(f"\n   ❌ Error: {e}")
        
        # Create notification about new predictions
        print(f"\n{'='*60}")
        print("📬 Creating notification about new predictions...")
        print("{'='*60}")
        
        try:
            await session.post(f"{BACKEND_URL}/notifications/test")
            print("   ✅ Notification created")
        except Exception as e:
            print(f"   ⚠️ Notification failed: {e}")
        
        # Get updated recommendations
        print(f"\n{'='*60}")
        print("📋 CHECKING ACTIVE RECOMMENDATIONS")
        print("{'='*60}")
        
        try:
            async with session.get(f"{BACKEND_URL}/recommendations?limit=10") as response:
                if response.status == 200:
                    recs = await response.json()
                    print(f"   Total active picks: {len(recs)}")
                    for r in recs[:5]:
                        print(f"   • {r.get('predicted_outcome')} ({r.get('home_team')} vs {r.get('away_team')})")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 80)
    print("📊 PRE-GAME WINDOW SIMULATION COMPLETE")
    print("=" * 80)
    print(f"   Predictions created: {predictions_created}/{len(PREGAME_WINDOW_GAMES)}")
    print(f"   These picks are now available in the app!")
    print(f"\n   🎯 In production, these would be auto-generated by the scheduler")
    print(f"   45-75 minutes before each game's tip-off time.")
    
    return predictions_created

async def check_current_predictions():
    """Check current state of predictions"""
    print("\n" + "=" * 80)
    print("🔍 CURRENT PREDICTIONS STATE")
    print("=" * 80)
    
    async with aiohttp.ClientSession() as session:
        # Get V6 predictions
        try:
            async with session.get(f"{BACKEND_URL}/predictions/v6") as response:
                if response.status == 200:
                    data = await response.json()
                    stats = data.get("stats", {})
                    print(f"\n   V6 Algorithm Stats:")
                    print(f"   Total: {stats.get('total', 0)}")
                    print(f"   Wins: {stats.get('wins', 0)}")
                    print(f"   Losses: {stats.get('losses', 0)}")
                    print(f"   Pending: {stats.get('pending', 0)}")
                    print(f"   Win Rate: {stats.get('win_rate', 0)}%")
        except Exception as e:
            print(f"   Error fetching V6 stats: {e}")
        
        # Get comparison
        try:
            async with session.get(f"{BACKEND_URL}/predictions/comparison") as response:
                if response.status == 200:
                    comparison = await response.json()
                    print(f"\n   Algorithm Comparison:")
                    for algo, data in comparison.items():
                        if isinstance(data, dict) and "total" in data:
                            print(f"   {algo}: {data.get('total', 0)} predictions, {data.get('win_rate', 0)}% win rate")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Get performance
        try:
            async with session.get(f"{BACKEND_URL}/performance") as response:
                if response.status == 200:
                    perf = await response.json()
                    print(f"\n   Overall Performance:")
                    print(f"   Total: {perf.get('total', 0)}")
                    print(f"   Wins: {perf.get('wins', 0)}")
                    print(f"   Losses: {perf.get('losses', 0)}")
                    print(f"   Win Rate: {perf.get('win_rate', 0)}%")
                    print(f"   ROI: {perf.get('roi', 0)}%")
        except Exception as e:
            print(f"   Error: {e}")

async def main():
    # First check current state
    await check_current_predictions()
    
    # Simulate pre-game window
    await simulate_pregame_predictions()
    
    # Check state after
    await check_current_predictions()

if __name__ == "__main__":
    asyncio.run(main())
