#!/usr/bin/env python3
"""
Create sample picks with COMPREHENSIVE V6-style analysis
"""

import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
import uuid

BACKEND_URL = "http://localhost:8001/api"

# Sample picks with comprehensive V6-style analysis
SAMPLE_PICKS = [
    # SPREAD pick with full analysis
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Boston Celtics",
        "away_team": "Portland Trail Blazers",
        "commence_time": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
        "prediction_type": "spread",
        "predicted_outcome": "Boston Celtics -12.5",
        "confidence": 0.75,
        "analysis": """✓ 80% of models agree on Boston Celtics
📊 Ensemble confidence: 75.2%
💰 Estimated edge: +8.5%

✓ ELO advantage: +185 points (Boston 1720 vs Portland 1535)
✓ REST ADVANTAGE: Boston (3 days rest) vs Portland (back-to-back)
✓ HOME COURT: Boston 28-3 at TD Garden this season
✓ INJURIES: Portland missing Anfernee Simons (hamstring), Scoot Henderson (ankle)

🎲 Monte Carlo: 78.5% home win probability (1000 simulations)
📈 Line Movement: Opened -10.5, sharp money moved to -12.5""",
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
        "predicted_outcome": "Denver Nuggets -14.5",
        "confidence": 0.78,
        "analysis": """✓ 100% of models agree on Denver Nuggets
📊 Ensemble confidence: 78.4%
💰 Estimated edge: +10.2%

✓ ELO advantage: +210 points (Denver 1695 vs San Antonio 1485)
✓ ALTITUDE: Ball Arena at 5,280 ft - Spurs playing 4th game in 5 days
✓ FATIGUE: San Antonio on back-to-back after overtime loss
✓ INJURIES: Wembanyama questionable (rest), Keldon Johnson OUT

🎲 Monte Carlo: 82.1% home win probability
📈 Line Movement: Sharp money flooding Denver, line moved from -11.5 to -14.5
🧠 Psychology: Public on Spurs (+7.2% contrarian value)""",
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
        "predicted_outcome": "Oklahoma City Thunder -15.5",
        "confidence": 0.72,
        "analysis": """✓ 80% of models agree on Oklahoma City Thunder
📊 Ensemble confidence: 72.1%
💰 Estimated edge: +7.8%

✓ ELO advantage: +225 points (OKC 1745 vs Utah 1520)
✓ FORM: OKC on 8-game win streak, Utah lost 6 straight
✓ HOME/AWAY: OKC 25-2 at home, Utah 8-22 on road
✓ INJURIES: Lauri Markkanen OUT (back), Collin Sexton questionable

🎲 Monte Carlo: 79.2% home win probability
📈 Line Movement: Opened -12.5, professional money on Thunder""",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.91
    },
    # MONEYLINE pick
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Cleveland Cavaliers",
        "away_team": "Washington Wizards",
        "commence_time": (datetime.now(timezone.utc) + timedelta(hours=2.5)).isoformat(),
        "prediction_type": "moneyline",
        "predicted_outcome": "Cleveland Cavaliers",
        "confidence": 0.82,
        "analysis": """✓ 100% of models agree on Cleveland Cavaliers ML
📊 Ensemble confidence: 82.3%
💰 Estimated edge: +12.5%

✓ ELO advantage: +295 points (Cleveland 1780 vs Washington 1485)
✓ SEASON: Cleveland #1 in East (48-8), Washington worst record (5-50)
✓ HOME COURT: Cleveland 26-2 at Rocket Mortgage FieldHouse
✓ INJURIES: Washington missing Kuzma, Poole, Jones - decimated roster

🎲 Monte Carlo: 91.2% home win probability
📈 Line Movement: Heavy action on Cleveland, -18.5 spread
🧠 Psychology: Even at -1200 odds, edge exists vs implied probability""",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.15
    },
    # TOTAL pick
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Golden State Warriors",
        "away_team": "Charlotte Hornets",
        "commence_time": (datetime.now(timezone.utc) + timedelta(hours=3.5)).isoformat(),
        "prediction_type": "total",
        "predicted_outcome": "Over 225.5",
        "confidence": 0.73,
        "analysis": """✓ 80% of models agree on OVER 225.5
📊 Ensemble confidence: 73.1%
💰 Estimated edge: +6.8%

✓ PACE: Warriors 4th fastest pace (102.3), Hornets 6th (101.8)
✓ DEFENSE: Both teams bottom-10 in defensive rating
✓ RECENT GAMES: Combined average 235.2 points in last 5 meetings
✓ NO INJURIES: Both teams relatively healthy

🎲 Monte Carlo: Projected score 118-112 (230 total)
📈 Line Movement: Opened 222.5, sharp money pushed to 225.5
🧠 Poisson Model: 67% probability of 226+ total points""",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.91
    },
    # COMPLETED picks with results
    {
        "event_id": f"PICK_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Miami Heat",
        "away_team": "New York Knicks",
        "commence_time": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        "prediction_type": "spread",
        "predicted_outcome": "Miami Heat -3.5",
        "confidence": 0.71,
        "analysis": """✓ 60% of models agree on Miami Heat -3.5
📊 Ensemble confidence: 71.2%
💰 Estimated edge: +5.2%

✓ ELO advantage: +65 points
✓ HOME COURT: Miami 18-8 at Kaseya Center
✓ REST: Miami 2 days rest vs Knicks back-to-back
✓ INJURIES: Brunson questionable (knee)

🎲 Monte Carlo: 62.3% home cover probability
📈 Line Movement: Stable at -3.5""",
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
        "analysis": """✓ 60% of models agree on Lakers ML
📊 Ensemble confidence: 70.1%
💰 Estimated edge: +4.8%

✓ HOME COURT: Lakers 20-10 at Crypto.com Arena
✓ HEALTH: AD and LeBron both available
✓ OPPONENT: Phoenix missing Durant (calf)

🎲 Monte Carlo: 58.5% home win probability""",
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
        "analysis": """✓ 80% of models agree on Bucks -8.5
📊 Ensemble confidence: 74.3%
💰 Estimated edge: +7.2%

✓ ELO advantage: +145 points
✓ Giannis triple-double in last 3 vs Bulls
✓ Bulls 5-15 vs winning teams

🎲 Monte Carlo: 71.8% home cover probability""",
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
        "analysis": """✓ 60% of models agree on UNDER 218.5
📊 Ensemble confidence: 72.0%
💰 Estimated edge: +5.5%

✓ Pace: Both teams playing slower (98.2, 97.8)
✓ Defense: 76ers top-10 defensive rating this month

🎲 Monte Carlo: 62.1% under probability""",
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
        "analysis": """✓ 80% of models agree on Hawks -11.5
📊 Ensemble confidence: 76.2%
💰 Estimated edge: +8.8%

✓ ELO advantage: +180 points
✓ Detroit on 18-game losing streak
✓ Trae Young averaging 32 PPG vs Pistons

🎲 Monte Carlo: 75.3% home cover probability""",
        "ai_model": "betpredictor_v6",
        "odds_at_prediction": 1.91,
        "result": "win"
    }
]

async def create_picks():
    """Create sample picks with comprehensive analysis"""
    print("Creating sample picks with comprehensive V6 analysis...\n")
    
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
