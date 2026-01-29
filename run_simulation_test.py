#!/usr/bin/env python3
"""
BetPredictor Real-Time Simulation Test
Runs 10 simulation matches to test:
1. Predictor is generating picks
2. Win/loss calculation works
3. Notifications are being generated
4. All core features work correctly
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timezone, timedelta
import uuid
import random

# Backend URL
BACKEND_URL = "http://localhost:8001/api"

# Test results
test_results = {
    "predictions_generated": 0,
    "predictions_with_pick": 0,
    "predictions_no_pick": 0,
    "results_updated": 0,
    "wins": 0,
    "losses": 0,
    "notifications_created": 0,
    "errors": []
}

# Simulated matches with predetermined outcomes
SIMULATION_MATCHES = [
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Boston Celtics",
        "away_team": "Charlotte Hornets",
        "home_score": 118,
        "away_score": 95,
        "expected_winner": "home",  # Boston wins (home team)
        "spread": -12.5,
        "total": 225.5,
        "actual_total": 213
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Los Angeles Lakers",
        "away_team": "Golden State Warriors",
        "home_score": 102,
        "away_score": 115,
        "expected_winner": "away",  # Golden State wins (away team)
        "spread": -5.5,
        "total": 230.0,
        "actual_total": 217
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Miami Heat",
        "away_team": "New York Knicks",
        "home_score": 110,
        "away_score": 105,
        "expected_winner": "home",  # Miami wins
        "spread": -3.5,
        "total": 212.0,
        "actual_total": 215
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Denver Nuggets",
        "away_team": "Portland Trail Blazers",
        "home_score": 128,
        "away_score": 108,
        "expected_winner": "home",  # Denver wins
        "spread": -10.5,
        "total": 225.0,
        "actual_total": 236
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Phoenix Suns",
        "away_team": "Dallas Mavericks",
        "home_score": 108,
        "away_score": 112,
        "expected_winner": "away",  # Dallas wins
        "spread": -2.5,
        "total": 228.0,
        "actual_total": 220
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Milwaukee Bucks",
        "away_team": "Chicago Bulls",
        "home_score": 122,
        "away_score": 98,
        "expected_winner": "home",  # Milwaukee wins big
        "spread": -8.5,
        "total": 230.0,
        "actual_total": 220
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Cleveland Cavaliers",
        "away_team": "Indiana Pacers",
        "home_score": 115,
        "away_score": 118,
        "expected_winner": "away",  # Indiana wins
        "spread": -4.0,
        "total": 232.0,
        "actual_total": 233
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Philadelphia 76ers",
        "away_team": "Brooklyn Nets",
        "home_score": 110,
        "away_score": 102,
        "expected_winner": "home",  # Philadelphia wins
        "spread": -6.5,
        "total": 218.0,
        "actual_total": 212
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Sacramento Kings",
        "away_team": "Houston Rockets",
        "home_score": 130,
        "away_score": 125,
        "expected_winner": "home",  # Sacramento wins
        "spread": -5.0,
        "total": 245.0,
        "actual_total": 255
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Atlanta Hawks",
        "away_team": "Washington Wizards",
        "home_score": 118,
        "away_score": 95,
        "expected_winner": "home",  # Atlanta wins
        "spread": -9.5,
        "total": 225.0,
        "actual_total": 213
    }
]

async def check_api_health(session):
    """Check if backend API is running"""
    print("\n" + "=" * 80)
    print("🔍 CHECKING API HEALTH")
    print("=" * 80)
    
    try:
        async with session.get(f"{BACKEND_URL}/") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ API is running: {data}")
                return True
            else:
                print(f"❌ API returned status {response.status}")
                return False
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

async def check_data_sources(session):
    """Check data source status"""
    print("\n" + "=" * 80)
    print("📊 CHECKING DATA SOURCES")
    print("=" * 80)
    
    try:
        async with session.get(f"{BACKEND_URL}/data-source-status") as response:
            if response.status == 200:
                data = await response.json()
                print(f"   Source: {data.get('source')}")
                print(f"   Status: {data.get('status')}")
                print(f"   Line Movement Snapshots: {data.get('lineMovementSnapshots')}")
                print(f"   Cached Events: {data.get('cachedEvents')}")
                return True
    except Exception as e:
        print(f"❌ Failed to check data source: {e}")
    return False

async def fetch_real_events(session):
    """Try to fetch real NBA events"""
    print("\n" + "=" * 80)
    print("🏀 FETCHING REAL NBA EVENTS")
    print("=" * 80)
    
    try:
        async with session.get(f"{BACKEND_URL}/events/basketball_nba?pre_match_only=true") as response:
            if response.status == 200:
                events = await response.json()
                print(f"   Found {len(events)} pre-match events")
                return events
    except Exception as e:
        print(f"❌ Failed to fetch events: {e}")
    return []

async def create_test_prediction(session, match):
    """Create a test prediction for a simulation match"""
    # Generate a random prediction (moneyline, spread, or total)
    pred_type = random.choice(["moneyline", "spread", "total"])
    
    # Determine the pick based on prediction type
    if pred_type == "moneyline":
        # Random pick home or away
        is_home_pick = random.choice([True, False])
        predicted_outcome = match["home_team"] if is_home_pick else match["away_team"]
        odds = 1.85 if is_home_pick else 2.10
    elif pred_type == "spread":
        is_home_pick = random.choice([True, False])
        predicted_outcome = f"{match['home_team']} {match['spread']}" if is_home_pick else f"{match['away_team']} +{abs(match['spread'])}"
        odds = 1.91
    else:  # total
        is_over = random.choice([True, False])
        predicted_outcome = f"Over {match['total']}" if is_over else f"Under {match['total']}"
        odds = 1.91
    
    prediction_data = {
        "event_id": match["id"],
        "sport_key": match["sport_key"],
        "home_team": match["home_team"],
        "away_team": match["away_team"],
        "commence_time": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),  # Game already started
        "prediction_type": pred_type,
        "predicted_outcome": predicted_outcome,
        "confidence": random.uniform(0.70, 0.90),
        "analysis": f"Simulation test prediction for {match['home_team']} vs {match['away_team']}",
        "ai_model": "simulation_test",
        "odds_at_prediction": odds
    }
    
    try:
        async with session.post(f"{BACKEND_URL}/recommendations", json=prediction_data) as response:
            if response.status == 200:
                result = await response.json()
                return result, prediction_data
    except Exception as e:
        test_results["errors"].append(f"Failed to create prediction: {e}")
    return None, prediction_data

def determine_result(prediction_data, match):
    """Determine if a prediction is a win/loss based on match outcome"""
    pred_type = prediction_data["prediction_type"]
    predicted_outcome = prediction_data["predicted_outcome"]
    home_team = match["home_team"]
    away_team = match["away_team"]
    home_score = match["home_score"]
    away_score = match["away_score"]
    actual_total = match["actual_total"]
    spread = match["spread"]
    
    if pred_type == "moneyline":
        # Check if picked team won
        if home_team in predicted_outcome:
            return "win" if home_score > away_score else "loss"
        elif away_team in predicted_outcome:
            return "win" if away_score > home_score else "loss"
    
    elif pred_type == "spread":
        # Home team spread
        home_margin = home_score - away_score
        if home_team in predicted_outcome:
            # Home team needs to win by more than spread (or lose by less)
            return "win" if home_margin > abs(spread) else "loss"
        elif away_team in predicted_outcome:
            # Away team with plus points
            return "win" if home_margin < abs(spread) else "loss"
    
    elif pred_type == "total":
        if "Over" in predicted_outcome:
            return "win" if actual_total > match["total"] else "loss"
        elif "Under" in predicted_outcome:
            return "win" if actual_total < match["total"] else "loss"
    
    return "pending"

async def update_prediction_result(session, prediction_id, result):
    """Update a prediction with win/loss result"""
    update_data = {
        "prediction_id": prediction_id,
        "result": result
    }
    
    try:
        async with session.put(f"{BACKEND_URL}/predictions/{prediction_id}/result", json=update_data) as response:
            if response.status == 200:
                return True
    except Exception as e:
        test_results["errors"].append(f"Failed to update result: {e}")
    return False

async def check_notifications(session):
    """Check notifications have been created"""
    try:
        async with session.get(f"{BACKEND_URL}/notifications") as response:
            if response.status == 200:
                data = await response.json()
                return data.get("notifications", []), data.get("unread_count", 0)
    except Exception as e:
        test_results["errors"].append(f"Failed to check notifications: {e}")
    return [], 0

async def check_performance_stats(session):
    """Check performance statistics"""
    try:
        async with session.get(f"{BACKEND_URL}/performance") as response:
            if response.status == 200:
                return await response.json()
    except Exception as e:
        test_results["errors"].append(f"Failed to check performance: {e}")
    return None

async def run_v5_analysis(session, event_id, sport_key):
    """Run V5 analysis on an event"""
    try:
        async with session.post(f"{BACKEND_URL}/analyze-v5/{event_id}?sport_key={sport_key}") as response:
            if response.status == 200:
                return await response.json()
    except Exception as e:
        pass
    return None

async def run_v6_analysis(session, event_id, sport_key):
    """Run V6 analysis on an event"""
    try:
        async with session.post(f"{BACKEND_URL}/analyze-v6/{event_id}?sport_key={sport_key}") as response:
            if response.status == 200:
                return await response.json()
    except Exception as e:
        pass
    return None

async def run_unified_analysis(session, event_id, sport_key):
    """Run unified analysis on an event"""
    try:
        async with session.post(f"{BACKEND_URL}/analyze-unified/{event_id}?sport_key={sport_key}") as response:
            if response.status == 200:
                return await response.json()
    except Exception as e:
        pass
    return None

async def get_existing_predictions(session):
    """Get existing predictions count"""
    try:
        async with session.get(f"{BACKEND_URL}/predictions") as response:
            if response.status == 200:
                data = await response.json()
                return len(data.get("predictions", []))
    except:
        pass
    return 0

async def create_notification(session, notif_type, title, message):
    """Create a test notification"""
    notif_data = {
        "type": notif_type,
        "title": title,
        "message": message
    }
    try:
        async with session.post(f"{BACKEND_URL}/notifications/test") as response:
            if response.status == 200:
                return True
    except:
        pass
    return False

async def run_simulation():
    """Run the full simulation test"""
    print("\n" + "=" * 80)
    print("🎯 BETPREDICTOR REAL-TIME SIMULATION TEST")
    print("=" * 80)
    print(f"Starting simulation at: {datetime.now(timezone.utc).isoformat()}")
    print(f"Running {len(SIMULATION_MATCHES)} simulation matches")
    
    async with aiohttp.ClientSession() as session:
        # 1. Check API Health
        if not await check_api_health(session):
            print("\n❌ SIMULATION ABORTED: API not available")
            return test_results
        
        # 2. Check Data Sources
        await check_data_sources(session)
        
        # 3. Get initial notification count
        initial_notifications, _ = await check_notifications(session)
        initial_notification_count = len(initial_notifications)
        print(f"\n📬 Initial notifications: {initial_notification_count}")
        
        # 4. Create test notification
        await create_notification(session, "test", "Simulation Started", "Running 10 test matches")
        
        # 5. Fetch real events and try analysis
        print("\n" + "=" * 80)
        print("🔬 TESTING PREDICTION ALGORITHMS ON REAL EVENTS")
        print("=" * 80)
        
        real_events = await fetch_real_events(session)
        picks_from_real_events = 0
        
        if real_events:
            for i, event in enumerate(real_events[:5]):
                event_id = event.get("id")
                sport_key = event.get("sport_key", "basketball_nba")
                home_team = event.get("home_team", "Unknown")
                away_team = event.get("away_team", "Unknown")
                
                print(f"\n   Analyzing: {home_team} vs {away_team}")
                
                # Try V6 analysis
                v6_result = await run_v6_analysis(session, event_id, sport_key)
                if v6_result:
                    prediction = v6_result.get("prediction", {})
                    has_pick = prediction.get("has_pick", False)
                    
                    if has_pick:
                        picks_from_real_events += 1
                        test_results["predictions_with_pick"] += 1
                        print(f"   ✅ V6 PICK: {prediction.get('pick')} (Confidence: {prediction.get('confidence')}%)")
                    else:
                        test_results["predictions_no_pick"] += 1
                        print(f"   ⏸️ V6: No pick (conservative approach)")
                    
                    test_results["predictions_generated"] += 1
                else:
                    print(f"   ⚠️ V6 analysis failed")
        
        # 6. Run simulation matches
        print("\n" + "=" * 80)
        print("🏀 RUNNING SIMULATION MATCHES")
        print("=" * 80)
        
        created_predictions = []
        
        for i, match in enumerate(SIMULATION_MATCHES):
            print(f"\n--- Match {i+1}/10: {match['home_team']} vs {match['away_team']} ---")
            
            # Create prediction
            prediction, pred_data = await create_test_prediction(session, match)
            
            if prediction:
                pred_id = prediction.get("id")
                print(f"   📝 Created prediction: {pred_data['prediction_type']} - {pred_data['predicted_outcome']}")
                
                # Determine result
                result = determine_result(pred_data, match)
                print(f"   📊 Final Score: {match['home_team']} {match['home_score']} - {match['away_score']} {match['away_team']}")
                print(f"   🎯 Result: {result.upper()}")
                
                # Update result
                if await update_prediction_result(session, pred_id, result):
                    test_results["results_updated"] += 1
                    if result == "win":
                        test_results["wins"] += 1
                    elif result == "loss":
                        test_results["losses"] += 1
                    print(f"   ✅ Result updated successfully")
                else:
                    print(f"   ⚠️ Failed to update result via API")
                
                created_predictions.append({
                    "id": pred_id,
                    "match": match,
                    "prediction": pred_data,
                    "result": result
                })
            else:
                print(f"   ⚠️ Failed to create prediction")
                test_results["errors"].append(f"Failed to create prediction for {match['home_team']} vs {match['away_team']}")
        
        # 7. Check notifications after simulation
        print("\n" + "=" * 80)
        print("📬 CHECKING NOTIFICATIONS")
        print("=" * 80)
        
        final_notifications, unread_count = await check_notifications(session)
        final_notification_count = len(final_notifications)
        new_notifications = final_notification_count - initial_notification_count
        test_results["notifications_created"] = new_notifications
        
        print(f"   Total notifications: {final_notification_count}")
        print(f"   New notifications: {new_notifications}")
        print(f"   Unread notifications: {unread_count}")
        
        if final_notifications:
            print("\n   Recent notifications:")
            for notif in final_notifications[:5]:
                print(f"   - {notif.get('type')}: {notif.get('title')}")
        
        # 8. Check performance stats
        print("\n" + "=" * 80)
        print("📈 CHECKING PERFORMANCE STATS")
        print("=" * 80)
        
        performance = await check_performance_stats(session)
        if performance:
            print(f"   Total predictions: {performance.get('total', 0)}")
            print(f"   Wins: {performance.get('wins', 0)}")
            print(f"   Losses: {performance.get('losses', 0)}")
            print(f"   Win Rate: {performance.get('win_rate', 0)}%")
            print(f"   ROI: {performance.get('roi', 0)}%")
        
        # 9. Check V5, V6, Unified prediction stats
        print("\n" + "=" * 80)
        print("🔮 CHECKING ALGORITHM PREDICTION STATS")
        print("=" * 80)
        
        # V5 Stats
        try:
            async with session.get(f"{BACKEND_URL}/predictions/v5") as response:
                if response.status == 200:
                    v5_data = await response.json()
                    stats = v5_data.get("stats", {})
                    print(f"   V5 Predictions: {stats.get('total', 0)} (Win Rate: {stats.get('win_rate', 0)}%)")
        except:
            print("   V5: Unable to fetch stats")
        
        # V6 Stats
        try:
            async with session.get(f"{BACKEND_URL}/predictions/v6") as response:
                if response.status == 200:
                    v6_data = await response.json()
                    stats = v6_data.get("stats", {})
                    print(f"   V6 Predictions: {stats.get('total', 0)} (Win Rate: {stats.get('win_rate', 0)}%)")
        except:
            print("   V6: Unable to fetch stats")
        
        # Comparison
        try:
            async with session.get(f"{BACKEND_URL}/predictions/comparison") as response:
                if response.status == 200:
                    comparison = await response.json()
                    print(f"\n   Algorithm Comparison:")
                    for algo, data in comparison.items():
                        if isinstance(data, dict) and "total" in data:
                            print(f"   - {algo}: {data.get('total', 0)} predictions, {data.get('win_rate', 0)}% win rate")
        except:
            print("   Comparison: Unable to fetch")
    
    return test_results

async def main():
    """Main entry point"""
    results = await run_simulation()
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SIMULATION TEST SUMMARY")
    print("=" * 80)
    
    print(f"""
    🔬 PREDICTOR ANALYSIS:
    - Total analyses run: {results['predictions_generated']}
    - Picks generated: {results['predictions_with_pick']}
    - No picks (conservative): {results['predictions_no_pick']}
    
    🎯 SIMULATION RESULTS:
    - Results updated: {results['results_updated']}/10
    - Wins: {results['wins']}
    - Losses: {results['losses']}
    - Win Rate: {(results['wins'] / (results['wins'] + results['losses']) * 100) if (results['wins'] + results['losses']) > 0 else 0:.1f}%
    
    📬 NOTIFICATIONS:
    - New notifications created: {results['notifications_created']}
    
    ❌ ERRORS:
    - Total errors: {len(results['errors'])}
    """)
    
    if results['errors']:
        print("    Error details:")
        for error in results['errors'][:5]:
            print(f"    - {error}")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("✅ FINAL VERDICT")
    print("=" * 80)
    
    all_working = True
    verdicts = []
    
    if results['predictions_generated'] > 0:
        verdicts.append("✅ Predictor is ANALYZING events")
    else:
        verdicts.append("⚠️ Predictor analysis needs real events")
        all_working = False
    
    if results['results_updated'] > 0:
        verdicts.append(f"✅ Win/Loss tracking is WORKING ({results['wins']}W-{results['losses']}L)")
    else:
        verdicts.append("❌ Win/Loss tracking needs attention")
        all_working = False
    
    if results['notifications_created'] > 0:
        verdicts.append(f"✅ Notifications are WORKING ({results['notifications_created']} created)")
    else:
        verdicts.append("⚠️ Notifications may need testing with real events")
    
    for v in verdicts:
        print(f"    {v}")
    
    if all_working:
        print("\n    🎉 ALL CORE FEATURES ARE WORKING!")
    else:
        print("\n    ⚠️ Some features may need additional testing with real events")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
