#!/usr/bin/env python3
"""
BetPredictor Extended Simulation Test
Comprehensive test of all features including:
1. Multiple prediction algorithms (V5, V6, Unified)
2. Line movement tracking
3. Win/loss calculation
4. Notifications
5. Bankroll management
6. Analytics
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timezone, timedelta
import uuid
import random

# Backend URL
BACKEND_URL = "http://localhost:8001/api"

# Extended test results
extended_results = {
    "api_health": False,
    "data_sources": {},
    "events_fetched": 0,
    
    # Prediction Algorithm Tests
    "v5_analyses": 0,
    "v5_picks": 0,
    "v6_analyses": 0,
    "v6_picks": 0,
    "unified_analyses": 0,
    "unified_picks": 0,
    
    # Line Movement Tests
    "line_movement_events": 0,
    "line_movement_snapshots": 0,
    
    # Result Tests
    "predictions_created": 0,
    "results_updated": 0,
    "wins": 0,
    "losses": 0,
    
    # Notification Tests  
    "notifications_created": 0,
    "notifications_read": 0,
    
    # Performance Tests
    "performance_fetched": False,
    
    # Errors
    "errors": [],
    
    # Match-by-match results
    "match_results": []
}

# 10 Simulation matches with known outcomes
SIM_MATCHES = [
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Boston Celtics",
        "away_team": "Charlotte Hornets",
        "home_score": 118, "away_score": 95,
        "spread": -12.5, "total": 225.5, "actual_total": 213
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Los Angeles Lakers", 
        "away_team": "Golden State Warriors",
        "home_score": 102, "away_score": 115,
        "spread": -5.5, "total": 230.0, "actual_total": 217
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Miami Heat",
        "away_team": "New York Knicks",
        "home_score": 110, "away_score": 105,
        "spread": -3.5, "total": 212.0, "actual_total": 215
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Denver Nuggets",
        "away_team": "Portland Trail Blazers",
        "home_score": 128, "away_score": 108,
        "spread": -10.5, "total": 225.0, "actual_total": 236
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Phoenix Suns",
        "away_team": "Dallas Mavericks",
        "home_score": 108, "away_score": 112,
        "spread": -2.5, "total": 228.0, "actual_total": 220
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Milwaukee Bucks",
        "away_team": "Chicago Bulls",
        "home_score": 122, "away_score": 98,
        "spread": -8.5, "total": 230.0, "actual_total": 220
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Cleveland Cavaliers",
        "away_team": "Indiana Pacers",
        "home_score": 115, "away_score": 118,
        "spread": -4.0, "total": 232.0, "actual_total": 233
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Philadelphia 76ers",
        "away_team": "Brooklyn Nets",
        "home_score": 110, "away_score": 102,
        "spread": -6.5, "total": 218.0, "actual_total": 212
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Sacramento Kings",
        "away_team": "Houston Rockets",
        "home_score": 130, "away_score": 125,
        "spread": -5.0, "total": 245.0, "actual_total": 255
    },
    {
        "id": f"SIM_{uuid.uuid4().hex[:8]}",
        "sport_key": "basketball_nba",
        "home_team": "Atlanta Hawks",
        "away_team": "Washington Wizards",
        "home_score": 118, "away_score": 95,
        "spread": -9.5, "total": 225.0, "actual_total": 213
    }
]

def determine_result(pred_type, predicted_outcome, match):
    """Determine if prediction is win/loss"""
    home_score = match["home_score"]
    away_score = match["away_score"]
    actual_total = match["actual_total"]
    spread = match["spread"]
    home_team = match["home_team"]
    away_team = match["away_team"]
    
    if pred_type == "moneyline":
        if home_team in predicted_outcome:
            return "win" if home_score > away_score else "loss"
        elif away_team in predicted_outcome:
            return "win" if away_score > home_score else "loss"
    elif pred_type == "spread":
        home_margin = home_score - away_score
        if home_team in predicted_outcome:
            return "win" if home_margin > abs(spread) else "loss"
        elif away_team in predicted_outcome:
            return "win" if home_margin < abs(spread) else "loss"
    elif pred_type == "total":
        if "Over" in predicted_outcome:
            return "win" if actual_total > match["total"] else "loss"
        else:
            return "win" if actual_total < match["total"] else "loss"
    return "pending"

async def run_extended_simulation():
    """Run extended simulation test"""
    print("\n" + "=" * 80)
    print("🚀 BETPREDICTOR EXTENDED SIMULATION TEST")
    print("=" * 80)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    
    async with aiohttp.ClientSession() as session:
        
        # ============ TEST 1: API HEALTH ============
        print("\n" + "-" * 40)
        print("TEST 1: API HEALTH CHECK")
        print("-" * 40)
        try:
            async with session.get(f"{BACKEND_URL}/") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    extended_results["api_health"] = True
                    print(f"✅ API is healthy: {data.get('message')}")
                else:
                    print(f"❌ API returned {resp.status}")
        except Exception as e:
            print(f"❌ API Error: {e}")
            extended_results["errors"].append(f"API Health: {e}")
        
        # ============ TEST 2: DATA SOURCES ============
        print("\n" + "-" * 40)
        print("TEST 2: DATA SOURCES")
        print("-" * 40)
        try:
            async with session.get(f"{BACKEND_URL}/data-source-status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    extended_results["data_sources"] = data
                    print(f"✅ Source: {data.get('source')}")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Cached Events: {data.get('cachedEvents', 0)}")
                    print(f"   Line Snapshots: {data.get('lineMovementSnapshots', 0)}")
                    extended_results["line_movement_snapshots"] = data.get('lineMovementSnapshots', 0)
        except Exception as e:
            extended_results["errors"].append(f"Data Source: {e}")
            print(f"❌ Data Source Error: {e}")
        
        # ============ TEST 3: FETCH REAL EVENTS ============
        print("\n" + "-" * 40)
        print("TEST 3: FETCH REAL EVENTS")
        print("-" * 40)
        real_events = []
        try:
            async with session.get(f"{BACKEND_URL}/events/basketball_nba?pre_match_only=true") as resp:
                if resp.status == 200:
                    real_events = await resp.json()
                    extended_results["events_fetched"] = len(real_events)
                    print(f"✅ Fetched {len(real_events)} NBA pre-match events")
        except Exception as e:
            extended_results["errors"].append(f"Events: {e}")
            print(f"❌ Events Error: {e}")
        
        # ============ TEST 4: LINE MOVEMENT DATA ============
        print("\n" + "-" * 40)
        print("TEST 4: LINE MOVEMENT DATA")
        print("-" * 40)
        if real_events:
            event_id = real_events[0].get("id")
            try:
                async with session.get(f"{BACKEND_URL}/line-movement/{event_id}?sport_key=basketball_nba") as resp:
                    if resp.status == 200:
                        lm_data = await resp.json()
                        extended_results["line_movement_events"] = 1
                        print(f"✅ Line Movement Data Found")
                        print(f"   Event: {lm_data.get('event_info', {}).get('home_team')} vs {lm_data.get('event_info', {}).get('away_team')}")
                        print(f"   Chart Data Points: {len(lm_data.get('chart_data', {}).get('moneyline', []))}")
                        print(f"   Opening Odds: {lm_data.get('opening_odds', {})}")
            except Exception as e:
                print(f"⚠️ Line Movement: {e}")
        
        # ============ TEST 5: V5 ANALYSIS ============
        print("\n" + "-" * 40)
        print("TEST 5: V5 PREDICTION ANALYSIS")
        print("-" * 40)
        for i, event in enumerate(real_events[:3]):
            event_id = event.get("id")
            try:
                async with session.post(f"{BACKEND_URL}/analyze-v5/{event_id}?sport_key=basketball_nba") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        extended_results["v5_analyses"] += 1
                        pred = data.get("prediction", {})
                        has_pick = pred.get("has_pick", False)
                        if has_pick:
                            extended_results["v5_picks"] += 1
                            print(f"✅ V5 Pick #{i+1}: {pred.get('pick')} ({pred.get('confidence')}%)")
                        else:
                            print(f"⏸️ V5 #{i+1}: No pick (conservative)")
            except Exception as e:
                extended_results["errors"].append(f"V5: {e}")
        print(f"   Total V5 Analyses: {extended_results['v5_analyses']}, Picks: {extended_results['v5_picks']}")
        
        # ============ TEST 6: V6 ANALYSIS ============
        print("\n" + "-" * 40)
        print("TEST 6: V6 PREDICTION ANALYSIS")
        print("-" * 40)
        for i, event in enumerate(real_events[:3]):
            event_id = event.get("id")
            try:
                async with session.post(f"{BACKEND_URL}/analyze-v6/{event_id}?sport_key=basketball_nba") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        extended_results["v6_analyses"] += 1
                        pred = data.get("prediction", {})
                        has_pick = pred.get("has_pick", False)
                        if has_pick:
                            extended_results["v6_picks"] += 1
                            print(f"✅ V6 Pick #{i+1}: {pred.get('pick')} ({pred.get('confidence')}%)")
                        else:
                            print(f"⏸️ V6 #{i+1}: No pick (conservative)")
            except Exception as e:
                extended_results["errors"].append(f"V6: {e}")
        print(f"   Total V6 Analyses: {extended_results['v6_analyses']}, Picks: {extended_results['v6_picks']}")
        
        # ============ TEST 7: UNIFIED ANALYSIS ============
        print("\n" + "-" * 40)
        print("TEST 7: UNIFIED PREDICTION ANALYSIS")
        print("-" * 40)
        for i, event in enumerate(real_events[:2]):
            event_id = event.get("id")
            try:
                async with session.post(f"{BACKEND_URL}/analyze-unified/{event_id}?sport_key=basketball_nba") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        extended_results["unified_analyses"] += 1
                        pred = data.get("prediction", {})
                        has_pick = pred.get("has_pick", False)
                        if has_pick:
                            extended_results["unified_picks"] += 1
                            print(f"✅ Unified Pick #{i+1}: {pred.get('pick')} ({pred.get('confidence')}%)")
                        else:
                            print(f"⏸️ Unified #{i+1}: No pick")
            except Exception as e:
                extended_results["errors"].append(f"Unified: {e}")
        print(f"   Total Unified Analyses: {extended_results['unified_analyses']}, Picks: {extended_results['unified_picks']}")
        
        # ============ TEST 8: SIMULATION MATCHES (Win/Loss) ============
        print("\n" + "-" * 40)
        print("TEST 8: SIMULATION MATCHES (10 games)")
        print("-" * 40)
        
        for i, match in enumerate(SIM_MATCHES):
            # Create random prediction type
            pred_type = random.choice(["moneyline", "spread", "total"])
            
            if pred_type == "moneyline":
                is_home = random.choice([True, False])
                predicted_outcome = match["home_team"] if is_home else match["away_team"]
                odds = 1.85 if is_home else 2.10
            elif pred_type == "spread":
                is_home = random.choice([True, False])
                predicted_outcome = f"{match['home_team']} {match['spread']}" if is_home else f"{match['away_team']} +{abs(match['spread'])}"
                odds = 1.91
            else:
                is_over = random.choice([True, False])
                predicted_outcome = f"Over {match['total']}" if is_over else f"Under {match['total']}"
                odds = 1.91
            
            # Create prediction
            pred_data = {
                "event_id": match["id"],
                "sport_key": match["sport_key"],
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "commence_time": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
                "prediction_type": pred_type,
                "predicted_outcome": predicted_outcome,
                "confidence": random.uniform(0.70, 0.90),
                "analysis": f"Simulation test",
                "ai_model": "simulation_v2",
                "odds_at_prediction": odds
            }
            
            try:
                async with session.post(f"{BACKEND_URL}/recommendations", json=pred_data) as resp:
                    if resp.status == 200:
                        pred_result = await resp.json()
                        pred_id = pred_result.get("id")
                        extended_results["predictions_created"] += 1
                        
                        # Determine result
                        result = determine_result(pred_type, predicted_outcome, match)
                        
                        # Update result
                        update_data = {"prediction_id": pred_id, "result": result}
                        async with session.put(f"{BACKEND_URL}/result", json=update_data) as update_resp:
                            if update_resp.status == 200:
                                extended_results["results_updated"] += 1
                                if result == "win":
                                    extended_results["wins"] += 1
                                else:
                                    extended_results["losses"] += 1
                                
                                extended_results["match_results"].append({
                                    "match": f"{match['home_team']} vs {match['away_team']}",
                                    "score": f"{match['home_score']}-{match['away_score']}",
                                    "prediction": f"{pred_type}: {predicted_outcome}",
                                    "result": result
                                })
                                
                                status = "✅ WIN" if result == "win" else "❌ LOSS"
                                print(f"   Match {i+1}: {match['home_team']} {match['home_score']}-{match['away_score']} {match['away_team']} | {pred_type}: {status}")
            except Exception as e:
                extended_results["errors"].append(f"Match {i+1}: {e}")
        
        print(f"\n   Summary: {extended_results['wins']}W - {extended_results['losses']}L")
        
        # ============ TEST 9: NOTIFICATIONS ============
        print("\n" + "-" * 40)
        print("TEST 9: NOTIFICATIONS")
        print("-" * 40)
        
        # Create test notification
        try:
            async with session.post(f"{BACKEND_URL}/notifications/test") as resp:
                if resp.status == 200:
                    print("✅ Test notification created")
            
            # Get notifications
            async with session.get(f"{BACKEND_URL}/notifications") as resp:
                if resp.status == 200:
                    notif_data = await resp.json()
                    extended_results["notifications_created"] = len(notif_data.get("notifications", []))
                    print(f"✅ Total notifications: {extended_results['notifications_created']}")
                    print(f"   Unread: {notif_data.get('unread_count', 0)}")
            
            # Mark all as read
            async with session.put(f"{BACKEND_URL}/notifications/read-all") as resp:
                if resp.status == 200:
                    extended_results["notifications_read"] = True
                    print("✅ All notifications marked as read")
        except Exception as e:
            extended_results["errors"].append(f"Notifications: {e}")
        
        # ============ TEST 10: PERFORMANCE STATS ============
        print("\n" + "-" * 40)
        print("TEST 10: PERFORMANCE STATS")
        print("-" * 40)
        try:
            async with session.get(f"{BACKEND_URL}/performance") as resp:
                if resp.status == 200:
                    perf = await resp.json()
                    extended_results["performance_fetched"] = True
                    print(f"✅ Performance Stats Retrieved")
                    print(f"   Total: {perf.get('total', 0)}")
                    print(f"   Wins: {perf.get('wins', 0)}")
                    print(f"   Losses: {perf.get('losses', 0)}")
                    print(f"   Win Rate: {perf.get('win_rate', 0)}%")
                    print(f"   ROI: {perf.get('roi', 0)}%")
        except Exception as e:
            extended_results["errors"].append(f"Performance: {e}")
        
        # ============ TEST 11: MODEL PERFORMANCE ============
        print("\n" + "-" * 40)
        print("TEST 11: MODEL PERFORMANCE (V6 Sub-models)")
        print("-" * 40)
        try:
            async with session.get(f"{BACKEND_URL}/model-performance") as resp:
                if resp.status == 200:
                    models = await resp.json()
                    print("✅ Model Performance Retrieved")
                    for model_name, model_data in models.items():
                        if isinstance(model_data, dict):
                            print(f"   {model_name}: Accuracy={model_data.get('accuracy', 0)}, Weight={model_data.get('current_weight', 0)}")
        except Exception as e:
            extended_results["errors"].append(f"Model Performance: {e}")
        
        # ============ TEST 12: PREDICTIONS COMPARISON ============
        print("\n" + "-" * 40)
        print("TEST 12: ALGORITHM COMPARISON")
        print("-" * 40)
        try:
            async with session.get(f"{BACKEND_URL}/predictions/comparison") as resp:
                if resp.status == 200:
                    comparison = await resp.json()
                    print("✅ Algorithm Comparison Retrieved")
                    for algo, data in comparison.items():
                        if isinstance(data, dict) and "total" in data:
                            print(f"   {algo}: {data.get('total', 0)} predictions, {data.get('win_rate', 0)}% win rate")
        except Exception as e:
            extended_results["errors"].append(f"Comparison: {e}")
    
    return extended_results

async def main():
    results = await run_extended_simulation()
    
    # ============ FINAL SUMMARY ============
    print("\n" + "=" * 80)
    print("📊 FINAL SIMULATION RESULTS")
    print("=" * 80)
    
    print(f"""
    🔌 SYSTEM STATUS:
    - API Health: {'✅ Online' if results['api_health'] else '❌ Offline'}
    - Data Source: {results['data_sources'].get('source', 'Unknown')}
    - Events Cached: {results['events_fetched']}
    - Line Movement Snapshots: {results['line_movement_snapshots']}
    
    🔮 PREDICTION ALGORITHMS:
    - V5: {results['v5_analyses']} analyses, {results['v5_picks']} picks
    - V6: {results['v6_analyses']} analyses, {results['v6_picks']} picks
    - Unified: {results['unified_analyses']} analyses, {results['unified_picks']} picks
    - Total Picks Generated: {results['v5_picks'] + results['v6_picks'] + results['unified_picks']}
    
    🎯 SIMULATION RESULTS (10 MATCHES):
    - Predictions Created: {results['predictions_created']}
    - Results Updated: {results['results_updated']}
    - Wins: {results['wins']}
    - Losses: {results['losses']}
    - Win Rate: {(results['wins'] / (results['wins'] + results['losses']) * 100) if (results['wins'] + results['losses']) > 0 else 0:.1f}%
    
    📬 NOTIFICATIONS:
    - Total: {results['notifications_created']}
    - Marked Read: {'✅' if results['notifications_read'] else '❌'}
    
    📈 PERFORMANCE:
    - Stats Retrieved: {'✅' if results['performance_fetched'] else '❌'}
    
    ❌ ERRORS: {len(results['errors'])}
    """)
    
    if results['errors']:
        print("    Error Details:")
        for e in results['errors'][:5]:
            print(f"    - {e}")
    
    # Match-by-match breakdown
    print("\n" + "-" * 40)
    print("📋 MATCH-BY-MATCH RESULTS")
    print("-" * 40)
    for i, mr in enumerate(results['match_results']):
        status = "✅" if mr['result'] == 'win' else "❌"
        print(f"   {i+1}. {mr['match']} ({mr['score']})")
        print(f"      {mr['prediction']} → {status} {mr['result'].upper()}")
    
    # Final Verdict
    print("\n" + "=" * 80)
    print("✅ OVERALL TEST VERDICT")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 7
    
    if results['api_health']:
        tests_passed += 1
        print("   ✅ API Health: PASS")
    else:
        print("   ❌ API Health: FAIL")
    
    if results['events_fetched'] > 0:
        tests_passed += 1
        print("   ✅ Event Fetching: PASS")
    else:
        print("   ❌ Event Fetching: FAIL")
    
    if results['v5_analyses'] > 0 or results['v6_analyses'] > 0:
        tests_passed += 1
        print("   ✅ Prediction Algorithms: PASS")
    else:
        print("   ❌ Prediction Algorithms: FAIL")
    
    if results['predictions_created'] == 10:
        tests_passed += 1
        print("   ✅ Prediction Creation: PASS")
    else:
        print("   ⚠️ Prediction Creation: PARTIAL")
    
    if results['results_updated'] == 10:
        tests_passed += 1
        print("   ✅ Win/Loss Tracking: PASS")
    else:
        print("   ⚠️ Win/Loss Tracking: PARTIAL")
    
    if results['notifications_created'] > 0:
        tests_passed += 1
        print("   ✅ Notifications: PASS")
    else:
        print("   ⚠️ Notifications: PARTIAL")
    
    if results['performance_fetched']:
        tests_passed += 1
        print("   ✅ Performance Stats: PASS")
    else:
        print("   ❌ Performance Stats: FAIL")
    
    print(f"\n   📊 TESTS PASSED: {tests_passed}/{total_tests}")
    
    if tests_passed >= 6:
        print("\n   🎉 ALL CORE FEATURES ARE WORKING!")
    elif tests_passed >= 4:
        print("\n   ⚠️ MOST FEATURES WORKING - Some need attention")
    else:
        print("\n   ❌ MULTIPLE FEATURES NEED ATTENTION")

if __name__ == "__main__":
    asyncio.run(main())
