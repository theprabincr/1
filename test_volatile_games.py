#!/usr/bin/env python3
"""
BetPredictor Volatile Games Test
Tests with games that have clear mismatches to trigger predictor picks
"""

import asyncio
import sys
sys.path.insert(0, '/app/backend')

from datetime import datetime, timezone, timedelta
from betpredictor_v6 import generate_v6_prediction, BetPredictorV6
import uuid

# Volatile test scenarios - games with clear mismatches
VOLATILE_SCENARIOS = [
    {
        "name": "Heavy Favorite (Boston vs Charlotte)",
        "event": {
            "id": f"VOL_{uuid.uuid4().hex[:8]}",
            "espn_id": "vol_001",
            "sport_key": "basketball_nba",
            "home_team": "Boston Celtics",
            "away_team": "Charlotte Hornets",
            "commence_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "odds": {"spread": -12.5, "total": 225.5, "home_ml_decimal": 1.22, "away_ml_decimal": 4.50}
        },
        "matchup_data": {
            "home_team": {
                "name": "Boston Celtics",
                "form": {"wins": 45, "losses": 8, "win_pct": 0.85, "avg_margin": 10.5, "streak": 7},
                "recent_games": [{"won": True, "margin": 15}, {"won": True, "margin": 12}, {"won": True, "margin": 18}, {"won": True, "margin": 8}, {"won": True, "margin": 20}]
            },
            "away_team": {
                "name": "Charlotte Hornets",
                "form": {"wins": 15, "losses": 38, "win_pct": 0.28, "avg_margin": -8.5, "streak": -5},
                "recent_games": [{"won": False, "margin": -12}, {"won": False, "margin": -18}, {"won": False, "margin": -15}, {"won": False, "margin": -10}, {"won": False, "margin": -20}]
            }
        },
        "squad_data": {
            "home_team": {"injuries": []},
            "away_team": {"injuries": [{"name": "LaMelo Ball", "position": "PG", "status": "Out"}, {"name": "Brandon Miller", "position": "SF", "status": "Out"}]}
        },
        "line_movement": [
            {"home_ml": 1.28, "away_ml": 4.00, "spread": -10.5, "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
            {"home_ml": 1.25, "away_ml": 4.25, "spread": -11.5, "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
            {"home_ml": 1.22, "away_ml": 4.50, "spread": -12.5, "timestamp": datetime.now(timezone.utc).isoformat()}
        ]
    },
    {
        "name": "Hot Team vs Cold Team (Milwaukee vs Detroit)",
        "event": {
            "id": f"VOL_{uuid.uuid4().hex[:8]}",
            "espn_id": "vol_002",
            "sport_key": "basketball_nba",
            "home_team": "Milwaukee Bucks",
            "away_team": "Detroit Pistons",
            "commence_time": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
            "odds": {"spread": -14.0, "total": 230.0, "home_ml_decimal": 1.15, "away_ml_decimal": 5.50}
        },
        "matchup_data": {
            "home_team": {
                "name": "Milwaukee Bucks",
                "form": {"wins": 42, "losses": 12, "win_pct": 0.78, "avg_margin": 9.0, "streak": 10},
                "recent_games": [{"won": True, "margin": 22}, {"won": True, "margin": 15}, {"won": True, "margin": 18}, {"won": True, "margin": 12}, {"won": True, "margin": 25}]
            },
            "away_team": {
                "name": "Detroit Pistons",
                "form": {"wins": 8, "losses": 46, "win_pct": 0.15, "avg_margin": -15.0, "streak": -12},
                "recent_games": [{"won": False, "margin": -25}, {"won": False, "margin": -18}, {"won": False, "margin": -22}, {"won": False, "margin": -20}, {"won": False, "margin": -15}]
            }
        },
        "squad_data": {
            "home_team": {"injuries": []},
            "away_team": {"injuries": [{"name": "Cade Cunningham", "position": "PG", "status": "Out"}]}
        },
        "line_movement": [
            {"home_ml": 1.20, "away_ml": 5.00, "spread": -12.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
            {"home_ml": 1.18, "away_ml": 5.25, "spread": -13.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
            {"home_ml": 1.15, "away_ml": 5.50, "spread": -14.0, "timestamp": datetime.now(timezone.utc).isoformat()}
        ]
    },
    {
        "name": "Revenge Game (LA Lakers vs Phoenix)",
        "event": {
            "id": f"VOL_{uuid.uuid4().hex[:8]}",
            "espn_id": "vol_003",
            "sport_key": "basketball_nba",
            "home_team": "Los Angeles Lakers",
            "away_team": "Phoenix Suns",
            "commence_time": (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat(),
            "odds": {"spread": -3.5, "total": 232.0, "home_ml_decimal": 1.55, "away_ml_decimal": 2.50}
        },
        "matchup_data": {
            "home_team": {
                "name": "Los Angeles Lakers",
                "form": {"wins": 35, "losses": 20, "win_pct": 0.64, "avg_margin": 4.5, "streak": 3},
                "recent_games": [{"won": True, "margin": 8}, {"won": True, "margin": 5}, {"won": True, "margin": 12}, {"won": False, "margin": -3}, {"won": True, "margin": 7}]
            },
            "away_team": {
                "name": "Phoenix Suns",
                "form": {"wins": 28, "losses": 27, "win_pct": 0.51, "avg_margin": 0.5, "streak": -2},
                "recent_games": [{"won": False, "margin": -5}, {"won": False, "margin": -8}, {"won": True, "margin": 3}, {"won": False, "margin": -10}, {"won": True, "margin": 2}]
            }
        },
        "squad_data": {
            "home_team": {"injuries": []},
            "away_team": {"injuries": [{"name": "Kevin Durant", "position": "SF", "status": "Questionable"}]}
        },
        "line_movement": [
            {"home_ml": 1.65, "away_ml": 2.35, "spread": -2.5, "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
            {"home_ml": 1.60, "away_ml": 2.40, "spread": -3.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
            {"home_ml": 1.55, "away_ml": 2.50, "spread": -3.5, "timestamp": datetime.now(timezone.utc).isoformat()}
        ]
    },
    {
        "name": "Back-to-Back Fatigue (Denver vs Utah)",
        "event": {
            "id": f"VOL_{uuid.uuid4().hex[:8]}",
            "espn_id": "vol_004",
            "sport_key": "basketball_nba",
            "home_team": "Denver Nuggets",
            "away_team": "Utah Jazz",
            "commence_time": (datetime.now(timezone.utc) + timedelta(hours=1.5)).isoformat(),
            "odds": {"spread": -10.5, "total": 228.0, "home_ml_decimal": 1.30, "away_ml_decimal": 3.50}
        },
        "matchup_data": {
            "home_team": {
                "name": "Denver Nuggets",
                "form": {"wins": 40, "losses": 15, "win_pct": 0.73, "avg_margin": 7.5, "streak": 5},
                "recent_games": [{"won": True, "margin": 10, "date": "2026-01-25"}, {"won": True, "margin": 8, "date": "2026-01-23"}, {"won": True, "margin": 15, "date": "2026-01-21"}, {"won": True, "margin": 12, "date": "2026-01-19"}, {"won": True, "margin": 6, "date": "2026-01-17"}]
            },
            "away_team": {
                "name": "Utah Jazz",
                "form": {"wins": 20, "losses": 35, "win_pct": 0.36, "avg_margin": -5.0, "streak": -3},
                "recent_games": [{"won": False, "margin": -8, "date": "2026-01-28"}, {"won": False, "margin": -12, "date": "2026-01-27"}, {"won": False, "margin": -5, "date": "2026-01-25"}, {"won": True, "margin": 3, "date": "2026-01-23"}, {"won": False, "margin": -10, "date": "2026-01-21"}]
            }
        },
        "squad_data": {
            "home_team": {"injuries": []},
            "away_team": {"injuries": []}
        },
        "line_movement": [
            {"home_ml": 1.35, "away_ml": 3.20, "spread": -9.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
            {"home_ml": 1.32, "away_ml": 3.35, "spread": -9.5, "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
            {"home_ml": 1.30, "away_ml": 3.50, "spread": -10.5, "timestamp": datetime.now(timezone.utc).isoformat()}
        ]
    },
    {
        "name": "Elite vs Tanking (Cleveland vs Washington)",
        "event": {
            "id": f"VOL_{uuid.uuid4().hex[:8]}",
            "espn_id": "vol_005",
            "sport_key": "basketball_nba",
            "home_team": "Cleveland Cavaliers",
            "away_team": "Washington Wizards",
            "commence_time": (datetime.now(timezone.utc) + timedelta(hours=2.5)).isoformat(),
            "odds": {"spread": -15.5, "total": 222.0, "home_ml_decimal": 1.12, "away_ml_decimal": 6.00}
        },
        "matchup_data": {
            "home_team": {
                "name": "Cleveland Cavaliers",
                "form": {"wins": 48, "losses": 8, "win_pct": 0.86, "avg_margin": 12.0, "streak": 8},
                "recent_games": [{"won": True, "margin": 18}, {"won": True, "margin": 15}, {"won": True, "margin": 22}, {"won": True, "margin": 10}, {"won": True, "margin": 25}]
            },
            "away_team": {
                "name": "Washington Wizards",
                "form": {"wins": 5, "losses": 50, "win_pct": 0.09, "avg_margin": -18.0, "streak": -15},
                "recent_games": [{"won": False, "margin": -20}, {"won": False, "margin": -25}, {"won": False, "margin": -18}, {"won": False, "margin": -22}, {"won": False, "margin": -15}]
            }
        },
        "squad_data": {
            "home_team": {"injuries": []},
            "away_team": {"injuries": [{"name": "Jordan Poole", "position": "SG", "status": "Out"}, {"name": "Kyle Kuzma", "position": "PF", "status": "Out"}]}
        },
        "line_movement": [
            {"home_ml": 1.18, "away_ml": 5.50, "spread": -13.5, "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
            {"home_ml": 1.15, "away_ml": 5.75, "spread": -14.5, "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
            {"home_ml": 1.12, "away_ml": 6.00, "spread": -15.5, "timestamp": datetime.now(timezone.utc).isoformat()}
        ]
    }
]

async def run_volatile_tests():
    """Run V6 analysis on volatile game scenarios"""
    print("\n" + "=" * 80)
    print("🔥 BETPREDICTOR VOLATILE GAMES TEST")
    print("=" * 80)
    print("Testing scenarios designed to trigger predictor picks...\n")
    
    picks_generated = 0
    no_picks = 0
    results = []
    
    for i, scenario in enumerate(VOLATILE_SCENARIOS):
        print(f"\n{'='*60}")
        print(f"📊 SCENARIO {i+1}: {scenario['name']}")
        print(f"{'='*60}")
        
        event = scenario["event"]
        print(f"   {event['home_team']} vs {event['away_team']}")
        print(f"   Spread: {event['odds']['spread']}")
        print(f"   Home ML: {event['odds']['home_ml_decimal']} | Away ML: {event['odds']['away_ml_decimal']}")
        
        # Build opening odds from line movement
        opening_odds = {
            "home_ml": scenario["line_movement"][0]["home_ml"],
            "away_ml": scenario["line_movement"][0]["away_ml"],
            "spread": scenario["line_movement"][0]["spread"],
            "timestamp": scenario["line_movement"][0]["timestamp"]
        }
        
        current_odds = {
            "home_ml_decimal": event["odds"]["home_ml_decimal"],
            "away_ml_decimal": event["odds"]["away_ml_decimal"],
            "spread": event["odds"]["spread"],
            "total": event["odds"]["total"]
        }
        
        try:
            prediction = await generate_v6_prediction(
                event=event,
                sport_key="basketball_nba",
                squad_data=scenario["squad_data"],
                matchup_data=scenario["matchup_data"],
                line_movement_history=scenario["line_movement"],
                opening_odds=opening_odds,
                current_odds=current_odds
            )
            
            if prediction.get("has_pick"):
                picks_generated += 1
                result = {
                    "scenario": scenario["name"],
                    "pick": prediction.get("pick"),
                    "pick_type": prediction.get("pick_type"),
                    "confidence": prediction.get("confidence"),
                    "edge": prediction.get("edge"),
                    "model_agreement": prediction.get("model_agreement"),
                    "status": "PICK"
                }
                
                print(f"\n   ✅ PICK GENERATED!")
                print(f"   Pick: {prediction.get('pick')} ({prediction.get('pick_type')})")
                print(f"   Confidence: {prediction.get('confidence')}%")
                print(f"   Edge: {prediction.get('edge')}%")
                print(f"   Model Agreement: {prediction.get('model_agreement')}%")
                
                # Show model breakdown
                ensemble = prediction.get("ensemble_details", {})
                individual = ensemble.get("individual_predictions", {})
                if individual:
                    print(f"\n   Model Breakdown:")
                    for model_name, model_pred in individual.items():
                        pick = model_pred.get("pick", "None")
                        prob = model_pred.get("probability", 0) * 100
                        print(f"     {model_name}: {prob:.1f}% → {pick}")
            else:
                no_picks += 1
                result = {
                    "scenario": scenario["name"],
                    "pick": None,
                    "confidence": prediction.get("ensemble_confidence"),
                    "model_agreement": prediction.get("model_agreement"),
                    "reason": prediction.get("reasoning", "")[:100],
                    "status": "NO PICK"
                }
                
                print(f"\n   ⏸️ NO PICK")
                print(f"   Ensemble Confidence: {prediction.get('ensemble_confidence')}%")
                print(f"   Model Agreement: {prediction.get('model_agreement')}%")
                print(f"   Reason: {prediction.get('reasoning', '')[:80]}...")
            
            results.append(result)
            
        except Exception as e:
            print(f"\n   ❌ ERROR: {e}")
            results.append({"scenario": scenario["name"], "status": "ERROR", "error": str(e)})
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 VOLATILE GAMES TEST SUMMARY")
    print("=" * 80)
    
    print(f"\n   Total Scenarios Tested: {len(VOLATILE_SCENARIOS)}")
    print(f"   Picks Generated: {picks_generated}")
    print(f"   No Picks: {no_picks}")
    print(f"   Pick Rate: {(picks_generated / len(VOLATILE_SCENARIOS) * 100):.1f}%")
    
    print("\n   Results by Scenario:")
    print("-" * 60)
    for r in results:
        status_emoji = "✅" if r["status"] == "PICK" else "⏸️" if r["status"] == "NO PICK" else "❌"
        if r["status"] == "PICK":
            print(f"   {status_emoji} {r['scenario']}: {r['pick']} ({r['confidence']}% conf)")
        else:
            print(f"   {status_emoji} {r['scenario']}: {r['status']}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_volatile_tests())
