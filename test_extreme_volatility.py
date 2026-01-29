#!/usr/bin/env python3
"""
BetPredictor EXTREME Volatile Games Test
Creates scenarios with MAXIMUM edge to guarantee predictor picks
"""

import asyncio
import sys
sys.path.insert(0, '/app/backend')

from datetime import datetime, timezone, timedelta
from betpredictor_v6 import generate_v6_prediction
import uuid

# EXTREME scenarios - designed to MAXIMIZE model agreement and confidence
EXTREME_SCENARIOS = [
    {
        "name": "🔥 BLOWOUT: #1 Team vs #30 Team (Max Mismatch)",
        "event": {
            "id": f"EXT_{uuid.uuid4().hex[:8]}",
            "espn_id": "ext_001",
            "sport_key": "basketball_nba",
            "home_team": "Cleveland Cavaliers",
            "away_team": "Washington Wizards",
            "commence_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "odds": {"spread": -18.5, "total": 218.0, "home_ml_decimal": 1.05, "away_ml_decimal": 12.00}
        },
        "matchup_data": {
            "home_team": {
                "name": "Cleveland Cavaliers",
                "form": {"wins": 52, "losses": 5, "win_pct": 0.91, "avg_margin": 15.5, "streak": 15},
                "recent_games": [
                    {"won": True, "margin": 25, "date": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
                    {"won": True, "margin": 22, "date": (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()},
                    {"won": True, "margin": 30, "date": (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()},
                    {"won": True, "margin": 18, "date": (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()},
                    {"won": True, "margin": 28, "date": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()}
                ]
            },
            "away_team": {
                "name": "Washington Wizards",
                "form": {"wins": 3, "losses": 54, "win_pct": 0.05, "avg_margin": -22.0, "streak": -25},
                "recent_games": [
                    {"won": False, "margin": -30, "date": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
                    {"won": False, "margin": -28, "date": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
                    {"won": False, "margin": -35, "date": (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()},
                    {"won": False, "margin": -25, "date": (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()},
                    {"won": False, "margin": -32, "date": (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()}
                ]
            }
        },
        "squad_data": {
            "home_team": {"injuries": []},
            "away_team": {"injuries": [
                {"name": "Kyle Kuzma", "position": "Power Forward", "status": "Out", "details": {"type": "Knee"}},
                {"name": "Jordan Poole", "position": "Shooting Guard", "status": "Out", "details": {"type": "Ankle"}},
                {"name": "Tyus Jones", "position": "Point Guard", "status": "Out", "details": {"type": "Back"}}
            ]}
        },
        "line_movement": [
            {"home_ml": 1.12, "away_ml": 8.00, "spread": -15.5, "total": 220.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()},
            {"home_ml": 1.08, "away_ml": 10.00, "spread": -17.0, "total": 219.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
            {"home_ml": 1.06, "away_ml": 11.00, "spread": -18.0, "total": 218.5, "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
            {"home_ml": 1.05, "away_ml": 12.00, "spread": -18.5, "total": 218.0, "timestamp": datetime.now(timezone.utc).isoformat()}
        ]
    },
    {
        "name": "🏠 HOME DOMINATOR: Unbeaten at Home Team",
        "event": {
            "id": f"EXT_{uuid.uuid4().hex[:8]}",
            "espn_id": "ext_002",
            "sport_key": "basketball_nba",
            "home_team": "Boston Celtics",
            "away_team": "Portland Trail Blazers",
            "commence_time": (datetime.now(timezone.utc) + timedelta(hours=1.5)).isoformat(),
            "odds": {"spread": -16.5, "total": 222.0, "home_ml_decimal": 1.08, "away_ml_decimal": 9.00}
        },
        "matchup_data": {
            "home_team": {
                "name": "Boston Celtics",
                "form": {"wins": 48, "losses": 8, "win_pct": 0.86, "avg_margin": 12.0, "streak": 12},
                "recent_games": [
                    {"won": True, "margin": 20, "date": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
                    {"won": True, "margin": 18, "date": (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()},
                    {"won": True, "margin": 25, "date": (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()},
                    {"won": True, "margin": 15, "date": (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()},
                    {"won": True, "margin": 22, "date": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()}
                ]
            },
            "away_team": {
                "name": "Portland Trail Blazers",
                "form": {"wins": 12, "losses": 44, "win_pct": 0.21, "avg_margin": -12.0, "streak": -8},
                "recent_games": [
                    {"won": False, "margin": -18, "date": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
                    {"won": False, "margin": -15, "date": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
                    {"won": False, "margin": -22, "date": (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()},
                    {"won": False, "margin": -12, "date": (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()},
                    {"won": False, "margin": -20, "date": (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()}
                ]
            }
        },
        "squad_data": {
            "home_team": {"injuries": []},
            "away_team": {"injuries": [
                {"name": "Anfernee Simons", "position": "Point Guard", "status": "Out", "details": {"type": "Hamstring"}}
            ]}
        },
        "line_movement": [
            {"home_ml": 1.15, "away_ml": 6.50, "spread": -13.5, "total": 224.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()},
            {"home_ml": 1.12, "away_ml": 7.50, "spread": -14.5, "total": 223.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
            {"home_ml": 1.10, "away_ml": 8.25, "spread": -15.5, "total": 222.5, "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
            {"home_ml": 1.08, "away_ml": 9.00, "spread": -16.5, "total": 222.0, "timestamp": datetime.now(timezone.utc).isoformat()}
        ]
    },
    {
        "name": "😴 FATIGUE SPECIAL: Well-Rested vs 4th Game in 5 Days",
        "event": {
            "id": f"EXT_{uuid.uuid4().hex[:8]}",
            "espn_id": "ext_003",
            "sport_key": "basketball_nba",
            "home_team": "Denver Nuggets",
            "away_team": "San Antonio Spurs",
            "commence_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "odds": {"spread": -14.5, "total": 226.0, "home_ml_decimal": 1.10, "away_ml_decimal": 7.50}
        },
        "matchup_data": {
            "home_team": {
                "name": "Denver Nuggets",
                "form": {"wins": 45, "losses": 12, "win_pct": 0.79, "avg_margin": 10.0, "streak": 6},
                "recent_games": [
                    {"won": True, "margin": 15, "date": (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()},
                    {"won": True, "margin": 12, "date": (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()},
                    {"won": True, "margin": 18, "date": (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()},
                    {"won": True, "margin": 8, "date": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()},
                    {"won": True, "margin": 22, "date": (datetime.now(timezone.utc) - timedelta(days=12)).isoformat()}
                ]
            },
            "away_team": {
                "name": "San Antonio Spurs",
                "form": {"wins": 18, "losses": 38, "win_pct": 0.32, "avg_margin": -8.0, "streak": -4},
                "recent_games": [
                    {"won": False, "margin": -12, "date": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
                    {"won": False, "margin": -8, "date": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
                    {"won": False, "margin": -15, "date": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()},
                    {"won": False, "margin": -10, "date": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()},
                    {"won": True, "margin": 5, "date": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()}
                ]
            }
        },
        "squad_data": {
            "home_team": {"injuries": []},
            "away_team": {"injuries": [
                {"name": "Victor Wembanyama", "position": "Center", "status": "Questionable", "details": {"type": "Rest"}}
            ]}
        },
        "line_movement": [
            {"home_ml": 1.18, "away_ml": 5.50, "spread": -11.5, "total": 228.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()},
            {"home_ml": 1.15, "away_ml": 6.25, "spread": -12.5, "total": 227.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
            {"home_ml": 1.12, "away_ml": 7.00, "spread": -13.5, "total": 226.5, "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
            {"home_ml": 1.10, "away_ml": 7.50, "spread": -14.5, "total": 226.0, "timestamp": datetime.now(timezone.utc).isoformat()}
        ]
    },
    {
        "name": "💪 REVENGE + INJURIES: Strong vs Depleted Rival",
        "event": {
            "id": f"EXT_{uuid.uuid4().hex[:8]}",
            "espn_id": "ext_004",
            "sport_key": "basketball_nba",
            "home_team": "Milwaukee Bucks",
            "away_team": "Detroit Pistons",
            "commence_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "odds": {"spread": -17.5, "total": 224.0, "home_ml_decimal": 1.06, "away_ml_decimal": 10.00}
        },
        "matchup_data": {
            "home_team": {
                "name": "Milwaukee Bucks",
                "form": {"wins": 46, "losses": 10, "win_pct": 0.82, "avg_margin": 11.0, "streak": 10},
                "recent_games": [
                    {"won": True, "margin": 22, "date": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
                    {"won": True, "margin": 18, "date": (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()},
                    {"won": True, "margin": 25, "date": (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()},
                    {"won": True, "margin": 15, "date": (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()},
                    {"won": True, "margin": 20, "date": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()}
                ]
            },
            "away_team": {
                "name": "Detroit Pistons",
                "form": {"wins": 6, "losses": 50, "win_pct": 0.11, "avg_margin": -18.0, "streak": -18},
                "recent_games": [
                    {"won": False, "margin": -25, "date": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
                    {"won": False, "margin": -22, "date": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
                    {"won": False, "margin": -28, "date": (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()},
                    {"won": False, "margin": -20, "date": (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()},
                    {"won": False, "margin": -30, "date": (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()}
                ]
            }
        },
        "squad_data": {
            "home_team": {"injuries": []},
            "away_team": {"injuries": [
                {"name": "Cade Cunningham", "position": "Point Guard", "status": "Out", "details": {"type": "Knee"}},
                {"name": "Jaden Ivey", "position": "Shooting Guard", "status": "Out", "details": {"type": "Ankle"}}
            ]}
        },
        "line_movement": [
            {"home_ml": 1.12, "away_ml": 7.00, "spread": -14.5, "total": 226.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()},
            {"home_ml": 1.10, "away_ml": 8.00, "spread": -15.5, "total": 225.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
            {"home_ml": 1.08, "away_ml": 9.00, "spread": -16.5, "total": 224.5, "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
            {"home_ml": 1.06, "away_ml": 10.00, "spread": -17.5, "total": 224.0, "timestamp": datetime.now(timezone.utc).isoformat()}
        ]
    },
    {
        "name": "📈 SHARP MONEY FLOOD: Line Moving Hard One Direction",
        "event": {
            "id": f"EXT_{uuid.uuid4().hex[:8]}",
            "espn_id": "ext_005",
            "sport_key": "basketball_nba",
            "home_team": "Oklahoma City Thunder",
            "away_team": "Utah Jazz",
            "commence_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "odds": {"spread": -15.5, "total": 220.0, "home_ml_decimal": 1.08, "away_ml_decimal": 9.50}
        },
        "matchup_data": {
            "home_team": {
                "name": "Oklahoma City Thunder",
                "form": {"wins": 50, "losses": 7, "win_pct": 0.88, "avg_margin": 13.0, "streak": 8},
                "recent_games": [
                    {"won": True, "margin": 18, "date": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
                    {"won": True, "margin": 22, "date": (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()},
                    {"won": True, "margin": 15, "date": (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()},
                    {"won": True, "margin": 28, "date": (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()},
                    {"won": True, "margin": 12, "date": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()}
                ]
            },
            "away_team": {
                "name": "Utah Jazz",
                "form": {"wins": 15, "losses": 42, "win_pct": 0.26, "avg_margin": -10.0, "streak": -6},
                "recent_games": [
                    {"won": False, "margin": -15, "date": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
                    {"won": False, "margin": -18, "date": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()},
                    {"won": False, "margin": -12, "date": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()},
                    {"won": False, "margin": -20, "date": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()},
                    {"won": False, "margin": -10, "date": (datetime.now(timezone.utc) - timedelta(days=9)).isoformat()}
                ]
            }
        },
        "squad_data": {
            "home_team": {"injuries": []},
            "away_team": {"injuries": [
                {"name": "Lauri Markkanen", "position": "Power Forward", "status": "Out", "details": {"type": "Back"}}
            ]}
        },
        "line_movement": [
            {"home_ml": 1.25, "away_ml": 4.00, "spread": -8.5, "total": 224.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()},
            {"home_ml": 1.18, "away_ml": 5.50, "spread": -11.0, "total": 222.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
            {"home_ml": 1.12, "away_ml": 7.50, "spread": -13.5, "total": 221.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
            {"home_ml": 1.08, "away_ml": 9.50, "spread": -15.5, "total": 220.0, "timestamp": datetime.now(timezone.utc).isoformat()}
        ]
    },
    {
        "name": "🎯 PERFECT STORM: All Factors Aligned",
        "event": {
            "id": f"EXT_{uuid.uuid4().hex[:8]}",
            "espn_id": "ext_006",
            "sport_key": "basketball_nba",
            "home_team": "Golden State Warriors",
            "away_team": "Charlotte Hornets",
            "commence_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "odds": {"spread": -13.5, "total": 225.0, "home_ml_decimal": 1.12, "away_ml_decimal": 6.50}
        },
        "matchup_data": {
            "home_team": {
                "name": "Golden State Warriors",
                "form": {"wins": 42, "losses": 15, "win_pct": 0.74, "avg_margin": 8.5, "streak": 7},
                "recent_games": [
                    {"won": True, "margin": 15, "date": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()},
                    {"won": True, "margin": 12, "date": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()},
                    {"won": True, "margin": 18, "date": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()},
                    {"won": True, "margin": 10, "date": (datetime.now(timezone.utc) - timedelta(days=9)).isoformat()},
                    {"won": True, "margin": 14, "date": (datetime.now(timezone.utc) - timedelta(days=11)).isoformat()}
                ]
            },
            "away_team": {
                "name": "Charlotte Hornets",
                "form": {"wins": 10, "losses": 47, "win_pct": 0.18, "avg_margin": -14.0, "streak": -10},
                "recent_games": [
                    {"won": False, "margin": -20, "date": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
                    {"won": False, "margin": -18, "date": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
                    {"won": False, "margin": -15, "date": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()},
                    {"won": False, "margin": -22, "date": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()},
                    {"won": False, "margin": -12, "date": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()}
                ]
            }
        },
        "squad_data": {
            "home_team": {"injuries": []},
            "away_team": {"injuries": [
                {"name": "LaMelo Ball", "position": "Point Guard", "status": "Out", "details": {"type": "Ankle"}},
                {"name": "Brandon Miller", "position": "Small Forward", "status": "Out", "details": {"type": "Hip"}},
                {"name": "Mark Williams", "position": "Center", "status": "Out", "details": {"type": "Foot"}}
            ]}
        },
        "line_movement": [
            {"home_ml": 1.22, "away_ml": 4.50, "spread": -9.5, "total": 228.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()},
            {"home_ml": 1.18, "away_ml": 5.25, "spread": -11.0, "total": 227.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
            {"home_ml": 1.15, "away_ml": 5.75, "spread": -12.0, "total": 226.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
            {"home_ml": 1.12, "away_ml": 6.50, "spread": -13.5, "total": 225.0, "timestamp": datetime.now(timezone.utc).isoformat()}
        ]
    }
]

async def run_extreme_tests():
    """Run V6 analysis on EXTREME scenarios designed to trigger picks"""
    print("\n" + "=" * 80)
    print("🔥🔥🔥 BETPREDICTOR EXTREME VOLATILITY TEST 🔥🔥🔥")
    print("=" * 80)
    print("Testing scenarios with MAXIMUM edge to guarantee predictor picks...\n")
    
    picks_generated = 0
    no_picks = 0
    results = []
    
    for i, scenario in enumerate(EXTREME_SCENARIOS):
        print(f"\n{'='*70}")
        print(f"📊 SCENARIO {i+1}/{len(EXTREME_SCENARIOS)}: {scenario['name']}")
        print(f"{'='*70}")
        
        event = scenario["event"]
        home_form = scenario["matchup_data"]["home_team"]["form"]
        away_form = scenario["matchup_data"]["away_team"]["form"]
        
        print(f"\n   🏀 MATCHUP:")
        print(f"   {event['home_team']} ({home_form['wins']}-{home_form['losses']}, {home_form['win_pct']*100:.0f}%)")
        print(f"   vs {event['away_team']} ({away_form['wins']}-{away_form['losses']}, {away_form['win_pct']*100:.0f}%)")
        print(f"\n   📊 ODDS:")
        print(f"   Spread: {event['odds']['spread']} | ML: {event['odds']['home_ml_decimal']} / {event['odds']['away_ml_decimal']}")
        
        # Injuries
        away_injuries = scenario["squad_data"]["away_team"]["injuries"]
        if away_injuries:
            print(f"\n   🤕 AWAY INJURIES ({len(away_injuries)} players):")
            for inj in away_injuries:
                print(f"      • {inj['name']} ({inj['position']}) - {inj['status']}")
        
        # Line movement
        first_line = scenario["line_movement"][0]
        last_line = scenario["line_movement"][-1]
        spread_move = last_line["spread"] - first_line["spread"]
        print(f"\n   📈 LINE MOVEMENT:")
        print(f"   Spread: {first_line['spread']} → {last_line['spread']} ({spread_move:+.1f})")
        
        # Build opening odds
        opening_odds = {
            "home_ml": first_line["home_ml"],
            "away_ml": first_line["away_ml"],
            "spread": first_line["spread"],
            "total": first_line.get("total", 220),
            "timestamp": first_line["timestamp"]
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
            
            has_pick = prediction.get("has_pick", False)
            
            if has_pick:
                picks_generated += 1
                result = {
                    "scenario": scenario["name"],
                    "matchup": f"{event['home_team']} vs {event['away_team']}",
                    "pick": prediction.get("pick"),
                    "pick_type": prediction.get("pick_type"),
                    "confidence": prediction.get("confidence"),
                    "edge": prediction.get("edge"),
                    "model_agreement": prediction.get("model_agreement"),
                    "status": "✅ PICK"
                }
                
                print(f"\n   ✅ ✅ ✅ PICK GENERATED! ✅ ✅ ✅")
                print(f"   Pick: {prediction.get('pick')} ({prediction.get('pick_type')})")
                print(f"   Confidence: {prediction.get('confidence')}%")
                print(f"   Edge: {prediction.get('edge')}%")
                print(f"   Model Agreement: {prediction.get('model_agreement')}%")
                
                # Model breakdown
                ensemble = prediction.get("ensemble_details", {})
                individual = ensemble.get("individual_predictions", {})
                if individual:
                    print(f"\n   📊 MODEL VOTES:")
                    for model_name, model_pred in individual.items():
                        pick = model_pred.get("pick", "None")
                        prob = model_pred.get("probability", 0) * 100
                        conf = model_pred.get("confidence", 0)
                        vote = "✅" if pick and pick != "None" else "⏸️"
                        print(f"      {vote} {model_name}: {prob:.1f}% prob, {conf:.0f}% conf → {pick}")
            else:
                no_picks += 1
                result = {
                    "scenario": scenario["name"],
                    "matchup": f"{event['home_team']} vs {event['away_team']}",
                    "pick": None,
                    "confidence": prediction.get("ensemble_confidence"),
                    "model_agreement": prediction.get("model_agreement"),
                    "reason": prediction.get("reasoning", "")[:80],
                    "status": "⏸️ NO PICK"
                }
                
                print(f"\n   ⏸️ NO PICK GENERATED")
                print(f"   Ensemble Confidence: {prediction.get('ensemble_confidence')}%")
                print(f"   Model Agreement: {prediction.get('model_agreement')}%")
                print(f"   Reason: {prediction.get('reasoning', '')[:60]}...")
            
            results.append(result)
            
        except Exception as e:
            print(f"\n   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"scenario": scenario["name"], "status": "❌ ERROR", "error": str(e)})
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 80)
    print("📊 EXTREME VOLATILITY TEST SUMMARY")
    print("=" * 80)
    
    print(f"\n   📈 STATISTICS:")
    print(f"   Total Scenarios: {len(EXTREME_SCENARIOS)}")
    print(f"   Picks Generated: {picks_generated}")
    print(f"   No Picks: {no_picks}")
    print(f"   Pick Rate: {(picks_generated / len(EXTREME_SCENARIOS) * 100):.1f}%")
    
    print(f"\n   📋 RESULTS BY SCENARIO:")
    print("-" * 70)
    for r in results:
        if r["status"] == "✅ PICK":
            print(f"   {r['status']} {r['matchup']}")
            print(f"         → {r['pick']} ({r['confidence']}% conf, {r['edge']}% edge)")
        else:
            print(f"   {r['status']} {r['matchup']}")
            if 'reason' in r:
                print(f"         Reason: {r['reason'][:50]}...")
    
    if picks_generated > 0:
        print(f"\n   🎯 PICKS READY FOR BETTING:")
        print("-" * 70)
        for r in results:
            if r["status"] == "✅ PICK":
                print(f"   • {r['pick']} @ {r['confidence']}% confidence")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_extreme_tests())
