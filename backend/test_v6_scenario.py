"""
V6 Algorithm Test Scenario Generator
Creates synthetic game data that meets all V6 prediction criteria
"""
import asyncio
import sys
import os
sys.path.insert(0, '/app/backend')

from datetime import datetime, timezone, timedelta
from betpredictor_v6 import generate_v6_prediction

# Create a PERFECT test scenario for V6 to make a pick
async def create_test_scenario():
    """
    Test Case: Boston Celtics (Strong) vs Charlotte Hornets (Weak)
    
    Scenario Design:
    - Clear ELO advantage: 200+ points
    - Recent form advantage: 80% vs 20%
    - Rest advantage: Celtics well-rested, Hornets on B2B
    - Home court advantage
    - Key injuries for Hornets
    - Aligned model predictions (70-75% range)
    """
    
    print("\n" + "="*80)
    print("V6 ALGORITHM TEST SCENARIO")
    print("="*80)
    
    # Test Event
    test_event = {
        "id": "TEST_001",
        "espn_id": "TEST_001",
        "sport_key": "basketball_nba",
        "home_team": "Boston Celtics",
        "away_team": "Charlotte Hornets",
        "home_team_id": "2",
        "away_team_id": "30",
        "commence_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "odds": {
            "spread": -12.5,
            "total": 225.5,
            "home_ml_decimal": 1.25,  # Celtics heavy favorite
            "away_ml_decimal": 4.00,
            "home_ml_american": "-400",
            "away_ml_american": "+300"
        },
        "bookmakers": [{
            "key": "draft_kings",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": 1.25},
                        {"name": "Charlotte Hornets", "price": 4.00}
                    ]
                }
            ]
        }]
    }
    
    # Squad Data with MAJOR injury for Hornets
    squad_data = {
        "home_team": {
            "injuries": []  # Celtics fully healthy
        },
        "away_team": {
            "injuries": [
                {
                    "name": "LaMelo Ball",
                    "position": "Point Guard",
                    "status": "Out",
                    "details": {"type": "Ankle"}
                },
                {
                    "name": "Brandon Miller", 
                    "position": "Forward",
                    "status": "Out",
                    "details": {"type": "Hip"}
                }
            ]
        }
    }
    
    # Matchup Data - STRONG CELTICS vs WEAK HORNETS
    matchup_data = {
        "home_team": {
            "name": "Boston Celtics",
            "form": {
                "wins": 40,
                "losses": 10,
                "win_pct": 0.80,  # 80% win rate
                "avg_margin": 8.5,
                "streak": 5  # 5-game win streak
            },
            "recent_games": [
                {"won": True, "our_score": 125, "opponent_score": 110, "margin": 15, "date": "2026-01-27T00:00Z"},
                {"won": True, "our_score": 118, "opponent_score": 105, "margin": 13, "date": "2026-01-25T00:00Z"},
                {"won": True, "our_score": 130, "opponent_score": 115, "margin": 15, "date": "2026-01-23T00:00Z"},
                {"won": True, "our_score": 122, "opponent_score": 108, "margin": 14, "date": "2026-01-21T00:00Z"},
                {"won": True, "our_score": 128, "opponent_score": 112, "margin": 16, "date": "2026-01-19T00:00Z"},
                {"won": False, "our_score": 105, "opponent_score": 110, "margin": -5, "date": "2026-01-17T00:00Z"},
                {"won": True, "our_score": 120, "opponent_score": 108, "margin": 12, "date": "2026-01-15T00:00Z"},
            ]
        },
        "away_team": {
            "name": "Charlotte Hornets",
            "form": {
                "wins": 10,
                "losses": 40,
                "win_pct": 0.20,  # 20% win rate
                "avg_margin": -7.2,
                "streak": -6  # 6-game losing streak
            },
            "recent_games": [
                {"won": False, "our_score": 95, "opponent_score": 110, "margin": -15, "date": "2026-01-28T00:00Z"},  # LAST NIGHT (B2B)
                {"won": False, "our_score": 98, "opponent_score": 115, "margin": -17, "date": "2026-01-26T00:00Z"},
                {"won": False, "our_score": 102, "opponent_score": 118, "margin": -16, "date": "2026-01-24T00:00Z"},
                {"won": False, "our_score": 88, "opponent_score": 105, "margin": -17, "date": "2026-01-22T00:00Z"},
                {"won": False, "our_score": 92, "opponent_score": 108, "margin": -16, "date": "2026-01-20T00:00Z"},
                {"won": False, "our_score": 100, "opponent_score": 112, "margin": -12, "date": "2026-01-18T00:00Z"},
                {"won": True, "our_score": 110, "opponent_score": 105, "margin": 5, "date": "2026-01-16T00:00Z"},
            ]
        }
    }
    
    # Line Movement History - Shows sharp money on Celtics
    line_movement_history = [
        {
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
            "home_ml": 1.30,
            "away_ml": 3.50,
            "spread": -10.5,
            "total": 228.0
        },
        {
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
            "home_ml": 1.28,
            "away_ml": 3.75,
            "spread": -11.0,
            "total": 227.0
        },
        {
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            "home_ml": 1.26,
            "away_ml": 3.90,
            "spread": -11.5,
            "total": 226.0
        },
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "home_ml": 1.25,
            "away_ml": 4.00,
            "spread": -12.5,
            "total": 225.5
        }
    ]
    
    opening_odds = {
        "home_ml": 1.30,
        "away_ml": 3.50,
        "spread": -10.5,
        "total": 228.0,
        "timestamp": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    }
    
    current_odds = test_event["odds"]
    
    print("\n📋 TEST SCENARIO DETAILS:")
    print("-" * 80)
    print(f"Game: {test_event['home_team']} vs {test_event['away_team']}")
    print(f"Time: {test_event['commence_time']}")
    print(f"\n🏀 TEAM COMPARISON:")
    print(f"  Boston Celtics: 40-10 (80% win rate), +8.5 avg margin, 5-game win streak")
    print(f"  Charlotte Hornets: 10-40 (20% win rate), -7.2 avg margin, 6-game losing streak")
    print(f"\n🤕 INJURIES:")
    print(f"  Celtics: HEALTHY (no injuries)")
    print(f"  Hornets: OUT - LaMelo Ball (PG), Brandon Miller (F) - 2 KEY PLAYERS")
    print(f"\n📊 ODDS:")
    print(f"  Celtics: -400 (1.25 decimal)")
    print(f"  Spread: Celtics -12.5")
    print(f"  Total: 225.5")
    print(f"\n📈 LINE MOVEMENT:")
    print(f"  Opening: Celtics -10.5")
    print(f"  Current: Celtics -12.5")
    print(f"  Movement: +2.0 points toward Celtics (SHARP MONEY)")
    print(f"\n⏰ REST SITUATION:")
    print(f"  Celtics: 3 days rest (WELL RESTED)")
    print(f"  Hornets: 0 days rest (BACK-TO-BACK)")
    
    print(f"\n{'='*80}")
    print("RUNNING V6 ANALYSIS...")
    print("="*80)
    
    # Run V6 Analysis
    try:
        prediction = await generate_v6_prediction(
            event=test_event,
            sport_key="basketball_nba",
            squad_data=squad_data,
            matchup_data=matchup_data,
            line_movement_history=line_movement_history,
            opening_odds=opening_odds,
            current_odds=current_odds
        )
        
        print("\n✅ V6 ANALYSIS COMPLETE!")
        print("="*80)
        
        if prediction.get("has_pick"):
            print("\n🎯 PREDICTION RESULT: PICK GENERATED!")
            print("-" * 80)
            print(f"Pick: {prediction.get('pick')} ({prediction.get('pick_type')})")
            print(f"Confidence: {prediction.get('confidence')}%")
            print(f"Edge: {prediction.get('edge')}%")
            print(f"Model Agreement: {prediction.get('model_agreement')}%")
            print(f"Odds: {prediction.get('odds')}")
            
            print(f"\n📊 MODEL BREAKDOWN:")
            ensemble = prediction.get("ensemble_details", {})
            for model_name, model_data in ensemble.get("individual_predictions", {}).items():
                pick = model_data.get("pick", "None")
                prob = model_data.get("probability", 0) * 100
                conf = model_data.get("confidence", 0)
                print(f"  {model_name}: {prob:.1f}% probability, {conf:.1f}% confidence → {pick}")
            
            print(f"\n💰 BETTING RECOMMENDATION:")
            print(f"  Bet: {prediction.get('pick')}")
            print(f"  Amount: Calculate using Kelly Criterion")
            print(f"  Expected Value: +{prediction.get('edge')}%")
            
            print(f"\n📝 REASONING:")
            reasoning = prediction.get('reasoning', '').split('\n')
            for line in reasoning:
                if line.strip():
                    print(f"  {line}")
            
            print(f"\n✅ TEST PASSED: V6 SUCCESSFULLY GENERATED A PICK!")
            
        else:
            print("\n❌ NO PICK GENERATED")
            print(f"Reason: {prediction.get('reasoning', 'Unknown')}")
            print(f"\nEnsemble Probability: {prediction.get('ensemble_probability')}%")
            print(f"Ensemble Confidence: {prediction.get('ensemble_confidence')}%")
            print(f"Model Agreement: {prediction.get('model_agreement')}%")
            
            print(f"\n📊 MODEL BREAKDOWN:")
            ensemble = prediction.get("ensemble_details", {})
            for model_name, model_data in ensemble.get("individual_predictions", {}).items():
                pick = model_data.get("pick", "None")
                prob = model_data.get("probability", 0) * 100
                conf = model_data.get("confidence", 0)
                print(f"  {model_name}: {prob:.1f}% probability, {conf:.1f}% confidence → {pick}")
        
        return prediction
        
    except Exception as e:
        print(f"\n❌ ERROR during V6 analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    prediction = asyncio.run(create_test_scenario())
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
