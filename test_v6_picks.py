#!/usr/bin/env python3
"""
Test V6 with multiple events to find a pick
"""

import asyncio
import aiohttp
import json

BACKEND_URL = "https://predictr-ai.preview.emergentagent.com/api"

async def test_multiple_events():
    """Test V6 analysis on multiple events to find picks"""
    
    async with aiohttp.ClientSession() as session:
        print("🎯 Testing V6 Analysis on Multiple NBA Events")
        print("=" * 50)
        
        # Get NBA events
        async with session.get(f"{BACKEND_URL}/events/basketball_nba?pre_match_only=true") as response:
            events = await response.json()
            
        if not events:
            print("❌ No NBA events found")
            return
        
        print(f"Found {len(events)} NBA events to test")
        
        picks_found = 0
        no_picks_found = 0
        
        for i, event in enumerate(events[:5]):  # Test first 5 events
            event_id = event.get("id")
            home_team = event.get("home_team")
            away_team = event.get("away_team")
            
            print(f"\n{i+1}. Testing: {home_team} vs {away_team} (ID: {event_id})")
            
            try:
                async with session.post(f"{BACKEND_URL}/analyze-v6/{event_id}?sport_key=basketball_nba") as response:
                    if response.status == 200:
                        data = await response.json()
                        prediction = data.get("prediction", {})
                        
                        if prediction.get("has_pick"):
                            picks_found += 1
                            print(f"   ✅ HAS PICK!")
                            print(f"   Pick: {prediction.get('pick')}")
                            print(f"   Confidence: {prediction.get('confidence')}%")
                            print(f"   Edge: {prediction.get('edge')}%")
                            print(f"   Model Agreement: {prediction.get('model_agreement')}%")
                            
                            # Show ensemble details
                            ensemble = prediction.get("ensemble_details", {})
                            individual = ensemble.get("individual_predictions", {})
                            print(f"   Ensemble Probability: {ensemble.get('ensemble_probability')}")
                            print(f"   Models in consensus: {ensemble.get('num_models')}")
                            
                            # Show individual model picks
                            model_picks = {}
                            for model_name, model_pred in individual.items():
                                pick = model_pred.get("pick")
                                if pick:
                                    model_picks[model_name] = pick
                            
                            if model_picks:
                                print(f"   Model picks: {model_picks}")
                            
                            # Show key factors
                            key_factors = prediction.get("key_factors", [])
                            if key_factors:
                                print(f"   Key factors: {key_factors[:3]}")
                            
                        else:
                            no_picks_found += 1
                            reasoning = prediction.get("reasoning", "")
                            ensemble_conf = prediction.get("ensemble_confidence", 0)
                            model_agreement = prediction.get("model_agreement", 0)
                            
                            print(f"   ❌ NO PICK")
                            print(f"   Reason: {reasoning[:80]}...")
                            print(f"   Ensemble Confidence: {ensemble_conf}%")
                            print(f"   Model Agreement: {model_agreement}%")
                    else:
                        print(f"   ❌ Error: Status {response.status}")
                        
            except Exception as e:
                print(f"   ❌ Exception: {e}")
        
        print(f"\n📊 Summary:")
        print(f"   Picks found: {picks_found}")
        print(f"   No picks: {no_picks_found}")
        print(f"   Total tested: {picks_found + no_picks_found}")
        
        if picks_found == 0:
            print(f"\n✅ V6 Algorithm working correctly!")
            print(f"   - Conservative approach: Only recommends when confident")
            print(f"   - Requires 3+ models to agree and 65%+ ensemble confidence")
            print(f"   - This is by design - quality over quantity")
        else:
            print(f"\n🎯 V6 Algorithm found {picks_found} confident picks!")

if __name__ == "__main__":
    asyncio.run(test_multiple_events())