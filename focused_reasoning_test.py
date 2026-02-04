#!/usr/bin/env python3
"""
Focused test for consolidated reasoning text in Events modal
Tests specific requirements from the review request
"""

import requests
import json
from pathlib import Path

# Get backend URL from frontend .env
def get_backend_url():
    frontend_env_path = Path("/app/frontend/.env")
    if frontend_env_path.exists():
        with open(frontend_env_path, 'r') as f:
            for line in f:
                if line.startswith('REACT_APP_BACKEND_URL='):
                    base_url = line.split('=', 1)[1].strip()
                    return f"{base_url}/api"
    return "http://localhost:8001/api"

BASE_URL = get_backend_url()
print(f"Testing backend at: {BASE_URL}")

def test_endpoint_detailed(endpoint, description):
    """Test endpoint and show detailed response"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n{'='*80}")
    print(f"🧪 TESTING: {endpoint}")
    print(f"📝 Description: {description}")
    print(f"🔗 URL: {url}")
    print(f"{'='*80}")
    
    try:
        response = requests.post(url, timeout=30)
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            prediction = data.get('prediction', {})
            
            # Key fields
            print(f"\n🎯 KEY FIELDS:")
            print(f"   has_pick: {prediction.get('has_pick', 'MISSING')}")
            print(f"   pick_display: '{prediction.get('pick_display', 'MISSING')}'")
            print(f"   algorithm: {prediction.get('algorithm', 'MISSING')}")
            
            # Favored outcomes
            print(f"\n🏆 FAVORED OUTCOMES:")
            print(f"   ml_favored_team: '{prediction.get('ml_favored_team', 'MISSING')}'")
            print(f"   spread_favored_team: '{prediction.get('spread_favored_team', 'MISSING')}'")
            print(f"   totals_favored: '{prediction.get('totals_favored', 'MISSING')}'")
            
            # Reasoning analysis
            reasoning = prediction.get('reasoning', '') or prediction.get('analysis', '')
            if reasoning:
                print(f"\n📝 REASONING ANALYSIS:")
                print(f"   Total length: {len(reasoning)} characters")
                
                # Split by sections
                sections = reasoning.split('\n\n')
                sections = [s.strip() for s in sections if s.strip()]
                print(f"   Sections (split by \\n\\n): {len(sections)}")
                
                # Count OVER mentions
                over_count = reasoning.upper().count('OVER')
                print(f"   'OVER' mentions: {over_count}")
                
                # Check for standalone "Pick:" lines
                standalone_picks = []
                for section in sections:
                    lines = section.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line == "Pick:" or (line.startswith("Pick:") and len(line.split()) <= 2):
                            standalone_picks.append(line)
                
                if standalone_picks:
                    print(f"   ⚠️  Standalone 'Pick:' lines found: {standalone_picks}")
                else:
                    print(f"   ✅ No standalone 'Pick:' lines found")
                
                print(f"\n📄 REASONING TEXT:")
                print(f"   {'-'*60}")
                for i, section in enumerate(sections, 1):
                    print(f"   Section {i}: {section[:100]}{'...' if len(section) > 100 else ''}")
                print(f"   {'-'*60}")
                
                # Show full reasoning if short enough
                if len(reasoning) <= 1000:
                    print(f"\n📋 FULL REASONING TEXT:")
                    print(f"   {reasoning}")
            else:
                print(f"\n❌ No reasoning text found")
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Run focused tests for the review request"""
    print("🎯 FOCUSED CONSOLIDATED REASONING TEXT TESTS")
    print("Testing specific requirements from the review request")
    
    # Test 1: Toronto vs Minnesota (should have pick)
    test_endpoint_detailed(
        "/analyze-unified/401810582?sport_key=basketball_nba",
        "Toronto vs Minnesota - Should have pick_display='Toronto Raptors ML', ≤7 sections, 1 OVER mention, no standalone Pick: lines"
    )
    
    # Test 2: Knicks vs Nuggets (should NOT have pick)
    test_endpoint_detailed(
        "/analyze-unified/401810581?sport_key=basketball_nba", 
        "Knicks vs Nuggets - Should have has_pick=false (edge too low), reasoning explains why no pick"
    )
    
    print(f"\n{'='*80}")
    print("🏁 FOCUSED TESTING COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()