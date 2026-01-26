import requests
import sys
import json
from datetime import datetime, timezone

class FocusedBettingTester:
    def __init__(self, base_url="https://smart-forecast-13.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    return success, response_data
                except:
                    return success, {}
            else:
                self.failed_tests.append({
                    'name': name,
                    'expected': expected_status,
                    'actual': response.status_code,
                    'response': response.text[:200]
                })
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False, {}

        except Exception as e:
            self.failed_tests.append({
                'name': name,
                'error': str(e)
            })
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_upcoming_events_only(self):
        """Test that events endpoint returns only UPCOMING events (not past events)"""
        print("\n🔍 Testing UPCOMING EVENTS ONLY requirement...")
        
        sports = ["basketball_nba", "americanfootball_nfl", "baseball_mlb"]
        
        for sport in sports:
            success, events = self.run_test(f"Upcoming Events - {sport}", "GET", f"events/{sport}", 200)
            if success and events:
                print(f"   Found {len(events)} events for {sport}")
                
                # Check if all events are in the future
                now = datetime.now(timezone.utc)
                future_events = 0
                past_events = 0
                
                for event in events:
                    commence_time_str = event.get('commence_time', '')
                    if commence_time_str:
                        try:
                            commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
                            if commence_time > now:
                                future_events += 1
                            else:
                                past_events += 1
                                print(f"   ⚠️  Found past event: {event.get('home_team')} vs {event.get('away_team')} at {commence_time_str}")
                        except:
                            print(f"   ⚠️  Invalid date format: {commence_time_str}")
                
                print(f"   ✅ Future events: {future_events}, Past events: {past_events}")
                if past_events > 0:
                    print(f"   ❌ REQUIREMENT FAILED: Found {past_events} past events!")
                else:
                    print(f"   ✅ REQUIREMENT PASSED: All events are upcoming")

    def test_decimal_odds_format(self):
        """Test that odds are displayed in European/decimal format (e.g., 1.14, 6.20)"""
        print("\n🔍 Testing DECIMAL ODDS FORMAT requirement...")
        
        success, events = self.run_test("Events for Decimal Odds Check", "GET", "events/basketball_nba", 200)
        if success and events and len(events) > 0:
            event = events[0]
            bookmakers = event.get('bookmakers', [])
            
            decimal_odds_found = False
            american_odds_found = False
            
            for bm in bookmakers:
                for market in bm.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            price = outcome.get('price', 0)
                            
                            # Decimal odds are typically between 1.01 and 50.00
                            # American odds are typically negative (-200) or positive (+150)
                            if 1.0 <= price <= 50.0:
                                decimal_odds_found = True
                                print(f"   ✅ Found decimal odds: {outcome.get('name')} @ {price}")
                            elif price < 0 or price > 100:
                                american_odds_found = True
                                print(f"   ❌ Found American odds: {outcome.get('name')} @ {price}")
            
            if decimal_odds_found and not american_odds_found:
                print(f"   ✅ REQUIREMENT PASSED: All odds are in decimal format")
            elif american_odds_found:
                print(f"   ❌ REQUIREMENT FAILED: Found American odds format")
            else:
                print(f"   ⚠️  No odds data found to verify format")

    def test_check_results_endpoint(self):
        """Test that /api/check-results endpoint exists for auto-tracking"""
        print("\n🔍 Testing CHECK-RESULTS ENDPOINT requirement...")
        success, data = self.run_test("Check Results Endpoint", "POST", "check-results", 200)
        if success:
            print(f"   ✅ REQUIREMENT PASSED: /api/check-results endpoint exists")
            print(f"   Response: {data}")
        else:
            print(f"   ❌ REQUIREMENT FAILED: /api/check-results endpoint not working")

    def test_scores_endpoint(self):
        """Test that /api/scores/{sport} endpoint exists for fetching results"""
        print("\n🔍 Testing SCORES ENDPOINT requirement...")
        
        sports = ["basketball_nba", "americanfootball_nfl"]
        
        for sport in sports:
            success, data = self.run_test(f"Scores Endpoint - {sport}", "GET", f"scores/{sport}", 200)
            if success:
                print(f"   ✅ REQUIREMENT PASSED: /api/scores/{sport} endpoint exists")
                if isinstance(data, list):
                    print(f"   Found {len(data)} completed events")
                elif isinstance(data, dict) and 'scores' in data:
                    scores = data.get('scores', [])
                    print(f"   Found {len(scores)} completed events")
            else:
                print(f"   ❌ REQUIREMENT FAILED: /api/scores/{sport} endpoint not working")

    def test_odds_comparison_decimal_format(self):
        """Test that Odds Comparison shows decimal odds from multiple sportsbooks"""
        print("\n🔍 Testing ODDS COMPARISON DECIMAL FORMAT requirement...")
        
        # Get events first
        success, events = self.run_test("Events for Odds Comparison", "GET", "events/basketball_nba", 200)
        if success and events and len(events) > 0:
            event_id = events[0].get('id')
            if event_id:
                success, comparison = self.run_test("Odds Comparison Decimal Check", "GET", f"odds-comparison/{event_id}?sport_key=basketball_nba", 200)
                if success and comparison:
                    # Check h2h market for decimal odds
                    h2h_data = comparison.get('h2h', [])
                    decimal_odds_count = 0
                    
                    for bm in h2h_data:
                        outcomes = bm.get('outcomes', [])
                        for outcome in outcomes:
                            price = outcome.get('price', 0)
                            if 1.0 <= price <= 50.0:
                                decimal_odds_count += 1
                                print(f"   ✅ {bm.get('title', 'Unknown')}: {outcome.get('name')} @ {price}")
                    
                    if decimal_odds_count > 0:
                        print(f"   ✅ REQUIREMENT PASSED: Odds comparison shows decimal odds from {len(h2h_data)} sportsbooks")
                    else:
                        print(f"   ❌ REQUIREMENT FAILED: No decimal odds found in comparison")
                else:
                    print(f"   ❌ REQUIREMENT FAILED: Odds comparison endpoint not working")

def main():
    print("🎯 Starting FOCUSED BetPredictor Requirements Testing...")
    print("=" * 70)
    
    tester = FocusedBettingTester()
    
    # Test specific requirements from review request
    tester.test_upcoming_events_only()
    tester.test_decimal_odds_format()
    tester.test_odds_comparison_decimal_format()
    tester.test_check_results_endpoint()
    tester.test_scores_endpoint()
    
    # Print final results
    print("\n" + "=" * 70)
    print(f"📊 Focused Test Results: {tester.tests_passed}/{tester.tests_run} passed")
    
    if tester.failed_tests:
        print(f"\n❌ Failed Tests ({len(tester.failed_tests)}):")
        for i, failure in enumerate(tester.failed_tests, 1):
            print(f"{i}. {failure['name']}")
            if 'error' in failure:
                print(f"   Error: {failure['error']}")
            else:
                print(f"   Expected: {failure['expected']}, Got: {failure['actual']}")
    
    success_rate = (tester.tests_passed / tester.tests_run * 100) if tester.tests_run > 0 else 0
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    return 0 if success_rate >= 80 else 1

if __name__ == "__main__":
    sys.exit(main())