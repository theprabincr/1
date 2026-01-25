import requests
import sys
import json
from datetime import datetime

class BettingPredictorAPITester:
    def __init__(self, base_url="https://sportspredictai-4.preview.emergentagent.com/api"):
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
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    if isinstance(response_data, list):
                        print(f"   Response: List with {len(response_data)} items")
                    elif isinstance(response_data, dict):
                        print(f"   Response keys: {list(response_data.keys())}")
                except:
                    print(f"   Response: {response.text[:100]}...")
            else:
                self.failed_tests.append({
                    'name': name,
                    'expected': expected_status,
                    'actual': response.status_code,
                    'response': response.text[:200]
                })
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}")

            return success, response.json() if success and response.text else {}

        except Exception as e:
            self.failed_tests.append({
                'name': name,
                'error': str(e)
            })
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test root API endpoint"""
        return self.run_test("Root API", "GET", "", 200)

    def test_sports_endpoint(self):
        """Test sports list endpoint"""
        return self.run_test("Sports List", "GET", "sports", 200)

    def test_events_endpoints(self):
        """Test events endpoints for different sports"""
        sports = ["basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl", "soccer_epl"]
        
        for sport in sports:
            success, data = self.run_test(f"Events - {sport}", "GET", f"events/{sport}", 200)
            if success and data:
                print(f"   Found {len(data)} events for {sport}")
                
                # Test event details if events exist
                if len(data) > 0:
                    event_id = data[0].get('id')
                    if event_id:
                        self.run_test(f"Event Details - {sport}", "GET", f"event/{event_id}?sport_key={sport}", 200)

    def test_line_movement(self):
        """Test line movement endpoint"""
        # Get events first
        success, events = self.run_test("Events for Line Movement", "GET", "events/basketball_nba", 200)
        if success and events and len(events) > 0:
            event_id = events[0].get('id')
            if event_id:
                self.run_test("Line Movement", "GET", f"line-movement/{event_id}", 200)
        else:
            print("⚠️  Skipping line movement test - no events available")

    def test_odds_comparison(self):
        """Test odds comparison endpoint"""
        # Get events first
        success, events = self.run_test("Events for Odds Comparison", "GET", "events/basketball_nba", 200)
        if success and events and len(events) > 0:
            event_id = events[0].get('id')
            if event_id:
                self.run_test("Odds Comparison", "GET", f"odds-comparison/{event_id}?sport_key=basketball_nba", 200)
        else:
            print("⚠️  Skipping odds comparison test - no events available")

    def test_ai_analysis(self):
        """Test AI analysis endpoint"""
        # Get events first
        success, events = self.run_test("Events for AI Analysis", "GET", "events/basketball_nba", 200)
        if success and events and len(events) > 0:
            event = events[0]
            analysis_data = {
                "event_id": event.get('id'),
                "home_team": event.get('home_team'),
                "away_team": event.get('away_team'),
                "sport_key": "basketball_nba",
                "odds_data": {
                    "bookmakers": event.get('bookmakers', [])
                }
            }
            self.run_test("AI Analysis", "POST", "analyze", 200, analysis_data, timeout=60)
        else:
            print("⚠️  Skipping AI analysis test - no events available")

    def test_recommendations(self):
        """Test recommendations endpoints"""
        self.run_test("Get Recommendations", "GET", "recommendations?limit=5", 200)
        self.run_test("Generate Recommendations", "POST", "generate-recommendations?sport_key=basketball_nba", 200, timeout=60)

    def test_performance(self):
        """Test performance endpoint"""
        self.run_test("Performance Stats", "GET", "performance", 200)
        self.run_test("Performance by Sport", "GET", "performance?sport_key=basketball_nba", 200)

    def test_prediction_workflow(self):
        """Test prediction creation and update workflow"""
        # Create a test prediction
        prediction_data = {
            "event_id": "test_event_123",
            "sport_key": "basketball_nba",
            "home_team": "Test Home Team",
            "away_team": "Test Away Team",
            "commence_time": datetime.now().isoformat(),
            "prediction_type": "moneyline",
            "predicted_outcome": "Test Home Team",
            "confidence": 0.75,
            "analysis": "Test AI analysis for prediction",
            "ai_model": "gpt-5.2",
            "odds_at_prediction": -110
        }
        
        success, response = self.run_test("Create Prediction", "POST", "recommendations", 201, prediction_data)
        
        if success and response.get('id'):
            prediction_id = response['id']
            
            # Update prediction result
            update_data = {
                "prediction_id": prediction_id,
                "result": "win"
            }
            self.run_test("Update Prediction Result", "PUT", "result", 200, update_data)

def main():
    print("🚀 Starting BetPredictor API Testing...")
    print("=" * 60)
    
    tester = BettingPredictorAPITester()
    
    # Run all tests
    print("\n📡 Testing Core API Endpoints...")
    tester.test_root_endpoint()
    tester.test_sports_endpoint()
    
    print("\n🏀 Testing Events & Odds...")
    tester.test_events_endpoints()
    tester.test_line_movement()
    tester.test_odds_comparison()
    
    print("\n🤖 Testing AI Features...")
    tester.test_ai_analysis()
    tester.test_recommendations()
    
    print("\n📊 Testing Performance & Predictions...")
    tester.test_performance()
    tester.test_prediction_workflow()
    
    # Print final results
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} passed")
    
    if tester.failed_tests:
        print(f"\n❌ Failed Tests ({len(tester.failed_tests)}):")
        for i, failure in enumerate(tester.failed_tests, 1):
            print(f"{i}. {failure['name']}")
            if 'error' in failure:
                print(f"   Error: {failure['error']}")
            else:
                print(f"   Expected: {failure['expected']}, Got: {failure['actual']}")
                print(f"   Response: {failure['response']}")
    
    success_rate = (tester.tests_passed / tester.tests_run * 100) if tester.tests_run > 0 else 0
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    return 0 if success_rate >= 80 else 1

if __name__ == "__main__":
    sys.exit(main())