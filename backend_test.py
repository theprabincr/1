import requests
import sys
import json
from datetime import datetime

class BettingPredictorAPITester:
    def __init__(self, base_url="https://live-bet-updater.preview.emergentagent.com/api"):
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
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    # Handle CSV responses differently
                    if 'text/csv' in response.headers.get('content-type', ''):
                        print(f"   Response: CSV data ({len(response.text)} characters)")
                        return success, response.text
                    else:
                        response_data = response.json()
                        if isinstance(response_data, list):
                            print(f"   Response: List with {len(response_data)} items")
                        elif isinstance(response_data, dict):
                            print(f"   Response keys: {list(response_data.keys())}")
                        return success, response_data
                except:
                    print(f"   Response: {response.text[:100]}...")
                    return success, response.text
            else:
                self.failed_tests.append({
                    'name': name,
                    'expected': expected_status,
                    'actual': response.status_code,
                    'response': response.text[:200]
                })
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}")

            return success, {}

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

    def test_api_usage(self):
        """Test API usage endpoint - key requirement"""
        success, data = self.run_test("API Usage", "GET", "api-usage", 200)
        if success and data:
            # Check if requests_remaining is present
            if 'requests_remaining' in data:
                print(f"   ✅ API Usage tracking working - Requests remaining: {data['requests_remaining']}")
            else:
                print(f"   ⚠️  API Usage response missing 'requests_remaining' field")
                print(f"   Response: {data}")

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
        
        success, response = self.run_test("Create Prediction", "POST", "recommendations", 200, prediction_data)
        
        if success and response.get('id'):
            prediction_id = response['id']
            
            # Update prediction result
            update_data = {
                "prediction_id": prediction_id,
                "result": "win"
            }
            self.run_test("Update Prediction Result", "PUT", "result", 200, update_data)

    def test_api_key_management(self):
        """Test API Key Management CRUD operations"""
        print("\n🔑 Testing API Key Management...")
        
        # 1. List existing API keys
        success, keys_data = self.run_test("List API Keys", "GET", "api-keys", 200)
        
        # 2. Add new API key
        new_key_data = {
            "key": "test123abc",
            "name": "Test Key"
        }
        success, add_response = self.run_test("Add API Key", "POST", "api-keys", 200, new_key_data)
        
        if success and add_response.get('id'):
            key_id = add_response['id']
            print(f"   ✅ Created API key with ID: {key_id}")
            
            # 3. Activate the key
            self.run_test("Activate API Key", "PUT", f"api-keys/{key_id}/activate", 200)
            
            # 4. Reset key usage
            self.run_test("Reset API Key", "PUT", f"api-keys/{key_id}/reset", 200)
            
            # 5. Delete the key
            self.run_test("Delete API Key", "DELETE", f"api-keys/{key_id}", 200)
        else:
            print("   ⚠️  Skipping key operations - failed to create test key")

    def test_bankroll_management(self):
        """Test Bankroll Management operations"""
        print("\n💰 Testing Bankroll Management...")
        
        # 1. Get current bankroll status
        success, bankroll_data = self.run_test("Get Bankroll", "GET", "bankroll", 200)
        if success:
            print(f"   Current balance: ${bankroll_data.get('current_balance', 0)}")
        
        # 2. Make a deposit
        deposit_data = {
            "amount": 1000,
            "description": "Initial deposit"
        }
        success, deposit_response = self.run_test("Deposit Funds", "POST", "bankroll/deposit", 200, deposit_data)
        if success:
            print(f"   New balance after deposit: ${deposit_response.get('new_balance', 0)}")
        
        # 3. Make a withdrawal
        withdrawal_data = {
            "amount": 100,
            "description": "Test withdrawal"
        }
        success, withdrawal_response = self.run_test("Withdraw Funds", "POST", "bankroll/withdraw", 200, withdrawal_data)
        if success:
            print(f"   New balance after withdrawal: ${withdrawal_response.get('new_balance', 0)}")
        
        # 4. Get transactions
        self.run_test("Get Transactions", "GET", "bankroll/transactions", 200)

    def test_notifications(self):
        """Test Notifications system"""
        print("\n🔔 Testing Notifications...")
        
        # 1. Get all notifications
        success, notifications_data = self.run_test("Get Notifications", "GET", "notifications", 200)
        if success:
            notifications = notifications_data.get('notifications', [])
            unread_count = notifications_data.get('unread_count', 0)
            print(f"   Found {len(notifications)} notifications, {unread_count} unread")
            
            # 2. Mark a notification as read (if any exist)
            if notifications and len(notifications) > 0:
                notif_id = notifications[0].get('id')
                if notif_id:
                    self.run_test("Mark Notification Read", "PUT", f"notifications/{notif_id}/read", 200)
        
        # 3. Mark all notifications as read
        self.run_test("Mark All Notifications Read", "PUT", "notifications/read-all", 200)

    def test_settings(self):
        """Test Settings management"""
        print("\n⚙️ Testing Settings...")
        
        # 1. Get current settings
        success, settings_data = self.run_test("Get Settings", "GET", "settings", 200)
        
        # 2. Update settings
        updated_settings = {
            "cache_duration_minutes": 45,
            "auto_rotate_keys": True,
            "priority_sports": ["basketball_nba", "americanfootball_nfl"],
            "notification_preferences": {
                "line_movement_alerts": True,
                "line_movement_threshold": 7.5,
                "low_api_alerts": True,
                "low_api_threshold": 25,
                "result_alerts": True,
                "daily_summary": False
            }
        }
        self.run_test("Update Settings", "PUT", "settings", 200, updated_settings)

    def test_analytics(self):
        """Test Analytics endpoints"""
        print("\n📈 Testing Analytics...")
        
        # 1. Get trends for last 30 days
        self.run_test("Analytics Trends (30 days)", "GET", "analytics/trends?days=30", 200)
        
        # 2. Get trends for last 7 days
        self.run_test("Analytics Trends (7 days)", "GET", "analytics/trends?days=7", 200)
        
        # 3. Get streaks
        success, streaks_data = self.run_test("Analytics Streaks", "GET", "analytics/streaks", 200)
        if success:
            current_streak = streaks_data.get('current_streak', 0)
            streak_type = streaks_data.get('streak_type', 'none')
            print(f"   Current streak: {current_streak} {streak_type}")

    def test_export_functionality(self):
        """Test Export endpoints"""
        print("\n📤 Testing Export Functionality...")
        
        # 1. Export predictions as JSON
        self.run_test("Export Predictions (JSON)", "GET", "export/predictions?format=json", 200)
        
        # 2. Export predictions as CSV
        self.run_test("Export Predictions (CSV)", "GET", "export/predictions?format=csv", 200)
        
        # 3. Export bankroll as JSON
        self.run_test("Export Bankroll (JSON)", "GET", "export/bankroll?format=json", 200)
        
        # 4. Export bankroll as CSV
        self.run_test("Export Bankroll (CSV)", "GET", "export/bankroll?format=csv", 200)
        
        # 5. Export performance report
        success, report_data = self.run_test("Export Performance Report", "GET", "export/performance-report", 200)
        if success:
            print(f"   Performance report generated at: {report_data.get('generated_at', 'N/A')}")

    def test_enhanced_api_usage(self):
        """Test enhanced API usage endpoint with new fields"""
        print("\n📊 Testing Enhanced API Usage...")
        
        success, data = self.run_test("Enhanced API Usage", "GET", "api-usage", 200)
        if success and data:
            # Check for new required fields
            required_fields = ['total_remaining_all_keys', 'active_keys_count']
            missing_fields = []
            
            for field in required_fields:
                if field in data:
                    print(f"   ✅ {field}: {data[field]}")
                else:
                    missing_fields.append(field)
                    print(f"   ❌ Missing field: {field}")
            
            # Check other important fields
            optional_fields = ['requests_remaining', 'requests_used', 'monthly_limit']
            for field in optional_fields:
                if field in data:
                    print(f"   ℹ️  {field}: {data[field]}")
            
            if missing_fields:
                print(f"   ⚠️  API Usage endpoint missing required fields: {missing_fields}")
            else:
                print(f"   ✅ All required fields present in API Usage response")

    def test_espn_scores_integration(self):
        """Test ESPN Scores Integration endpoints"""
        print("\n🏆 Testing ESPN Scores Integration...")
        
        # Test scores for basketball_nba
        success, data = self.run_test("ESPN Scores - NBA", "GET", "scores/basketball_nba", 200)
        if success and data:
            games_count = data.get('games_count', 0)
            games = data.get('games', [])
            print(f"   Found {games_count} NBA games")
            
            # Verify response structure
            if games and len(games) > 0:
                game = games[0]
                required_fields = ['espn_id', 'home_team', 'away_team', 'home_score', 'away_score', 'status']
                missing_fields = []
                for field in required_fields:
                    if field not in game:
                        missing_fields.append(field)
                
                if missing_fields:
                    print(f"   ⚠️  Game data missing fields: {missing_fields}")
                else:
                    print(f"   ✅ Game data structure correct")
                    if game.get('winner'):
                        print(f"   Winner: {game.get('winner')}")
        
        # Test with status filter - final games
        success, data = self.run_test("ESPN Scores - Final Games", "GET", "scores/basketball_nba?status=final", 200)
        if success and data:
            final_games = data.get('games', [])
            print(f"   Found {len(final_games)} final games")
            
        # Test with status filter - live games
        success, data = self.run_test("ESPN Scores - Live Games", "GET", "scores/basketball_nba?status=live", 200)
        if success and data:
            live_games = data.get('games', [])
            print(f"   Found {len(live_games)} live games")

    def test_live_scores(self):
        """Test Live Scores endpoint"""
        print("\n🔴 Testing Live Scores...")
        
        success, data = self.run_test("Live Scores - All Sports", "GET", "live-scores", 200)
        if success and data:
            live_games_count = data.get('live_games_count', 0)
            games = data.get('games', [])
            print(f"   Found {live_games_count} live games across all sports")
            
            # Verify each game has sport_key
            if games:
                for i, game in enumerate(games[:3]):  # Check first 3 games
                    if 'sport_key' in game:
                        print(f"   Game {i+1}: {game.get('home_team')} vs {game.get('away_team')} ({game.get('sport_key')})")
                    else:
                        print(f"   ⚠️  Game {i+1} missing sport_key")

    def test_pending_results(self):
        """Test Pending Results endpoint"""
        print("\n⏳ Testing Pending Results...")
        
        success, data = self.run_test("Pending Results", "GET", "pending-results", 200)
        if success and data:
            total_pending = data.get('total_pending', 0)
            awaiting_start = data.get('awaiting_start', [])
            in_progress = data.get('in_progress', [])
            awaiting_result = data.get('awaiting_result', [])
            
            print(f"   Total pending predictions: {total_pending}")
            print(f"   Awaiting start: {len(awaiting_start)}")
            print(f"   In progress: {len(in_progress)}")
            print(f"   Awaiting result: {len(awaiting_result)}")
            
            # Verify structure
            required_fields = ['total_pending', 'awaiting_start', 'in_progress', 'awaiting_result']
            missing_fields = []
            for field in required_fields:
                if field not in data:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"   ⚠️  Response missing fields: {missing_fields}")
            else:
                print(f"   ✅ Response structure correct")

    def test_recommendations_70_percent_filter(self):
        """Test Recommendations with 70% confidence filter"""
        print("\n🎯 Testing 70% Confidence Filter...")
        
        # Test with 70% minimum confidence (default)
        success, data = self.run_test("Recommendations - 70% Filter", "GET", "recommendations?min_confidence=0.70&limit=10", 200)
        if success and isinstance(data, list):
            print(f"   Found {len(data)} recommendations with 70%+ confidence")
            
            # Verify all recommendations meet confidence threshold
            low_confidence_count = 0
            for rec in data:
                confidence = rec.get('confidence', 0)
                if confidence < 0.70:
                    low_confidence_count += 1
            
            if low_confidence_count > 0:
                print(f"   ⚠️  Found {low_confidence_count} recommendations below 70% confidence")
            else:
                print(f"   ✅ All recommendations meet 70% confidence threshold")
        
        # Test with include_all=true (should include all confidence levels)
        success, data = self.run_test("Recommendations - All Confidence", "GET", "recommendations?include_all=true", 200)
        if success and isinstance(data, list):
            print(f"   Found {len(data)} recommendations (all confidence levels)")
            
            # Check confidence distribution
            if data:
                confidences = [rec.get('confidence', 0) for rec in data]
                min_conf = min(confidences) if confidences else 0
                max_conf = max(confidences) if confidences else 0
                print(f"   Confidence range: {min_conf:.2f} - {max_conf:.2f}")

    def test_check_results_trigger(self):
        """Test Check Results Trigger endpoint"""
        print("\n🔄 Testing Check Results Trigger...")
        
        success, data = self.run_test("Check Results Trigger", "POST", "check-results", 200)
        if success and data:
            message = data.get('message', '')
            if 'background' in message.lower() or 'started' in message.lower():
                print(f"   ✅ Background task triggered successfully")
                print(f"   Message: {message}")
            else:
                print(f"   ⚠️  Unexpected response: {message}")

    def test_performance_stats_updated(self):
        """Test Performance Stats with updated fields"""
        print("\n📈 Testing Updated Performance Stats...")
        
        success, data = self.run_test("Performance Stats", "GET", "performance", 200)
        if success and data:
            # Check for required fields
            required_fields = ['wins', 'losses', 'pushes', 'win_rate', 'roi']
            missing_fields = []
            
            for field in required_fields:
                if field in data:
                    value = data[field]
                    print(f"   ✅ {field}: {value}")
                else:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"   ⚠️  Performance stats missing fields: {missing_fields}")
            else:
                print(f"   ✅ All required performance fields present")
            
            # Check calculations
            total_predictions = data.get('total_predictions', 0)
            wins = data.get('wins', 0)
            losses = data.get('losses', 0)
            pushes = data.get('pushes', 0)
            
            if total_predictions > 0:
                calculated_total = wins + losses + pushes
                if calculated_total == total_predictions:
                    print(f"   ✅ Win/Loss/Push totals match total predictions")
                else:
                    print(f"   ⚠️  Total mismatch: {calculated_total} vs {total_predictions}")

    def test_existing_endpoints_still_work(self):
        """Test that existing endpoints still work after updates"""
        print("\n🔧 Testing Existing Endpoints...")
        
        # Test core endpoints that should still work
        endpoints_to_test = [
            ("Sports List", "GET", "sports"),
            ("Events NBA", "GET", "events/basketball_nba"),
            ("Notifications", "GET", "notifications"),
            ("Settings", "GET", "settings")
        ]
        
        for name, method, endpoint in endpoints_to_test:
            success, data = self.run_test(f"Existing - {name}", method, endpoint, 200)
            if not success:
                print(f"   ❌ CRITICAL: Existing endpoint {endpoint} is broken!")

def main():
    print("🚀 Starting BetPredictor API Testing...")
    print("=" * 60)
    
    tester = BettingPredictorAPITester()
    
    # Run all tests
    print("\n📡 Testing Core API Endpoints...")
    tester.test_root_endpoint()
    tester.test_sports_endpoint()
    tester.test_enhanced_api_usage()  # Test the enhanced API usage endpoint
    
    print("\n🔑 Testing New BetPredictor Enhancements...")
    tester.test_api_key_management()
    tester.test_bankroll_management()
    tester.test_notifications()
    tester.test_settings()
    tester.test_analytics()
    tester.test_export_functionality()
    
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