import requests
import sys
import json
from datetime import datetime

class BettingPredictorAPITester:
    def __init__(self, base_url="https://fix-finder-31.preview.emergentagent.com/api"):
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

    def test_all_markets_ml_spread_total(self):
        """Test All Markets (ML/Spread/Total) - NEW FEATURE"""
        print("\n🎯 Testing All Markets (ML/Spread/Total)...")
        
        success, data = self.run_test("All Markets - NBA", "GET", "events/basketball_nba?pre_match_only=true", 200)
        if success and isinstance(data, list) and len(data) > 0:
            event = data[0]
            bookmakers = event.get('bookmakers', [])
            
            if bookmakers:
                bookmaker = bookmakers[0]
                markets = bookmaker.get('markets', [])
                market_keys = [m.get('key') for m in markets]
                
                # Check for all three market types
                required_markets = ['h2h', 'spreads', 'totals']
                missing_markets = []
                
                for market_type in required_markets:
                    if market_type in market_keys:
                        print(f"   ✅ {market_type} market found")
                        
                        # Find the market and check structure
                        market = next((m for m in markets if m.get('key') == market_type), None)
                        if market:
                            outcomes = market.get('outcomes', [])
                            
                            if market_type == 'spreads':
                                # Check spreads have point field
                                has_points = any(outcome.get('point') is not None for outcome in outcomes)
                                if has_points:
                                    print(f"   ✅ Spreads have point values")
                                else:
                                    print(f"   ⚠️  Spreads missing point values")
                            
                            elif market_type == 'totals':
                                # Check totals have Over/Under with point field
                                over_under = [o.get('name') for o in outcomes]
                                has_over_under = 'Over' in over_under and 'Under' in over_under
                                has_points = any(outcome.get('point') is not None for outcome in outcomes)
                                
                                if has_over_under and has_points:
                                    print(f"   ✅ Totals have Over/Under with point values")
                                else:
                                    print(f"   ⚠️  Totals structure incomplete")
                    else:
                        missing_markets.append(market_type)
                
                if missing_markets:
                    print(f"   ❌ Missing markets: {missing_markets}")
                else:
                    print(f"   ✅ All three markets (ML/Spread/Total) available")
            else:
                print(f"   ⚠️  No bookmakers found in event data")
        else:
            print(f"   ⚠️  No events found for market testing")

    def test_pre_match_only_filter(self):
        """Test Pre-match Only Filter - NEW FEATURE"""
        print("\n⏰ Testing Pre-match Only Filter...")
        
        # Test with pre_match_only=true (default)
        success_pre, data_pre = self.run_test("Pre-match Only (True)", "GET", "events/basketball_nba?pre_match_only=true", 200)
        
        # Test with pre_match_only=false
        success_all, data_all = self.run_test("Pre-match Only (False)", "GET", "events/basketball_nba?pre_match_only=false", 200)
        
        if success_pre and success_all:
            pre_count = len(data_pre) if isinstance(data_pre, list) else 0
            all_count = len(data_all) if isinstance(data_all, list) else 0
            
            print(f"   Pre-match only events: {pre_count}")
            print(f"   All events: {all_count}")
            
            # Verify pre-match events have future commence_time
            if isinstance(data_pre, list) and len(data_pre) > 0:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                future_events = 0
                
                for event in data_pre[:5]:  # Check first 5 events
                    commence_str = event.get('commence_time', '')
                    if commence_str:
                        try:
                            commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                            if commence_time > now:
                                future_events += 1
                        except:
                            pass
                
                if future_events > 0:
                    print(f"   ✅ Pre-match filter working - {future_events} future events verified")
                else:
                    print(f"   ⚠️  Pre-match filter may not be working correctly")
            
            # All events should be >= pre-match only events
            if all_count >= pre_count:
                print(f"   ✅ Filter logic correct (all >= pre-match)")
            else:
                print(f"   ⚠️  Filter logic issue (all < pre-match)")

    def test_continuous_score_sync(self):
        """Test Continuous Score Sync - NEW FEATURE"""
        print("\n🔄 Testing Continuous Score Sync...")
        
        success, data = self.run_test("Live Scores Sync", "GET", "live-scores", 200)
        if success and data:
            # Check required fields
            required_fields = ['live_games_count', 'games']
            missing_fields = []
            
            for field in required_fields:
                if field in data:
                    print(f"   ✅ {field}: {data[field] if field != 'games' else f'{len(data[field])} games'}")
                else:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"   ❌ Missing fields: {missing_fields}")
            else:
                print(f"   ✅ Live scores structure correct")
            
            # Check games have real-time scores
            games = data.get('games', [])
            if games:
                game = games[0]
                score_fields = ['home_score', 'away_score', 'status']
                has_scores = all(field in game for field in score_fields)
                
                if has_scores:
                    print(f"   ✅ Games have real-time score data")
                    print(f"   Sample: {game.get('home_team')} {game.get('home_score')} - {game.get('away_score')} {game.get('away_team')} ({game.get('status')})")
                else:
                    print(f"   ⚠️  Games missing score fields")

    def test_line_movement_cleanup(self):
        """Test Line Movement Cleanup - NEW FEATURE"""
        print("\n🧹 Testing Line Movement Cleanup...")
        
        success, data = self.run_test("Line Movement Cleanup", "POST", "cleanup-line-movement", 200)
        if success and data:
            deleted_count = data.get('deleted_count', 0)
            message = data.get('message', '')
            
            print(f"   ✅ Cleanup executed successfully")
            print(f"   Deleted records: {deleted_count}")
            print(f"   Message: {message}")
            
            # Verify response structure
            if 'deleted_count' in data:
                print(f"   ✅ Response includes deleted_count")
            else:
                print(f"   ⚠️  Response missing deleted_count field")

    def test_performance_stats_enhanced(self):
        """Test Enhanced Performance Stats - NEW FEATURE"""
        print("\n📊 Testing Enhanced Performance Stats...")
        
        success, data = self.run_test("Enhanced Performance Stats", "GET", "performance", 200)
        if success and data:
            # Check for all required fields
            required_fields = ['wins', 'losses', 'pushes', 'win_rate', 'roi']
            missing_fields = []
            
            for field in required_fields:
                if field in data:
                    value = data[field]
                    print(f"   ✅ {field}: {value}")
                else:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"   ❌ Missing required fields: {missing_fields}")
            else:
                print(f"   ✅ All performance fields present")
            
            # Check recent_predictions includes final_score when available
            recent_predictions = data.get('recent_predictions', [])
            if recent_predictions:
                predictions_with_scores = 0
                for pred in recent_predictions[:5]:  # Check first 5
                    if 'final_score' in pred:
                        predictions_with_scores += 1
                        final_score = pred['final_score']
                        print(f"   ✅ Prediction has final_score: {final_score}")
                        break
                
                if predictions_with_scores > 0:
                    print(f"   ✅ Recent predictions include final_score data")
                else:
                    print(f"   ℹ️  No recent predictions with final_score (may be normal)")

    def test_espn_scores_api_enhanced(self):
        """Test Enhanced ESPN Scores API - NEW FEATURE"""
        print("\n🏀 Testing Enhanced ESPN Scores API...")
        
        # Test basic ESPN scores
        success, data = self.run_test("ESPN Scores - NBA", "GET", "scores/basketball_nba", 200)
        if success and data:
            games = data.get('games', [])
            print(f"   Found {len(games)} NBA games")
            
            if games:
                game = games[0]
                # Check required structure
                required_fields = ['espn_id', 'home_team', 'away_team', 'home_score', 'away_score', 'status']
                missing_fields = []
                
                for field in required_fields:
                    if field not in game:
                        missing_fields.append(field)
                
                if missing_fields:
                    print(f"   ❌ Game missing fields: {missing_fields}")
                else:
                    print(f"   ✅ Game structure correct")
                    
                    # Check if winner field exists for final games
                    if game.get('status') == 'final' and 'winner' in game:
                        print(f"   ✅ Final game has winner: {game['winner']}")
        
        # Test status filter - final games
        success, data = self.run_test("ESPN Scores - Final Status", "GET", "scores/basketball_nba?status=final", 200)
        if success and data:
            final_games = data.get('games', [])
            print(f"   Found {len(final_games)} final games")
            
            # Verify all games have final status
            if final_games:
                non_final = [g for g in final_games if g.get('status') != 'final']
                if not non_final:
                    print(f"   ✅ Status filter working - all games are final")
                else:
                    print(f"   ⚠️  Status filter issue - {len(non_final)} non-final games")

    def test_pending_results_enhanced(self):
        """Test Enhanced Pending Results - NEW FEATURE"""
        print("\n⏳ Testing Enhanced Pending Results...")
        
        success, data = self.run_test("Enhanced Pending Results", "GET", "pending-results", 200)
        if success and data:
            # Check all required fields
            required_fields = ['total_pending', 'awaiting_start', 'in_progress', 'awaiting_result']
            missing_fields = []
            
            for field in required_fields:
                if field in data:
                    if isinstance(data[field], list):
                        print(f"   ✅ {field}: {len(data[field])} items")
                    else:
                        print(f"   ✅ {field}: {data[field]}")
                else:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"   ❌ Missing fields: {missing_fields}")
            else:
                print(f"   ✅ All pending results fields present")
            
            # Verify categorization logic
            total_pending = data.get('total_pending', 0)
            awaiting_start = len(data.get('awaiting_start', []))
            in_progress = len(data.get('in_progress', []))
            awaiting_result = len(data.get('awaiting_result', []))
            
            calculated_total = awaiting_start + in_progress + awaiting_result
            if calculated_total == total_pending:
                print(f"   ✅ Categorization totals match: {calculated_total}")
            else:
                print(f"   ⚠️  Total mismatch: {calculated_total} vs {total_pending}")

    def test_recommendations_70_percent_enhanced(self):
        """Test Enhanced 70% Confidence Filter - NEW FEATURE"""
        print("\n🎯 Testing Enhanced 70% Confidence Filter...")
        
        # Test default 70% filter
        success, data = self.run_test("70% Confidence Filter", "GET", "recommendations?min_confidence=0.70&limit=5", 200)
        if success and isinstance(data, list):
            print(f"   Found {len(data)} recommendations with 70%+ confidence")
            
            # Verify confidence levels
            low_confidence = 0
            for rec in data:
                confidence = rec.get('confidence', 0)
                if confidence < 0.70:
                    low_confidence += 1
            
            if low_confidence == 0:
                print(f"   ✅ All recommendations meet 70% threshold")
            else:
                print(f"   ❌ {low_confidence} recommendations below 70%")
        
        # Test include_all parameter
        success, data = self.run_test("Include All Confidence", "GET", "recommendations?include_all=true", 200)
        if success and isinstance(data, list):
            print(f"   Found {len(data)} recommendations (all confidence levels)")
            
            if data:
                confidences = [rec.get('confidence', 0) for rec in data]
                min_conf = min(confidences)
                max_conf = max(confidences)
                print(f"   Confidence range: {min_conf:.2f} - {max_conf:.2f}")
                
                # Should include lower confidence when include_all=true
                has_low_confidence = any(c < 0.70 for c in confidences)
                if has_low_confidence:
                    print(f"   ✅ include_all=true returns lower confidence predictions")
                else:
                    print(f"   ℹ️  No low confidence predictions available")

    def test_v3_predictions_endpoint(self):
        """Test NEW V3 Predictions endpoint"""
        print("\n🚀 Testing V3 Predictions Endpoint...")
        
        success, data = self.run_test("V3 Predictions List", "GET", "predictions/v3?limit=20", 200)
        if success and data:
            predictions = data.get('predictions', [])
            stats = data.get('stats', {})
            algorithm = data.get('algorithm', '')
            
            print(f"   Found {len(predictions)} V3 predictions")
            print(f"   Algorithm: {algorithm}")
            
            # Check stats structure
            required_stats = ['total', 'wins', 'losses', 'pending', 'win_rate']
            missing_stats = []
            
            for stat in required_stats:
                if stat in stats:
                    print(f"   ✅ {stat}: {stats[stat]}")
                else:
                    missing_stats.append(stat)
            
            if missing_stats:
                print(f"   ❌ Missing stats: {missing_stats}")
            else:
                print(f"   ✅ All V3 stats present")
            
            # Verify predictions are from enhanced_v3 model
            if predictions:
                v3_predictions = [p for p in predictions if p.get('ai_model') == 'enhanced_v3']
                if len(v3_predictions) == len(predictions):
                    print(f"   ✅ All predictions are from enhanced_v3 model")
                else:
                    print(f"   ⚠️  Mixed models: {len(v3_predictions)}/{len(predictions)} are V3")
        
        # Test with result filter
        success, data = self.run_test("V3 Predictions - Wins Only", "GET", "predictions/v3?result=win", 200)
        if success and data:
            predictions = data.get('predictions', [])
            wins_only = all(p.get('result') == 'win' for p in predictions)
            if wins_only:
                print(f"   ✅ Result filter working - {len(predictions)} wins")
            else:
                print(f"   ⚠️  Result filter not working properly")

    def test_v3_algorithm_comparison(self):
        """Test NEW V2 vs V3 Algorithm Comparison endpoint"""
        print("\n⚖️ Testing V2 vs V3 Algorithm Comparison...")
        
        success, data = self.run_test("Algorithm Comparison", "GET", "predictions/comparison", 200)
        if success and data:
            v2_stats = data.get('v2_legacy', {})
            v3_stats = data.get('v3_enhanced', {})
            recommendation = data.get('recommendation', '')
            
            print(f"   Recommendation: {recommendation}")
            
            # Check V2 stats
            if v2_stats:
                print(f"   V2 Legacy Stats:")
                for key, value in v2_stats.items():
                    print(f"     {key}: {value}")
            else:
                print(f"   ⚠️  No V2 legacy stats found")
            
            # Check V3 stats
            if v3_stats:
                print(f"   V3 Enhanced Stats:")
                for key, value in v3_stats.items():
                    print(f"     {key}: {value}")
            else:
                print(f"   ⚠️  No V3 enhanced stats found")
            
            # Verify required fields in both
            required_fields = ['total', 'wins', 'losses', 'win_rate', 'avg_confidence']
            
            v2_missing = [f for f in required_fields if f not in v2_stats]
            v3_missing = [f for f in required_fields if f not in v3_stats]
            
            if not v2_missing and not v3_missing:
                print(f"   ✅ Both V2 and V3 stats complete")
            else:
                if v2_missing:
                    print(f"   ❌ V2 missing fields: {v2_missing}")
                if v3_missing:
                    print(f"   ❌ V3 missing fields: {v3_missing}")

    def test_upcoming_predictions_window(self):
        """Test NEW Upcoming Predictions Window endpoint"""
        print("\n🕐 Testing Upcoming Predictions Window...")
        
        success, data = self.run_test("Upcoming Predictions Window", "GET", "upcoming-predictions-window", 200)
        if success and data:
            current_time = data.get('current_time', '')
            prediction_window = data.get('prediction_window', {})
            games_in_window = data.get('games_in_window', [])
            upcoming_games = data.get('upcoming_games', [])
            total_in_window = data.get('total_in_window', 0)
            message = data.get('message', '')
            
            print(f"   Current time: {current_time}")
            print(f"   Games in window: {len(games_in_window)}")
            print(f"   Upcoming games: {len(upcoming_games)}")
            print(f"   Total in window: {total_in_window}")
            print(f"   Message: {message}")
            
            # Check prediction window structure
            if prediction_window:
                window_start = prediction_window.get('start', '')
                window_end = prediction_window.get('end', '')
                print(f"   Window: {window_start} to {window_end}")
                
                if window_start and window_end:
                    print(f"   ✅ Prediction window properly defined")
                else:
                    print(f"   ❌ Prediction window incomplete")
            
            # Check games structure
            if games_in_window:
                game = games_in_window[0]
                required_fields = ['event_id', 'sport', 'home_team', 'away_team', 'commence_time', 'minutes_to_start', 'has_v3_prediction']
                missing_fields = [f for f in required_fields if f not in game]
                
                if not missing_fields:
                    print(f"   ✅ Game structure complete")
                    print(f"   Sample: {game.get('home_team')} vs {game.get('away_team')} ({game.get('minutes_to_start')} min)")
                else:
                    print(f"   ❌ Game missing fields: {missing_fields}")
            
            # Verify total matches count
            if len(games_in_window) == total_in_window:
                print(f"   ✅ Total count matches games list")
            else:
                print(f"   ⚠️  Count mismatch: {len(games_in_window)} vs {total_in_window}")

    def test_manual_v3_analysis(self):
        """Test NEW Manual V3 Analysis endpoint"""
        print("\n🔬 Testing Manual V3 Analysis...")
        
        # First get an event to analyze
        success, events = self.run_test("Get Events for V3 Analysis", "GET", "events/basketball_nba?pre_match_only=true", 200)
        if success and events and len(events) > 0:
            event_id = events[0].get('id')
            if event_id:
                print(f"   Analyzing event: {events[0].get('home_team')} vs {events[0].get('away_team')}")
                
                success, data = self.run_test("Manual V3 Analysis", "POST", f"analyze-pregame/{event_id}?sport_key=basketball_nba", 200, timeout=60)
                if success and data:
                    status = data.get('status', '')
                    event_info = data.get('event', '')
                    prediction = data.get('prediction', {})
                    reason = data.get('reason', '')
                    analysis = data.get('analysis', {})
                    
                    print(f"   Status: {status}")
                    print(f"   Event: {event_info}")
                    
                    if status == 'prediction_created':
                        print(f"   ✅ V3 prediction created successfully")
                        if prediction:
                            confidence = prediction.get('confidence', 0)
                            pick = prediction.get('pick', '')
                            edge = prediction.get('edge', 0)
                            print(f"   Pick: {pick} (Confidence: {confidence:.1%}, Edge: {edge:.1f}%)")
                    elif status == 'no_pick':
                        print(f"   ✅ V3 algorithm correctly declined to make pick")
                        print(f"   Reason: {reason}")
                    else:
                        print(f"   ⚠️  Unexpected status: {status}")
                else:
                    print(f"   ❌ Manual V3 analysis failed")
            else:
                print(f"   ⚠️  No event ID found")
        else:
            print(f"   ⚠️  No events available for V3 analysis testing")

    def test_existing_endpoints_still_work(self):
        """Test that existing endpoints still work after V3 updates"""
        print("\n🔧 Testing Existing Endpoints Still Work...")
        
        # Test core endpoints that should still work
        endpoints_to_test = [
            ("Events NBA Pre-match", "GET", "events/basketball_nba?pre_match_only=true"),
            ("Recommendations", "GET", "recommendations"),
            ("Live Scores", "GET", "live-scores"),
            ("Sports List", "GET", "sports"),
            ("Notifications", "GET", "notifications"),
            ("Settings", "GET", "settings")
        ]
        
        for name, method, endpoint in endpoints_to_test:
            success, data = self.run_test(f"Existing - {name}", method, endpoint, 200)
            if not success:
                print(f"   ❌ CRITICAL: Existing endpoint {endpoint} is broken!")
        
        # Test line movement with event ID
        success, events = self.run_test("Get Events for Line Movement", "GET", "events/basketball_nba", 200)
        if success and events and len(events) > 0:
            event_id = events[0].get('id')
            if event_id:
                success, data = self.run_test("Existing - Line Movement", "GET", f"line-movement/{event_id}?sport_key=basketball_nba", 200)
                if not success:
                    print(f"   ❌ CRITICAL: Line movement endpoint is broken!")

    def test_smart_v4_predictions_endpoint(self):
        """Test NEW Smart V4 Predictions endpoint (NO LLM REQUIRED)"""
        print("\n🧠 Testing Smart V4 Predictions Endpoint...")
        
        success, data = self.run_test("Smart V4 Predictions List", "GET", "predictions/smart-v4?limit=20", 200)
        if success and data:
            predictions = data.get('predictions', [])
            stats = data.get('stats', {})
            algorithm = data.get('algorithm', '')
            description = data.get('description', '')
            
            print(f"   Found {len(predictions)} Smart V4 predictions")
            print(f"   Algorithm: {algorithm}")
            print(f"   Description: {description}")
            
            # Check stats structure
            required_stats = ['total', 'wins', 'losses', 'pending', 'win_rate', 'pick_types']
            missing_stats = []
            
            for stat in required_stats:
                if stat in stats:
                    if stat == 'pick_types':
                        pick_types = stats[stat]
                        print(f"   ✅ {stat}: ML={pick_types.get('moneyline', 0)}, Spread={pick_types.get('spread', 0)}, Total={pick_types.get('total', 0)}")
                    else:
                        print(f"   ✅ {stat}: {stats[stat]}")
                else:
                    missing_stats.append(stat)
            
            if missing_stats:
                print(f"   ❌ Missing stats: {missing_stats}")
            else:
                print(f"   ✅ All Smart V4 stats present including pick_types breakdown")
            
            # Verify predictions are from smart_v4 model
            if predictions:
                v4_predictions = [p for p in predictions if p.get('ai_model') == 'smart_v4']
                if len(v4_predictions) == len(predictions):
                    print(f"   ✅ All predictions are from smart_v4 model")
                else:
                    print(f"   ⚠️  Mixed models: {len(v4_predictions)}/{len(predictions)} are Smart V4")
                
                # Check for diverse prediction types
                prediction_types = set(p.get('prediction_type') for p in predictions)
                if len(prediction_types) > 1:
                    print(f"   ✅ Diverse predictions found: {list(prediction_types)}")
                else:
                    print(f"   ℹ️  Prediction types: {list(prediction_types)}")

    def test_smart_v4_algorithm_comparison(self):
        """Test NEW V2 vs V3 vs Smart V4 Algorithm Comparison endpoint"""
        print("\n⚖️ Testing V2 vs V3 vs Smart V4 Algorithm Comparison...")
        
        success, data = self.run_test("Algorithm Comparison (V2/V3/V4)", "GET", "predictions/comparison", 200)
        if success and data:
            v2_stats = data.get('v2_legacy', {})
            v3_stats = data.get('v3_enhanced', {})
            v4_stats = data.get('smart_v4', {})
            description = data.get('description', {})
            
            # Check all three algorithms are present
            algorithms = ['v2_legacy', 'v3_enhanced', 'smart_v4']
            missing_algorithms = []
            
            for algo in algorithms:
                if algo in data:
                    stats = data[algo]
                    print(f"   {algo.upper()} Stats:")
                    for key, value in stats.items():
                        if key == 'pick_types' and isinstance(value, dict):
                            print(f"     {key}: ML={value.get('moneyline', 0)}, Spread={value.get('spread', 0)}, Total={value.get('total', 0)}")
                        else:
                            print(f"     {key}: {value}")
                else:
                    missing_algorithms.append(algo)
            
            if missing_algorithms:
                print(f"   ❌ Missing algorithms: {missing_algorithms}")
            else:
                print(f"   ✅ All three algorithms (V2/V3/Smart V4) present")
            
            # Verify Smart V4 has pick_types breakdown
            if v4_stats and 'pick_types' in v4_stats:
                pick_types = v4_stats['pick_types']
                if all(key in pick_types for key in ['moneyline', 'spread', 'total']):
                    print(f"   ✅ Smart V4 has complete pick_types breakdown")
                else:
                    print(f"   ⚠️  Smart V4 pick_types incomplete")
            
            # Check descriptions
            if description:
                if 'smart_v4' in description:
                    v4_desc = description['smart_v4']
                    if 'no LLM' in v4_desc or 'NO LLM' in v4_desc:
                        print(f"   ✅ Smart V4 description confirms NO LLM requirement")
                    else:
                        print(f"   ⚠️  Smart V4 description missing NO LLM confirmation")

    def test_manual_smart_v4_analysis(self):
        """Test NEW Manual Smart V4 Analysis endpoint (NO LLM REQUIRED)"""
        print("\n🔬 Testing Manual Smart V4 Analysis...")
        
        # First get events to analyze
        success, events = self.run_test("Get Events for Smart V4 Analysis", "GET", "events/basketball_nba?pre_match_only=true", 200)
        if success and events and len(events) > 0:
            # Test with 2-3 different event IDs as requested
            events_to_test = events[:3] if len(events) >= 3 else events
            
            for i, event in enumerate(events_to_test, 1):
                event_id = event.get('id')
                if event_id:
                    print(f"   Testing event {i}: {event.get('home_team')} vs {event.get('away_team')}")
                    
                    success, data = self.run_test(f"Manual Smart V4 Analysis #{i}", "POST", f"analyze-pregame/{event_id}?sport_key=basketball_nba", 200, timeout=60)
                    if success and data:
                        status = data.get('status', '')
                        event_info = data.get('event', '')
                        prediction = data.get('prediction', {})
                        reason = data.get('reason', '')
                        data_sources = data.get('data_sources', {})
                        algorithm = data.get('algorithm', '')
                        
                        print(f"     Status: {status}")
                        print(f"     Algorithm: {algorithm}")
                        
                        # Verify it's Smart V4
                        if algorithm == 'smart_v4':
                            print(f"     ✅ Using Smart V4 algorithm")
                        else:
                            print(f"     ⚠️  Expected smart_v4, got: {algorithm}")
                        
                        if status == 'prediction_created':
                            print(f"     ✅ Smart V4 prediction created successfully")
                            if prediction:
                                confidence = prediction.get('confidence', 0)
                                pick = prediction.get('pick', '')
                                pick_type = prediction.get('pick_type', '')
                                edge_percent = prediction.get('edge_percent', 0)
                                reasoning = prediction.get('reasoning', '')
                                key_factors = prediction.get('key_factors', [])
                                
                                print(f"     Pick: {pick} ({pick_type})")
                                print(f"     Confidence: {confidence:.1%}")
                                print(f"     Edge: {edge_percent:.1f}%")
                                
                                # Check for diverse prediction types
                                if pick_type in ['moneyline', 'spread', 'total']:
                                    print(f"     ✅ Diverse prediction type: {pick_type}")
                                else:
                                    print(f"     ⚠️  Unexpected pick type: {pick_type}")
                                
                                # Check for key factors and reasoning (NO AI errors)
                                if reasoning and 'AI' not in reasoning and 'LLM' not in reasoning:
                                    print(f"     ✅ Reasoning provided (NO AI/LLM errors)")
                                elif 'AI' in reasoning or 'LLM' in reasoning:
                                    print(f"     ❌ CRITICAL: AI/LLM error in reasoning: {reasoning[:100]}...")
                                
                                if key_factors and isinstance(key_factors, list):
                                    print(f"     ✅ Key factors provided: {len(key_factors)} factors")
                                
                        elif status == 'no_pick':
                            print(f"     ✅ Smart V4 correctly declined to make pick")
                            print(f"     Reason: {reason}")
                            
                            # Should NOT return AI errors
                            if 'AI' in reason or 'LLM' in reason or 'unavailable' in reason:
                                print(f"     ❌ CRITICAL: AI/LLM error when NO LLM required: {reason}")
                            else:
                                print(f"     ✅ No AI/LLM errors (as expected)")
                        
                        # Check data sources
                        if data_sources:
                            sources = data_sources.get('multi_book_sources', [])
                            print(f"     Data sources: {len(sources)} bookmaker sources")
                            if sources:
                                print(f"     ✅ Multi-book odds integration working")
                    else:
                        print(f"     ❌ Manual Smart V4 analysis failed for event {i}")
                else:
                    print(f"     ⚠️  No event ID found for event {i}")
        else:
            print(f"   ⚠️  No events available for Smart V4 analysis testing")

    def test_upcoming_predictions_window_enhanced(self):
        """Test Enhanced Upcoming Predictions Window endpoint"""
        print("\n🕐 Testing Enhanced Upcoming Predictions Window...")
        
        success, data = self.run_test("Upcoming Predictions Window", "GET", "upcoming-predictions-window", 200)
        if success and data:
            games_in_window = data.get('games_in_window', [])
            upcoming_games = data.get('upcoming_games', [])
            
            print(f"   Games in prediction window: {len(games_in_window)}")
            print(f"   Upcoming games: {len(upcoming_games)}")
            
            # Check required fields
            required_fields = ['games_in_window', 'upcoming_games']
            missing_fields = []
            
            for field in required_fields:
                if field in data:
                    print(f"   ✅ {field}: {len(data[field])} items")
                else:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"   ❌ Missing fields: {missing_fields}")
            else:
                print(f"   ✅ All required fields present")

    def test_smart_v4_prediction_engine_complete(self):
        """Complete test suite for NEW Smart V4 Prediction Engine (NO LLM REQUIRED)"""
        print("\n🧠 Testing Complete Smart V4 Prediction Engine Suite...")
        
        # Test all Smart V4 endpoints in sequence
        self.test_smart_v4_predictions_endpoint()
        self.test_smart_v4_algorithm_comparison()
        self.test_upcoming_predictions_window_enhanced()
        self.test_manual_smart_v4_analysis()
        
        print(f"\n   ✅ Smart V4 Prediction Engine testing complete")

    def test_data_source_status_api(self):
        """Test Data Source Status API - Line Movement Feature"""
        print("\n📊 Testing Data Source Status API...")
        
        success, data = self.run_test("Data Source Status", "GET", "data-source-status", 200)
        if success and data:
            source = data.get('source', '')
            line_movement_snapshots = data.get('lineMovementSnapshots', 0)
            status = data.get('status', '')
            
            print(f"   Source: {source}")
            print(f"   Status: {status}")
            print(f"   Line Movement Snapshots: {line_movement_snapshots}")
            
            # Verify source is ESPN/DraftKings (not OddsPortal)
            if 'ESPN' in source and 'DraftKings' in source:
                print(f"   ✅ Source correctly shows ESPN/DraftKings")
            else:
                print(f"   ❌ Expected ESPN/DraftKings, got: {source}")
            
            # Verify lineMovementSnapshots count > 0
            if line_movement_snapshots > 0:
                print(f"   ✅ Line movement snapshots available: {line_movement_snapshots}")
            else:
                print(f"   ⚠️  No line movement snapshots found: {line_movement_snapshots}")
            
            # Check other required fields
            required_fields = ['source', 'status', 'lineMovementSnapshots', 'refreshInterval']
            missing_fields = []
            for field in required_fields:
                if field not in data:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"   ❌ Missing fields: {missing_fields}")
            else:
                print(f"   ✅ All required fields present")

    def test_line_movement_api_detailed(self):
        """Test Line Movement API with detailed verification"""
        print("\n📈 Testing Line Movement API (Detailed)...")
        
        # First get events from basketball_nba with pre_match_only=true
        success, events = self.run_test("Get NBA Events for Line Movement", "GET", "events/basketball_nba?pre_match_only=true", 200)
        if success and events and len(events) > 0:
            event = events[0]
            event_id = event.get('id')
            
            print(f"   Testing with event: {event.get('home_team')} vs {event.get('away_team')}")
            print(f"   Event ID: {event_id}")
            
            if event_id:
                # Test line movement API
                success, data = self.run_test("Line Movement Data", "GET", f"line-movement/{event_id}?sport_key=basketball_nba", 200)
                if success and data:
                    # Verify required response fields
                    required_fields = ['event_id', 'event_info', 'opening_odds', 'current_odds', 'bookmakers', 'chart_data', 'total_snapshots']
                    missing_fields = []
                    
                    for field in required_fields:
                        if field in data:
                            if field == 'event_info':
                                event_info = data[field]
                                if event_info and isinstance(event_info, dict):
                                    info_fields = ['home_team', 'away_team', 'commence_time']
                                    missing_info = [f for f in info_fields if f not in event_info]
                                    if not missing_info:
                                        print(f"   ✅ event_info complete: {event_info.get('home_team')} vs {event_info.get('away_team')}")
                                    else:
                                        print(f"   ⚠️  event_info missing: {missing_info}")
                                else:
                                    print(f"   ⚠️  event_info is null or invalid")
                            
                            elif field == 'opening_odds':
                                opening_odds = data[field]
                                if opening_odds and isinstance(opening_odds, dict):
                                    opening_fields = ['home_odds', 'away_odds', 'timestamp']
                                    missing_opening = [f for f in opening_fields if f not in opening_odds]
                                    if not missing_opening:
                                        print(f"   ✅ opening_odds complete: home={opening_odds.get('home_odds')}, away={opening_odds.get('away_odds')}")
                                    else:
                                        print(f"   ⚠️  opening_odds missing: {missing_opening}")
                                else:
                                    print(f"   ℹ️  opening_odds is null (may be normal for new events)")
                            
                            elif field == 'current_odds':
                                current_odds = data[field]
                                if current_odds and isinstance(current_odds, dict):
                                    print(f"   ✅ current_odds: home={current_odds.get('home')}, away={current_odds.get('away')}")
                                else:
                                    print(f"   ℹ️  current_odds is null")
                            
                            elif field == 'chart_data':
                                chart_data = data[field]
                                if isinstance(chart_data, list):
                                    print(f"   ✅ chart_data: {len(chart_data)} data points")
                                    if chart_data:
                                        sample = chart_data[0]
                                        chart_fields = ['timestamp', 'home_odds', 'away_odds']
                                        if all(f in sample for f in chart_fields):
                                            print(f"   ✅ chart_data structure correct")
                                        else:
                                            print(f"   ⚠️  chart_data structure incomplete")
                                else:
                                    print(f"   ⚠️  chart_data is not a list")
                            
                            elif field == 'bookmakers':
                                bookmakers = data[field]
                                if isinstance(bookmakers, list):
                                    print(f"   ✅ bookmakers: {len(bookmakers)} bookmaker sources")
                                else:
                                    print(f"   ⚠️  bookmakers is not a list")
                            
                            elif field == 'total_snapshots':
                                total_snapshots = data[field]
                                print(f"   ✅ total_snapshots: {total_snapshots}")
                            
                            else:
                                print(f"   ✅ {field}: present")
                        else:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        print(f"   ❌ Missing required fields: {missing_fields}")
                    else:
                        print(f"   ✅ All required line movement fields present")
                    
                    # Verify event_id matches
                    if data.get('event_id') == event_id:
                        print(f"   ✅ Event ID matches request")
                    else:
                        print(f"   ❌ Event ID mismatch: expected {event_id}, got {data.get('event_id')}")
                else:
                    print(f"   ❌ Line movement API failed")
            else:
                print(f"   ⚠️  No event ID available")
        else:
            print(f"   ⚠️  No NBA events available for line movement testing")

    def test_line_movement_cleanup_api(self):
        """Test Line Movement Cleanup API"""
        print("\n🧹 Testing Line Movement Cleanup API...")
        
        success, data = self.run_test("Line Movement Cleanup", "POST", "cleanup-line-movement", 200)
        if success and data:
            # Verify required response fields
            required_fields = ['message', 'deleted_history_count', 'deleted_opening_count', 'total_deleted']
            missing_fields = []
            
            for field in required_fields:
                if field in data:
                    value = data[field]
                    print(f"   ✅ {field}: {value}")
                else:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"   ❌ Missing fields: {missing_fields}")
            else:
                print(f"   ✅ All cleanup response fields present")
            
            # Verify message contains expected content
            message = data.get('message', '')
            if 'cleaned up' in message.lower() or 'cleanup' in message.lower():
                print(f"   ✅ Message indicates cleanup operation")
            else:
                print(f"   ⚠️  Unexpected message format: {message}")
            
            # Verify numeric fields are integers
            numeric_fields = ['deleted_history_count', 'deleted_opening_count', 'total_deleted']
            for field in numeric_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, int) and value >= 0:
                        print(f"   ✅ {field} is valid integer: {value}")
                    else:
                        print(f"   ⚠️  {field} is not a valid non-negative integer: {value}")
        else:
            print(f"   ❌ Line movement cleanup API failed")

    def test_manual_odds_refresh_api(self):
        """Test Manual Odds Refresh API"""
        print("\n🔄 Testing Manual Odds Refresh API...")
        
        success, data = self.run_test("Manual Odds Refresh", "POST", "refresh-odds?sport_key=basketball_nba", 200, timeout=30)
        if success and data:
            message = data.get('message', '')
            snapshots_stored = data.get('snapshots_stored', 0)
            source = data.get('source', '')
            
            print(f"   Message: {message}")
            print(f"   Snapshots stored: {snapshots_stored}")
            print(f"   Source: {source}")
            
            # Verify message contains "ESPN"
            if 'ESPN' in message:
                print(f"   ✅ Message contains ESPN reference")
            else:
                print(f"   ❌ Message should contain ESPN reference: {message}")
            
            # Verify snapshots_stored > 0 (if events are available)
            if snapshots_stored > 0:
                print(f"   ✅ Snapshots stored successfully: {snapshots_stored}")
            else:
                print(f"   ⚠️  No snapshots stored (may be normal if no pre-match events): {snapshots_stored}")
            
            # Verify source is ESPN-related
            if source and 'ESPN' in source:
                print(f"   ✅ Source correctly indicates ESPN")
            else:
                print(f"   ⚠️  Source field missing or incorrect: {source}")
            
            # Check required fields
            required_fields = ['message', 'snapshots_stored']
            missing_fields = []
            for field in required_fields:
                if field not in data:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"   ❌ Missing required fields: {missing_fields}")
            else:
                print(f"   ✅ All required refresh fields present")
        else:
            print(f"   ❌ Manual odds refresh API failed")

    def test_line_movement_functionality_complete(self):
        """Complete test suite for Line Movement functionality"""
        print("\n📊 Testing Complete Line Movement Functionality...")
        
        # Test all line movement endpoints in sequence
        self.test_data_source_status_api()
        self.test_line_movement_api_detailed()
        self.test_line_movement_cleanup_api()
        self.test_manual_odds_refresh_api()
        
        print(f"\n   ✅ Line Movement functionality testing complete")

    def test_v3_enhanced_betting_algorithm_complete(self):
        """Complete test suite for NEW Enhanced V3 Betting Algorithm"""
        print("\n🎯 Testing Complete V3 Enhanced Betting Algorithm Suite...")
        
        # Test all V3 endpoints in sequence
        self.test_v3_predictions_endpoint()
        self.test_v3_algorithm_comparison()
        self.test_upcoming_predictions_window()
        self.test_manual_v3_analysis()
        
        print(f"\n   ✅ V3 Enhanced Betting Algorithm testing complete")

    def test_betpredictor_v5_endpoints(self):
        """Test NEW BetPredictor V5 comprehensive analysis endpoints"""
        print("\n🎯 Testing BetPredictor V5 Comprehensive Analysis...")
        
        # 1. First get events from basketball_nba
        success, events = self.run_test("Get NBA Events for V5", "GET", "events/basketball_nba", 200)
        if success and events and len(events) > 0:
            event = events[0]
            event_id = event.get('id')
            home_team = event.get('home_team')
            away_team = event.get('away_team')
            
            print(f"   Testing V5 analysis with: {home_team} vs {away_team}")
            print(f"   Event ID: {event_id}")
            
            if event_id:
                # 2. Test V5 Analysis Endpoint
                success, data = self.run_test("V5 Analysis Endpoint", "POST", f"analyze-v5/{event_id}?sport_key=basketball_nba", 200, timeout=90)
                if success and data:
                    # Verify response structure
                    required_fields = ['event', 'prediction', 'line_movement_analysis', 'data_summary']
                    missing_fields = []
                    
                    for field in required_fields:
                        if field in data:
                            print(f"   ✅ {field}: Present")
                            
                            # Detailed verification for each section
                            if field == 'event':
                                event_data = data[field]
                                event_fields = ['id', 'home_team', 'away_team', 'commence_time']
                                missing_event_fields = [f for f in event_fields if f not in event_data]
                                if not missing_event_fields:
                                    print(f"     ✅ Event data complete")
                                else:
                                    print(f"     ⚠️  Event missing: {missing_event_fields}")
                            
                            elif field == 'prediction':
                                prediction = data[field]
                                pred_fields = ['has_pick', 'reasoning', 'factor_count', 'algorithm']
                                missing_pred_fields = [f for f in pred_fields if f not in prediction]
                                if not missing_pred_fields:
                                    print(f"     ✅ Prediction data complete")
                                    # Verify algorithm is betpredictor_v5
                                    if prediction.get('algorithm') == 'betpredictor_v5':
                                        print(f"     ✅ Algorithm correctly set to betpredictor_v5")
                                    else:
                                        print(f"     ⚠️  Expected betpredictor_v5, got: {prediction.get('algorithm')}")
                                else:
                                    print(f"     ⚠️  Prediction missing: {missing_pred_fields}")
                            
                            elif field == 'line_movement_analysis':
                                line_analysis = data[field]
                                line_fields = ['total_movement_pct', 'movement_direction', 'sharp_money_side', 'key_insights', 'summary', 'phases']
                                missing_line_fields = [f for f in line_fields if f not in line_analysis]
                                if not missing_line_fields:
                                    print(f"     ✅ Line movement analysis complete")
                                else:
                                    print(f"     ⚠️  Line analysis missing: {missing_line_fields}")
                            
                            elif field == 'data_summary':
                                data_summary = data[field]
                                summary_fields = ['line_movement_snapshots', 'has_opening_odds', 'squad_data_available']
                                missing_summary_fields = [f for f in summary_fields if f not in data_summary]
                                if not missing_summary_fields:
                                    print(f"     ✅ Data summary complete")
                                else:
                                    print(f"     ⚠️  Data summary missing: {missing_summary_fields}")
                        else:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        print(f"   ❌ V5 Analysis missing fields: {missing_fields}")
                    else:
                        print(f"   ✅ V5 Analysis response structure complete")
                else:
                    print(f"   ❌ V5 Analysis endpoint failed")
            else:
                print(f"   ⚠️  No event ID available for V5 testing")
        else:
            print(f"   ⚠️  No events available for V5 testing")
        
        # 3. Test V5 Predictions List
        success, data = self.run_test("V5 Predictions List", "GET", "predictions/v5", 200)
        if success and data:
            predictions = data.get('predictions', [])
            stats = data.get('stats', {})
            algorithm = data.get('algorithm', '')
            
            print(f"   Found {len(predictions)} V5 predictions")
            print(f"   Algorithm: {algorithm}")
            
            # Verify stats structure
            required_stats = ['total', 'wins', 'losses', 'pending', 'win_rate', 'avg_confidence']
            missing_stats = []
            
            for stat in required_stats:
                if stat in stats:
                    print(f"   ✅ {stat}: {stats[stat]}")
                else:
                    missing_stats.append(stat)
            
            if missing_stats:
                print(f"   ❌ V5 stats missing: {missing_stats}")
            else:
                print(f"   ✅ V5 predictions stats complete")
            
            # Verify algorithm is betpredictor_v5
            if algorithm == 'betpredictor_v5':
                print(f"   ✅ V5 predictions algorithm correct")
            else:
                print(f"   ⚠️  Expected betpredictor_v5, got: {algorithm}")
        
        # 4. Test Predictions Comparison (should include V5)
        success, data = self.run_test("Predictions Comparison with V5", "GET", "predictions/comparison", 200)
        if success and data:
            # Check if betpredictor_v5 is included
            if 'betpredictor_v5' in data:
                v5_stats = data['betpredictor_v5']
                print(f"   ✅ V5 included in comparison")
                
                # Check V5 description
                description = data.get('description', {})
                if 'betpredictor_v5' in description:
                    v5_desc = description['betpredictor_v5']
                    if 'comprehensive' in v5_desc.lower() or 'line movement' in v5_desc.lower():
                        print(f"   ✅ V5 description includes comprehensive analysis")
                    else:
                        print(f"   ⚠️  V5 description may be incomplete")
                else:
                    print(f"   ⚠️  V5 description missing")
            else:
                print(f"   ❌ betpredictor_v5 not found in comparison")
        
        # 5. Test Line Movement Endpoint (detailed verification)
        success, events = self.run_test("Get Events for Line Movement", "GET", "events/basketball_nba", 200)
        if success and events and len(events) > 0:
            event_id = events[0].get('id')
            if event_id:
                success, data = self.run_test("Line Movement Detailed", "GET", f"line-movement/{event_id}?sport_key=basketball_nba", 200)
                if success and data:
                    chart_data = data.get('chart_data', [])
                    total_snapshots = data.get('total_snapshots', 0)
                    
                    print(f"   Chart data points: {len(chart_data)}")
                    print(f"   Total snapshots: {total_snapshots}")
                    
                    # Verify chart_data has multiple snapshots
                    if len(chart_data) > 1:
                        print(f"   ✅ Multiple line movement snapshots available")
                    else:
                        print(f"   ⚠️  Limited line movement data: {len(chart_data)} points")
                    
                    # Verify total_snapshots matches chart_data length (approximately)
                    if total_snapshots >= len(chart_data):
                        print(f"   ✅ Snapshot count consistent")
                    else:
                        print(f"   ⚠️  Snapshot count mismatch: {total_snapshots} vs {len(chart_data)}")
                else:
                    print(f"   ❌ Line movement endpoint failed")

def main():
    print("🚀 Starting BetPredictor API Testing...")
    print("=" * 60)
    
    tester = BettingPredictorAPITester()
    
    # Run all tests
    print("\n📡 Testing Core API Endpoints...")
    tester.test_root_endpoint()
    tester.test_sports_endpoint()
    tester.test_enhanced_api_usage()  # Test the enhanced API usage endpoint
    
    print("\n📊 Testing Line Movement Functionality (PRIORITY)...")
    tester.test_line_movement_functionality_complete()
    
    print("\n🚀 Testing NEW Smart V4 Prediction Engine (NO LLM REQUIRED)...")
    tester.test_smart_v4_prediction_engine_complete()
    
    print("\n🚀 Testing NEW V3 Enhanced Betting Algorithm...")
    tester.test_v3_enhanced_betting_algorithm_complete()
    
    print("\n🎯 Testing NEW BetPredictor V5 Comprehensive Analysis...")
    tester.test_betpredictor_v5_endpoints()
    
    print("\n🔧 Testing Existing Endpoints Still Work...")
    tester.test_existing_endpoints_still_work()
    
    print("\n🔑 Testing New BetPredictor Enhancements...")
    tester.test_api_key_management()
    tester.test_bankroll_management()
    tester.test_notifications()
    tester.test_settings()
    tester.test_analytics()
    tester.test_export_functionality()
    
    print("\n🏆 Testing NEW Real-Time Score Features...")
    tester.test_espn_scores_integration()
    tester.test_live_scores()
    tester.test_pending_results()
    tester.test_recommendations_70_percent_filter()
    tester.test_check_results_trigger()
    tester.test_performance_stats_updated()
    
    print("\n🎯 Testing ENHANCED BetPredictor Features...")
    tester.test_all_markets_ml_spread_total()
    tester.test_pre_match_only_filter()
    tester.test_continuous_score_sync()
    tester.test_line_movement_cleanup()
    tester.test_performance_stats_enhanced()
    tester.test_espn_scores_api_enhanced()
    tester.test_pending_results_enhanced()
    tester.test_recommendations_70_percent_enhanced()
    
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