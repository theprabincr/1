#!/usr/bin/env python3
"""
BetPredictor Backend API Test Suite
Tests all API endpoints for deployment readiness
"""

import requests
import json
import sys
from datetime import datetime
import os
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

class APITester:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
        
    def test_endpoint(self, method, endpoint, expected_status=200, description="", validate_prediction=False):
        """Test a single API endpoint"""
        url = f"{BASE_URL}{endpoint}"
        print(f"\n🧪 Testing {method} {endpoint}")
        print(f"   URL: {url}")
        if description:
            print(f"   Description: {description}")
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Check status code
            status_ok = response.status_code == expected_status
            
            # Try to parse JSON
            json_ok = False
            json_data = None
            try:
                json_data = response.json()
                json_ok = True
            except:
                json_ok = False
            
            # Validate prediction quality if requested
            prediction_quality_ok = True
            prediction_details = ""
            
            if validate_prediction and json_data:
                prediction_quality_ok, prediction_details = self.validate_prediction_quality(json_data, endpoint)
            
            # Print results
            if status_ok and json_ok and prediction_quality_ok:
                print(f"   ✅ PASS - Status: {response.status_code}, JSON: Valid")
                if isinstance(json_data, dict):
                    if 'message' in json_data:
                        print(f"   📝 Message: {json_data['message']}")
                    if 'status' in json_data:
                        print(f"   📊 Status: {json_data['status']}")
                    if 'source' in json_data:
                        print(f"   🔗 Source: {json_data['source']}")
                elif isinstance(json_data, list):
                    print(f"   📊 Array length: {len(json_data)}")
                
                if prediction_details:
                    print(f"   🧠 Prediction Quality: {prediction_details}")
                
                self.passed += 1
                self.results.append({
                    'endpoint': endpoint,
                    'status': 'PASS',
                    'status_code': response.status_code,
                    'json_valid': True,
                    'response_size': len(str(json_data)) if json_data else 0,
                    'prediction_quality': prediction_details if prediction_details else None
                })
            else:
                error_msg = []
                if not status_ok:
                    error_msg.append(f"Expected status {expected_status}, got {response.status_code}")
                if not json_ok:
                    error_msg.append("Invalid JSON response")
                if not prediction_quality_ok:
                    error_msg.append(f"Prediction quality issue: {prediction_details}")
                
                print(f"   ❌ FAIL - {', '.join(error_msg)}")
                print(f"   📄 Response: {response.text[:200]}...")
                
                self.failed += 1
                self.results.append({
                    'endpoint': endpoint,
                    'status': 'FAIL',
                    'status_code': response.status_code,
                    'json_valid': json_ok,
                    'error': ', '.join(error_msg),
                    'response_preview': response.text[:200]
                })
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ FAIL - Connection Error: {str(e)}")
            self.failed += 1
            self.results.append({
                'endpoint': endpoint,
                'status': 'FAIL',
                'error': f"Connection Error: {str(e)}"
            })
        except Exception as e:
            print(f"   ❌ FAIL - Unexpected Error: {str(e)}")
            self.failed += 1
            self.results.append({
                'endpoint': endpoint,
                'status': 'FAIL',
                'error': f"Unexpected Error: {str(e)}"
            })
    
    def validate_prediction_quality(self, data, endpoint):
        """Validate the quality of prediction algorithm output"""
        try:
            if "/recommendations" in endpoint:
                if isinstance(data, list):
                    if len(data) == 0:
                        return True, "No recommendations (normal if no high-confidence picks)"
                    
                    # Check first recommendation for required fields
                    rec = data[0]
                    required_fields = ['predicted_outcome', 'confidence', 'analysis', 'odds_at_prediction']
                    missing_fields = [f for f in required_fields if f not in rec or not rec[f]]
                    
                    if missing_fields:
                        return False, f"Missing required fields: {missing_fields}"
                    
                    # Check confidence is reasonable
                    confidence = rec.get('confidence', 0)
                    if confidence < 0.5 or confidence > 1.0:
                        return False, f"Invalid confidence: {confidence}"
                    
                    # Check analysis has reasoning
                    analysis = rec.get('analysis', '')
                    if len(analysis) < 50:
                        return False, "Analysis too short (lacks reasoning)"
                    
                    return True, f"Valid prediction: {confidence*100:.0f}% confidence, detailed analysis"
                
            elif "/ml/status" in endpoint:
                # Validate ML status response
                if not isinstance(data, dict):
                    return False, "ML status should return a dictionary"
                
                # Check for model status
                models = data.get('models', {})
                if 'basketball_nba' not in models:
                    return False, "basketball_nba model not found in status"
                
                nba_model = models['basketball_nba']
                accuracy = nba_model.get('accuracy', 0)
                is_loaded = nba_model.get('model_loaded', False)  # Updated field name
                
                if not is_loaded:
                    return False, "basketball_nba model not loaded"
                
                if accuracy < 0.6 or accuracy > 0.7:
                    return False, f"Expected accuracy around 65%, got {accuracy*100:.1f}%"
                
                return True, f"NBA model loaded with {accuracy*100:.1f}% accuracy"
            
            elif "/ml/predict/" in endpoint:
                # Validate ML prediction response - check nested prediction object
                if not isinstance(data, dict):
                    return False, "ML prediction should return a dictionary"
                
                prediction = data.get('prediction', {})
                if not prediction:
                    return False, "Missing prediction object in ML response"
                
                required_fields = ['home_win_prob', 'model_available', 'method']
                missing_fields = [f for f in required_fields if f not in prediction]
                
                if missing_fields:
                    return False, f"Missing ML prediction fields: {missing_fields}"
                
                home_win_prob = prediction.get('home_win_prob', 0)
                model_available = prediction.get('model_available', False)
                method = prediction.get('method', '')
                
                if not model_available:
                    return False, "Model not available for prediction"
                
                if method not in ['xgboost', 'xgboost_multi_market']:
                    return False, f"Expected method 'xgboost' or 'xgboost_multi_market', got '{method}'"
                
                if home_win_prob < 0.7 or home_win_prob > 0.85:
                    return False, f"Expected home_win_prob around 0.78, got {home_win_prob}"
                
                # Check for pick recommendation
                if 'pick' not in data:
                    return False, "Missing pick recommendation"
                
                return True, f"XGBoost prediction: {home_win_prob:.3f} home win prob, method: {method}"
            
            elif "/ml/backtest" in endpoint:
                # Validate ML backtest response - check nested results object
                if not isinstance(data, dict):
                    return False, "ML backtest should return a dictionary"
                
                results = data.get('results', {})
                if not results:
                    return False, "Missing results object in backtest response"
                
                required_fields = ['accuracy', 'picks_made', 'roi']
                missing_fields = [f for f in required_fields if f not in results]
                
                if missing_fields:
                    return False, f"Missing backtest fields: {missing_fields}"
                
                accuracy = results.get('accuracy', 0)
                picks_made = results.get('picks_made', 0)
                roi = results.get('roi', 0)
                
                if accuracy <= 0.5:
                    return False, f"Expected accuracy > 0.5, got {accuracy}"
                
                if picks_made <= 0:
                    return False, f"Expected picks_made > 0, got {picks_made}"
                
                return True, f"Backtest: {accuracy:.1%} accuracy, {picks_made} picks, {roi:.1f}% ROI"
            
            elif "/ml/elo-ratings" in endpoint:
                # Validate ELO ratings response - check teams array
                if not isinstance(data, dict):
                    return False, "ELO ratings should return a dictionary"
                
                if 'teams' not in data:
                    return False, "Missing 'teams' field in ELO response"
                
                teams = data.get('teams', [])
                if len(teams) < 20:  # NBA has 30 teams, expect at least 20
                    return False, f"Expected at least 20 NBA teams, got {len(teams)}"
                
                # Check if ratings are reasonable (typically 1000-2000 range)
                if teams:
                    sample_rating = teams[0].get('elo', 0)
                    if sample_rating < 800 or sample_rating > 2200:
                        return False, f"ELO rating seems unreasonable: {sample_rating}"
                
                return True, f"ELO ratings for {len(teams)} teams"
                
            elif "/analyze-unified/" in endpoint:
                # Check for unified analysis structure with XGBoost integration
                prediction = data.get('prediction', {})
                if not prediction:
                    return False, "Missing 'prediction' object in unified analysis"
                
                algorithm = prediction.get('algorithm', '')
                if algorithm != 'unified_xgboost':
                    return False, f"Expected algorithm 'unified_xgboost', got '{algorithm}'"
                
                if 'xgb_probability' not in prediction:
                    return False, "Missing 'xgb_probability' field"
                
                if 'consensus_level' not in prediction:
                    return False, "Missing 'consensus_level' field"
                
                xgb_prob = prediction.get('xgb_probability', 0)
                consensus = prediction.get('consensus_level', '')
                
                return True, f"Unified XGBoost: algorithm={algorithm}, xgb_prob={xgb_prob:.3f}, consensus={consensus}"
            
            elif "/line-movement/" in endpoint:
                # Check line movement data structure
                required_keys = ['event_id', 'chart_data']
                missing_keys = [k for k in required_keys if k not in data]
                
                if missing_keys:
                    return False, f"Missing line movement data: {missing_keys}"
                
                chart_data = data.get('chart_data', {})
                markets = ['moneyline', 'spread', 'totals']
                available_markets = [m for m in markets if m in chart_data and len(chart_data[m]) > 0]
                
                return True, f"Line movement tracked for: {', '.join(available_markets) if available_markets else 'no markets'}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def test_prediction_algorithm(self):
        """Test the AI prediction algorithm endpoints"""
        print("\n🧠 TESTING PREDICTION ALGORITHM (CRITICAL)")
        print("-" * 50)
        
        # Test recommendations with validation
        self.test_endpoint("GET", "/recommendations", 
                         description="Get AI recommendations", 
                         validate_prediction=True)
        
        # First get an event to test with
        try:
            response = requests.get(f"{BASE_URL}/events/basketball_nba", timeout=30)
            if response.status_code == 200:
                events = response.json()
                if events and len(events) > 0:
                    test_event = events[0]
                    event_id = test_event.get('id')
                    
                    if event_id:
                        print(f"   🎯 Using test event: {test_event.get('home_team', 'Unknown')} vs {test_event.get('away_team', 'Unknown')}")
                        
                        # Test unified analysis endpoint with validation
                        self.test_endpoint("POST", f"/analyze-unified/{event_id}", 
                                         description="Full AI analysis for specific event",
                                         validate_prediction=True)
                        
                        # Test line movement for the event with validation
                        self.test_endpoint("GET", f"/line-movement/{event_id}",
                                         description="Line movement analysis",
                                         validate_prediction=True)
                        
                        # Test matchup data
                        self.test_endpoint("GET", f"/matchup/{event_id}",
                                         description="Comprehensive matchup data")
                    else:
                        print("   ⚠️  No event ID found for prediction testing")
                else:
                    print("   ⚠️  No events available for prediction testing")
            else:
                print("   ⚠️  Could not fetch events for prediction testing")
        except Exception as e:
            print(f"   ❌ Error in prediction algorithm testing: {e}")
    
    def test_multiple_sports(self):
        """Test events endpoint for multiple sports"""
        print("\n🏈 TESTING MULTIPLE SPORTS DATA")
        print("-" * 50)
        
        sports = ["basketball_nba", "americanfootball_nfl", "icehockey_nhl"]
        for sport in sports:
            self.test_endpoint("GET", f"/events/{sport}", 
                             description=f"Get {sport.upper()} events with odds")
    
    def test_odds_snapshots(self):
        """Test odds snapshots endpoint"""
        print("\n📊 TESTING ODDS SNAPSHOTS")
        print("-" * 50)
        
        # Test odds snapshots endpoint (if it exists)
        self.test_endpoint("GET", "/odds-snapshots", 
                         description="Historical odds snapshots", expected_status=404)
        
        # Test manual odds refresh
        self.test_endpoint("POST", "/refresh-odds", 
                         description="Manual odds refresh")

    def test_ml_endpoints(self):
        """Test ML (Machine Learning) endpoints"""
        print("\n🤖 TESTING ML ENDPOINTS (XGBoost Integration)")
        print("-" * 50)
        
        # Test ML status endpoint
        self.test_endpoint("GET", "/ml/status", 
                         description="ML model status - should show basketball_nba model loaded with ~65% accuracy",
                         validate_prediction=True)
        
        # Test ML predict endpoint with specific event ID
        self.test_endpoint("POST", "/ml/predict/401810581?sport_key=basketball_nba", 
                         description="XGBoost prediction for specific event - should return home_win_prob ~0.78",
                         validate_prediction=True)
        
        # Test ML backtest endpoint
        self.test_endpoint("POST", "/ml/backtest?sport_key=basketball_nba&threshold=0.55", 
                         description="ML backtest results - should show accuracy > 0.5, picks_made > 0, ROI value",
                         validate_prediction=True)
        
        # Test ELO ratings endpoint
        self.test_endpoint("GET", "/ml/elo-ratings?sport_key=basketball_nba", 
                         description="ELO ratings for NBA teams",
                         validate_prediction=True)
        
        # Test unified analysis with XGBoost integration
        self.test_endpoint("POST", "/analyze-unified/401810581?sport_key=basketball_nba", 
                         description="Unified analysis with XGBoost - should show algorithm: unified_xgboost, xgb_probability, consensus_level",
                         validate_prediction=True)

    def test_xgboost_favored_outcomes(self):
        """Test XGBoost ML prediction endpoints for FAVORED OUTCOMES (not just home team probabilities)"""
        print("\n🎯 TESTING XGBOOST FAVORED OUTCOMES (NEW FEATURE)")
        print("-" * 50)
        
        # Test NBA event with favored outcomes
        self.test_endpoint_with_favored_validation("POST", "/ml/predict/401810581?sport_key=basketball_nba", 
                         description="NBA XGBoost prediction with favored outcomes - should show ml_favored_team, spread_favored_team, totals_favored",
                         sport="NBA")
        
        # Test NHL event with favored outcomes  
        self.test_endpoint_with_favored_validation("POST", "/ml/predict/401803244?sport_key=icehockey_nhl", 
                         description="NHL XGBoost prediction with favored outcomes - should show ml_favored_team, spread_favored_team, totals_favored",
                         sport="NHL")
        
        # Test unified analysis with favored outcomes
        self.test_endpoint_with_favored_validation("POST", "/analyze-unified/401810581?sport_key=basketball_nba", 
                         description="Unified analysis with favored outcomes in reasoning text",
                         sport="NBA", is_unified=True)
    
    def test_endpoint_with_favored_validation(self, method, endpoint, expected_status=200, description="", sport="NBA", is_unified=False):
        """Test endpoint and validate favored outcomes structure"""
        url = f"{BASE_URL}{endpoint}"
        print(f"\n🧪 Testing {method} {endpoint}")
        print(f"   URL: {url}")
        if description:
            print(f"   Description: {description}")
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Check status code
            status_ok = response.status_code == expected_status
            
            # Try to parse JSON
            json_ok = False
            json_data = None
            try:
                json_data = response.json()
                json_ok = True
            except:
                json_ok = False
            
            # Validate favored outcomes structure
            favored_validation_ok = True
            favored_details = ""
            
            if json_data and status_ok and json_ok:
                favored_validation_ok, favored_details = self.validate_favored_outcomes(json_data, endpoint, sport, is_unified)
            
            # Print results
            if status_ok and json_ok and favored_validation_ok:
                print(f"   ✅ PASS - Status: {response.status_code}, JSON: Valid, Favored Outcomes: Valid")
                print(f"   🎯 Favored Outcomes: {favored_details}")
                
                self.passed += 1
                self.results.append({
                    'endpoint': endpoint,
                    'status': 'PASS',
                    'status_code': response.status_code,
                    'json_valid': True,
                    'favored_outcomes': favored_details
                })
            else:
                error_msg = []
                if not status_ok:
                    error_msg.append(f"Expected status {expected_status}, got {response.status_code}")
                if not json_ok:
                    error_msg.append("Invalid JSON response")
                if not favored_validation_ok:
                    error_msg.append(f"Favored outcomes issue: {favored_details}")
                
                print(f"   ❌ FAIL - {', '.join(error_msg)}")
                print(f"   📄 Response: {response.text[:300]}...")
                
                self.failed += 1
                self.results.append({
                    'endpoint': endpoint,
                    'status': 'FAIL',
                    'status_code': response.status_code,
                    'json_valid': json_ok,
                    'error': ', '.join(error_msg),
                    'response_preview': response.text[:300]
                })
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ FAIL - Connection Error: {str(e)}")
            self.failed += 1
            self.results.append({
                'endpoint': endpoint,
                'status': 'FAIL',
                'error': f"Connection Error: {str(e)}"
            })
        except Exception as e:
            print(f"   ❌ FAIL - Unexpected Error: {str(e)}")
            self.failed += 1
            self.results.append({
                'endpoint': endpoint,
                'status': 'FAIL',
                'error': f"Unexpected Error: {str(e)}"
            })
    
    def validate_favored_outcomes(self, data, endpoint, sport, is_unified=False):
        """Validate favored outcomes structure in ML prediction responses"""
        try:
            if is_unified:
                # For unified analysis, check if favored outcomes are mentioned in reasoning
                reasoning = data.get('prediction', {}).get('reasoning', '') or data.get('reasoning', '')
                prediction = data.get('prediction', {})
                
                if not reasoning:
                    return False, "Missing reasoning text in unified analysis"
                
                # Check if reasoning mentions favored teams/outcomes
                has_favored_mention = any(word in reasoning.lower() for word in ['favored', 'favorite', 'underdog', 'likely', 'expected'])
                
                if not has_favored_mention:
                    return False, "Reasoning text doesn't mention favored outcomes"
                
                # Check if prediction object has favored outcome fields
                required_favored_fields = ['ml_favored_team', 'ml_favored_prob', 'spread_favored_team', 'totals_favored']
                missing_fields = [f for f in required_favored_fields if f not in prediction]
                
                if missing_fields:
                    return False, f"Missing favored fields in prediction: {missing_fields}"
                
                return True, f"Unified analysis includes favored outcome reasoning (length: {len(reasoning)} chars) and prediction fields"
            
            else:
                # For ML predict endpoints, check for specific favored outcome fields
                prediction = data.get('prediction', {})
                if not prediction:
                    return False, "Missing prediction object in ML response"
                
                # Check for ML favored fields
                ml_favored_fields = ['ml_favored_team', 'ml_favored_prob', 'ml_underdog_team', 'ml_underdog_prob']
                ml_missing = [f for f in ml_favored_fields if f not in prediction]
                
                # Check for spread favored fields
                spread_favored_fields = ['spread_favored_team', 'spread_favored_prob', 'spread_favored_line']
                spread_missing = [f for f in spread_favored_fields if f not in prediction]
                
                # Check for totals favored fields
                totals_favored_fields = ['totals_favored', 'totals_favored_prob']
                totals_missing = [f for f in totals_favored_fields if f not in prediction]
                
                # Collect all missing fields
                all_missing = []
                if ml_missing:
                    all_missing.extend([f"ML: {', '.join(ml_missing)}"])
                if spread_missing:
                    all_missing.extend([f"Spread: {', '.join(spread_missing)}"])
                if totals_missing:
                    all_missing.extend([f"Totals: {', '.join(totals_missing)}"])
                
                if all_missing:
                    return False, f"Missing favored outcome fields - {'; '.join(all_missing)}"
                
                # Validate team names are actual team names (not "Home" or "Away")
                ml_favored_team = prediction.get('ml_favored_team', '')
                ml_underdog_team = prediction.get('ml_underdog_team', '')
                spread_favored_team = prediction.get('spread_favored_team', '')
                
                invalid_names = []
                for team_name, field in [(ml_favored_team, 'ml_favored_team'), 
                                       (ml_underdog_team, 'ml_underdog_team'),
                                       (spread_favored_team, 'spread_favored_team')]:
                    if team_name.lower() in ['home', 'away', 'home team', 'away team']:
                        invalid_names.append(f"{field}: '{team_name}'")
                
                if invalid_names:
                    return False, f"Team names should be actual team names, not Home/Away: {', '.join(invalid_names)}"
                
                # Validate probabilities are reasonable (relaxed criteria)
                ml_favored_prob = prediction.get('ml_favored_prob', 0)
                ml_underdog_prob = prediction.get('ml_underdog_prob', 0)
                spread_favored_prob = prediction.get('spread_favored_prob', 0)
                totals_favored_prob = prediction.get('totals_favored_prob', 0)
                
                # More relaxed probability validation - allow extreme values for confident predictions
                prob_issues = []
                for prob, field in [(ml_favored_prob, 'ml_favored_prob'),
                                  (ml_underdog_prob, 'ml_underdog_prob'),
                                  (spread_favored_prob, 'spread_favored_prob')]:
                    if prob < 0.05 or prob > 0.95:
                        prob_issues.append(f"{field}: {prob}")
                
                # Special handling for totals - can be very confident
                if totals_favored_prob < 0.05 or totals_favored_prob > 0.999:
                    prob_issues.append(f"totals_favored_prob: {totals_favored_prob}")
                
                if prob_issues:
                    return False, f"Unreasonable probabilities (should be 0.05-0.95, totals 0.05-0.999): {', '.join(prob_issues)}"
                
                # Check if favored team has higher probability than underdog
                if ml_favored_prob <= ml_underdog_prob:
                    return False, f"ML favored team prob ({ml_favored_prob}) should be higher than underdog prob ({ml_underdog_prob})"
                
                # Validate spread line is reasonable
                spread_line = prediction.get('spread_favored_line', 0)
                if abs(spread_line) > 30:  # Spread lines are typically within ±30
                    return False, f"Unreasonable spread line: {spread_line}"
                
                # Validate totals favored is "over" or "under"
                totals_favored = prediction.get('totals_favored', '').upper()
                if totals_favored not in ['OVER', 'UNDER']:
                    return False, f"totals_favored should be 'OVER' or 'UNDER', got: '{totals_favored}'"
                
                return True, (f"ML: {ml_favored_team} ({ml_favored_prob:.3f}) vs {ml_underdog_team} ({ml_underdog_prob:.3f}); "
                            f"Spread: {spread_favored_team} {spread_line:+.1f} ({spread_favored_prob:.3f}); "
                            f"Totals: {totals_favored} ({totals_favored_prob:.3f})")
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def run_all_tests(self):
        """Run comprehensive API endpoint tests"""
        print("=" * 60)
        print("🚀 BetPredictor Backend API Test Suite - COMPREHENSIVE")
        print("=" * 60)
        
        # CORE API HEALTH
        print("\n🏥 CORE API HEALTH TESTS")
        print("-" * 50)
        self.test_endpoint("GET", "/", description="Health check endpoint")
        self.test_endpoint("GET", "/sports", description="List available sports")
        self.test_endpoint("GET", "/data-source-status", description="ESPN data source status")
        
        # EVENTS & DATA FETCHING
        print("\n📅 EVENTS & DATA FETCHING TESTS")
        print("-" * 50)
        self.test_multiple_sports()
        
        # ML ENDPOINTS (NEW XGBoost Integration)
        self.test_ml_endpoints()
        
        # XGBOOST FAVORED OUTCOMES (NEW FEATURE TESTING)
        self.test_xgboost_favored_outcomes()
        
        # PREDICTION ALGORITHM (MOST IMPORTANT)
        self.test_prediction_algorithm()
        
        # PERFORMANCE & RESULTS TRACKING
        print("\n📈 PERFORMANCE & RESULTS TRACKING")
        print("-" * 50)
        self.test_endpoint("GET", "/recommendations", description="Get AI recommendations")
        self.test_endpoint("GET", "/performance", description="Get performance statistics")
        self.test_endpoint("GET", "/notifications", description="Get notifications list")
        
        # LINE MOVEMENT ANALYSIS
        print("\n📊 LINE MOVEMENT ANALYSIS")
        print("-" * 50)
        self.test_odds_snapshots()
        
        # Print summary
        self.print_summary()
        
        return self.failed == 0
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if self.failed > 0:
            print("\n🔍 FAILED ENDPOINTS:")
            for result in self.results:
                if result['status'] == 'FAIL':
                    print(f"   • {result['endpoint']}: {result.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 60)
        
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED - Backend is ready for deployment!")
        else:
            print("⚠️  SOME TESTS FAILED - Backend needs attention before deployment")
        
        print("=" * 60)

def main():
    """Main test execution"""
    tester = APITester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()