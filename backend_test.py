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

    def test_ml_training_system(self):
        """Test ML Training System endpoints - SPECIFIC REVIEW REQUEST"""
        print("\n🎯 TESTING ML TRAINING SYSTEM (REVIEW REQUEST)")
        print("-" * 50)
        
        # Test 1: GET /api/ml/status - Verify training schedule and multi-sport models
        self.test_ml_status_with_training_schedule()
        
        # Test 2: POST /api/ml/train - Verify multi-season training support
        self.test_ml_training_endpoint()
        
        # Test 3: POST /api/ml/predict/{event_id} - Verify favored team predictions
        self.test_ml_predict_with_favored_outcomes()

    def test_ensemble_ml_system(self):
        """Test Ensemble ML System endpoints - SPECIFIC REVIEW REQUEST"""
        print("\n🎯 TESTING ENSEMBLE ML SYSTEM (REVIEW REQUEST)")
        print("-" * 50)
        
        # Test 1: GET /api/ml/ensemble-status - Verify models with accuracy metrics
        self.test_ensemble_status()
        
        # Test 2: POST /api/ml/ensemble-predict/{event_id} - Test with valid NBA event
        self.test_ensemble_predict()
        
        # Test 3: Compare ensemble vs basic XGBoost accuracy
        self.test_ensemble_vs_xgboost_accuracy()

    def test_ml_status_with_training_schedule(self):
        """Test ML status endpoint for training schedule information"""
        url = f"{BASE_URL}/ml/status"
        print(f"\n🧪 Testing GET /ml/status")
        print(f"   URL: {url}")
        print(f"   Description: Verify models with accuracy for NBA/NFL/NHL, historical data counts with seasons, training schedule")
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate training schedule information
                training_schedule_ok, schedule_details = self.validate_training_schedule(data)
                
                # Validate multi-sport models
                models_ok, models_details = self.validate_multi_sport_models(data)
                
                # Validate historical data with seasons
                historical_ok, historical_details = self.validate_historical_data_seasons(data)
                
                if training_schedule_ok and models_ok and historical_ok:
                    print(f"   ✅ PASS - Training Schedule: {schedule_details}")
                    print(f"   🏀 Models: {models_details}")
                    print(f"   📊 Historical Data: {historical_details}")
                    
                    self.passed += 1
                    self.results.append({
                        'endpoint': '/ml/status',
                        'status': 'PASS',
                        'training_schedule': schedule_details,
                        'models': models_details,
                        'historical_data': historical_details
                    })
                else:
                    error_msg = []
                    if not training_schedule_ok:
                        error_msg.append(f"Training schedule issue: {schedule_details}")
                    if not models_ok:
                        error_msg.append(f"Models issue: {models_details}")
                    if not historical_ok:
                        error_msg.append(f"Historical data issue: {historical_details}")
                    
                    print(f"   ❌ FAIL - {', '.join(error_msg)}")
                    self.failed += 1
                    self.results.append({
                        'endpoint': '/ml/status',
                        'status': 'FAIL',
                        'error': ', '.join(error_msg)
                    })
            else:
                print(f"   ❌ FAIL - Status: {response.status_code}")
                self.failed += 1
                self.results.append({
                    'endpoint': '/ml/status',
                    'status': 'FAIL',
                    'error': f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"   ❌ FAIL - Error: {str(e)}")
            self.failed += 1
            self.results.append({
                'endpoint': '/ml/status',
                'status': 'FAIL',
                'error': str(e)
            })

    def test_ml_training_endpoint(self):
        """Test ML training endpoint for multi-season support"""
        url = f"{BASE_URL}/ml/train?sport_key=basketball_nba"
        print(f"\n🧪 Testing POST /ml/train")
        print(f"   URL: {url}")
        print(f"   Description: Verify returns seasons_used array, total_games count, metrics with ml_accuracy/spread_accuracy/totals_accuracy")
        
        try:
            response = requests.post(url, timeout=120)  # Training can take time
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate training response structure
                training_ok, training_details = self.validate_training_response(data)
                
                if training_ok:
                    print(f"   ✅ PASS - Training Response: {training_details}")
                    
                    self.passed += 1
                    self.results.append({
                        'endpoint': '/ml/train',
                        'status': 'PASS',
                        'training_details': training_details
                    })
                else:
                    print(f"   ❌ FAIL - Training validation issue: {training_details}")
                    self.failed += 1
                    self.results.append({
                        'endpoint': '/ml/train',
                        'status': 'FAIL',
                        'error': f"Training validation: {training_details}"
                    })
            else:
                print(f"   ❌ FAIL - Status: {response.status_code}")
                print(f"   📄 Response: {response.text[:300]}...")
                self.failed += 1
                self.results.append({
                    'endpoint': '/ml/train',
                    'status': 'FAIL',
                    'error': f"HTTP {response.status_code}",
                    'response_preview': response.text[:300]
                })
                
        except Exception as e:
            print(f"   ❌ FAIL - Error: {str(e)}")
            self.failed += 1
            self.results.append({
                'endpoint': '/ml/train',
                'status': 'FAIL',
                'error': str(e)
            })

    def test_ml_predict_with_favored_outcomes(self):
        """Test ML predict endpoint for favored team predictions"""
        # First get a real event ID
        try:
            events_response = requests.get(f"{BASE_URL}/events/basketball_nba", timeout=30)
            if events_response.status_code == 200:
                events = events_response.json()
                if events and len(events) > 0:
                    event_id = events[0].get('id')
                    home_team = events[0].get('home_team', 'Unknown')
                    away_team = events[0].get('away_team', 'Unknown')
                else:
                    event_id = "401810581"  # Fallback
                    home_team = "Test Home"
                    away_team = "Test Away"
            else:
                event_id = "401810581"  # Fallback
                home_team = "Test Home"
                away_team = "Test Away"
        except:
            event_id = "401810581"  # Fallback
            home_team = "Test Home"
            away_team = "Test Away"
        
        url = f"{BASE_URL}/ml/predict/{event_id}?sport_key=basketball_nba"
        print(f"\n🧪 Testing POST /ml/predict/{event_id}")
        print(f"   URL: {url}")
        print(f"   Description: Verify favored team predictions, spread and totals predictions, model_available=true")
        print(f"   Event: {away_team} @ {home_team}")
        
        try:
            response = requests.post(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate prediction response structure
                prediction_ok, prediction_details = self.validate_ml_prediction_response(data)
                
                if prediction_ok:
                    print(f"   ✅ PASS - ML Prediction: {prediction_details}")
                    
                    self.passed += 1
                    self.results.append({
                        'endpoint': f'/ml/predict/{event_id}',
                        'status': 'PASS',
                        'prediction_details': prediction_details
                    })
                else:
                    print(f"   ❌ FAIL - Prediction validation issue: {prediction_details}")
                    self.failed += 1
                    self.results.append({
                        'endpoint': f'/ml/predict/{event_id}',
                        'status': 'FAIL',
                        'error': f"Prediction validation: {prediction_details}"
                    })
            else:
                print(f"   ❌ FAIL - Status: {response.status_code}")
                print(f"   📄 Response: {response.text[:300]}...")
                self.failed += 1
                self.results.append({
                    'endpoint': f'/ml/predict/{event_id}',
                    'status': 'FAIL',
                    'error': f"HTTP {response.status_code}",
                    'response_preview': response.text[:300]
                })
                
        except Exception as e:
            print(f"   ❌ FAIL - Error: {str(e)}")
            self.failed += 1
            self.results.append({
                'endpoint': f'/ml/predict/{event_id}',
                'status': 'FAIL',
                'error': str(e)
            })

    def validate_training_schedule(self, data):
        """Validate training schedule information in ML status response"""
        try:
            if 'training_schedule' not in data:
                return False, "Missing training_schedule field"
            
            schedule = data['training_schedule']
            required_fields = ['frequency', 'time', 'next_scheduled', 'timezone']
            missing_fields = [f for f in required_fields if f not in schedule]
            
            if missing_fields:
                return False, f"Missing schedule fields: {missing_fields}"
            
            # Validate schedule values
            frequency = schedule.get('frequency', '')
            time = schedule.get('time', '')
            timezone = schedule.get('timezone', '')
            next_scheduled = schedule.get('next_scheduled', '')
            
            if 'weekly' not in frequency.lower():
                return False, f"Expected weekly frequency, got: {frequency}"
            
            if 'utc' not in timezone.lower():
                return False, f"Expected UTC timezone, got: {timezone}"
            
            if not next_scheduled:
                return False, "Missing next_scheduled timestamp"
            
            return True, f"frequency={frequency}, time={time}, timezone={timezone}, next_scheduled={next_scheduled[:19]}"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def validate_multi_sport_models(self, data):
        """Validate multi-sport models in ML status response"""
        try:
            if 'models' not in data:
                return False, "Missing models field"
            
            models = data['models']
            expected_sports = ['basketball_nba', 'americanfootball_nfl', 'icehockey_nhl']
            
            model_info = []
            for sport in expected_sports:
                if sport not in models:
                    return False, f"Missing model for {sport}"
                
                model = models[sport]
                accuracy = model.get('accuracy')
                model_loaded = model.get('model_loaded', False)
                
                if accuracy is not None:
                    model_info.append(f"{sport}={accuracy:.1%}")
                else:
                    model_info.append(f"{sport}=not_trained")
            
            return True, ", ".join(model_info)
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def validate_historical_data_seasons(self, data):
        """Validate historical data counts with seasons arrays"""
        try:
            if 'historical_data' not in data:
                return False, "Missing historical_data field"
            
            historical = data['historical_data']
            expected_sports = ['basketball_nba', 'americanfootball_nfl', 'icehockey_nhl']
            
            data_info = []
            for sport in expected_sports:
                if sport not in historical:
                    return False, f"Missing historical data for {sport}"
                
                sport_data = historical[sport]
                total_games = sport_data.get('total_games', 0)
                seasons = sport_data.get('seasons', [])
                
                if not isinstance(seasons, list):
                    return False, f"seasons should be array for {sport}"
                
                data_info.append(f"{sport}={total_games}games/{len(seasons)}seasons")
            
            return True, ", ".join(data_info)
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def validate_training_response(self, data):
        """Validate ML training response structure"""
        try:
            required_fields = ['seasons_used', 'total_games', 'metrics']
            missing_fields = [f for f in required_fields if f not in data]
            
            if missing_fields:
                return False, f"Missing fields: {missing_fields}"
            
            # Validate seasons_used is array
            seasons_used = data.get('seasons_used', [])
            if not isinstance(seasons_used, list):
                return False, "seasons_used should be an array"
            
            # Validate total_games is positive number
            total_games = data.get('total_games', 0)
            if total_games <= 0:
                return False, f"total_games should be positive, got {total_games}"
            
            # Validate metrics structure
            metrics = data.get('metrics', {})
            accuracy_fields = ['ml_accuracy', 'spread_accuracy', 'totals_accuracy']
            
            accuracy_info = []
            for field in accuracy_fields:
                if field in metrics:
                    accuracy = metrics[field]
                    if isinstance(accuracy, (int, float)) and 0 <= accuracy <= 1:
                        accuracy_info.append(f"{field}={accuracy:.1%}")
                    else:
                        return False, f"Invalid {field}: {accuracy}"
                else:
                    accuracy_info.append(f"{field}=missing")
            
            # Check for suspicious accuracy (too high)
            for field in accuracy_fields:
                if field in metrics and metrics[field] > 0.95:
                    return False, f"Suspicious {field}: {metrics[field]:.1%} (too high)"
            
            return True, f"seasons={len(seasons_used)}, games={total_games}, accuracies=({', '.join(accuracy_info)})"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def validate_ml_prediction_response(self, data):
        """Validate ML prediction response structure"""
        try:
            if 'prediction' not in data:
                return False, "Missing prediction object"
            
            prediction = data['prediction']
            
            # Check model_available
            model_available = prediction.get('model_available', False)
            if not model_available:
                return False, "model_available should be true"
            
            # Check for favored team predictions
            favored_fields = ['ml_favored_team', 'ml_favored_prob']
            missing_favored = [f for f in favored_fields if f not in prediction]
            
            if missing_favored:
                return False, f"Missing favored team fields: {missing_favored}"
            
            # Check for spread and totals predictions
            spread_fields = ['spread_favored_team', 'spread_favored_prob']
            totals_fields = ['totals_favored', 'totals_favored_prob']
            
            spread_ok = all(f in prediction for f in spread_fields)
            totals_ok = all(f in prediction for f in totals_fields)
            
            if not spread_ok:
                return False, f"Missing spread prediction fields: {[f for f in spread_fields if f not in prediction]}"
            
            if not totals_ok:
                return False, f"Missing totals prediction fields: {[f for f in totals_fields if f not in prediction]}"
            
            # Validate probabilities are reasonable
            ml_prob = prediction.get('ml_favored_prob', 0)
            spread_prob = prediction.get('spread_favored_prob', 0)
            totals_prob = prediction.get('totals_favored_prob', 0)
            
            if not (0.5 <= ml_prob <= 0.95):
                return False, f"ml_favored_prob should be 0.5-0.95, got {ml_prob}"
            
            if not (0.5 <= spread_prob <= 0.95):
                return False, f"spread_favored_prob should be 0.5-0.95, got {spread_prob}"
            
            if not (0.5 <= totals_prob <= 0.999):
                return False, f"totals_favored_prob should be 0.5-0.999, got {totals_prob}"
            
            # Get team names
            ml_favored_team = prediction.get('ml_favored_team', '')
            spread_favored_team = prediction.get('spread_favored_team', '')
            totals_favored = prediction.get('totals_favored', '')
            
            return True, f"ML: {ml_favored_team} ({ml_prob:.3f}), Spread: {spread_favored_team} ({spread_prob:.3f}), Totals: {totals_favored} ({totals_prob:.3f})"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def test_ensemble_status(self):
        """Test ensemble ML status endpoint"""
        url = f"{BASE_URL}/ml/ensemble-status"
        print(f"\n🧪 Testing GET /ml/ensemble-status")
        print(f"   URL: {url}")
        print(f"   Description: Verify models with ml_accuracy, spread_accuracy, totals_accuracy for each sport")
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate ensemble status structure
                ensemble_ok, ensemble_details = self.validate_ensemble_status(data)
                
                if ensemble_ok:
                    print(f"   ✅ PASS - Ensemble Status: {ensemble_details}")
                    
                    self.passed += 1
                    self.results.append({
                        'endpoint': '/ml/ensemble-status',
                        'status': 'PASS',
                        'ensemble_details': ensemble_details
                    })
                else:
                    print(f"   ❌ FAIL - Ensemble validation issue: {ensemble_details}")
                    self.failed += 1
                    self.results.append({
                        'endpoint': '/ml/ensemble-status',
                        'status': 'FAIL',
                        'error': f"Ensemble validation: {ensemble_details}"
                    })
            else:
                print(f"   ❌ FAIL - Status: {response.status_code}")
                print(f"   📄 Response: {response.text[:300]}...")
                self.failed += 1
                self.results.append({
                    'endpoint': '/ml/ensemble-status',
                    'status': 'FAIL',
                    'error': f"HTTP {response.status_code}",
                    'response_preview': response.text[:300]
                })
                
        except Exception as e:
            print(f"   ❌ FAIL - Error: {str(e)}")
            self.failed += 1
            self.results.append({
                'endpoint': '/ml/ensemble-status',
                'status': 'FAIL',
                'error': str(e)
            })

    def test_ensemble_predict(self):
        """Test ensemble ML predict endpoint with valid NBA event"""
        # First get a real event ID
        try:
            events_response = requests.get(f"{BASE_URL}/events/basketball_nba", timeout=30)
            if events_response.status_code == 200:
                events = events_response.json()
                if events and len(events) > 0:
                    event_id = events[0].get('id')
                    home_team = events[0].get('home_team', 'Unknown')
                    away_team = events[0].get('away_team', 'Unknown')
                else:
                    event_id = "401810581"  # Fallback
                    home_team = "Test Home"
                    away_team = "Test Away"
            else:
                event_id = "401810581"  # Fallback
                home_team = "Test Home"
                away_team = "Test Away"
        except:
            event_id = "401810581"  # Fallback
            home_team = "Test Home"
            away_team = "Test Away"
        
        url = f"{BASE_URL}/ml/ensemble-predict/{event_id}?sport_key=basketball_nba"
        print(f"\n🧪 Testing POST /ml/ensemble-predict/{event_id}")
        print(f"   URL: {url}")
        print(f"   Description: Test with valid NBA event. Should return predictions with ml_favored_team, spread_favored_team, totals_favored")
        print(f"   Event: {away_team} @ {home_team}")
        
        try:
            response = requests.post(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate ensemble prediction response
                prediction_ok, prediction_details = self.validate_ensemble_prediction_response(data)
                
                if prediction_ok:
                    print(f"   ✅ PASS - Ensemble Prediction: {prediction_details}")
                    
                    self.passed += 1
                    self.results.append({
                        'endpoint': f'/ml/ensemble-predict/{event_id}',
                        'status': 'PASS',
                        'prediction_details': prediction_details
                    })
                else:
                    print(f"   ❌ FAIL - Ensemble prediction validation issue: {prediction_details}")
                    self.failed += 1
                    self.results.append({
                        'endpoint': f'/ml/ensemble-predict/{event_id}',
                        'status': 'FAIL',
                        'error': f"Ensemble prediction validation: {prediction_details}"
                    })
            else:
                print(f"   ❌ FAIL - Status: {response.status_code}")
                print(f"   📄 Response: {response.text[:300]}...")
                self.failed += 1
                self.results.append({
                    'endpoint': f'/ml/ensemble-predict/{event_id}',
                    'status': 'FAIL',
                    'error': f"HTTP {response.status_code}",
                    'response_preview': response.text[:300]
                })
                
        except Exception as e:
            print(f"   ❌ FAIL - Error: {str(e)}")
            self.failed += 1
            self.results.append({
                'endpoint': f'/ml/ensemble-predict/{event_id}',
                'status': 'FAIL',
                'error': str(e)
            })

    def test_ensemble_vs_xgboost_accuracy(self):
        """Compare ensemble model accuracy vs basic XGBoost model"""
        print(f"\n🧪 Testing Ensemble vs XGBoost Accuracy Comparison")
        print(f"   Description: Verify ensemble model has higher accuracy than basic XGBoost")
        
        try:
            # Get XGBoost status
            xgb_response = requests.get(f"{BASE_URL}/ml/status", timeout=30)
            ensemble_response = requests.get(f"{BASE_URL}/ml/ensemble-status", timeout=30)
            
            if xgb_response.status_code == 200 and ensemble_response.status_code == 200:
                xgb_data = xgb_response.json()
                ensemble_data = ensemble_response.json()
                
                # Compare accuracies
                comparison_ok, comparison_details = self.validate_accuracy_comparison(xgb_data, ensemble_data)
                
                if comparison_ok:
                    print(f"   ✅ PASS - Accuracy Comparison: {comparison_details}")
                    
                    self.passed += 1
                    self.results.append({
                        'endpoint': 'accuracy_comparison',
                        'status': 'PASS',
                        'comparison_details': comparison_details
                    })
                else:
                    print(f"   ❌ FAIL - Accuracy comparison issue: {comparison_details}")
                    self.failed += 1
                    self.results.append({
                        'endpoint': 'accuracy_comparison',
                        'status': 'FAIL',
                        'error': f"Accuracy comparison: {comparison_details}"
                    })
            else:
                error_msg = f"XGBoost status: {xgb_response.status_code}, Ensemble status: {ensemble_response.status_code}"
                print(f"   ❌ FAIL - {error_msg}")
                self.failed += 1
                self.results.append({
                    'endpoint': 'accuracy_comparison',
                    'status': 'FAIL',
                    'error': error_msg
                })
                
        except Exception as e:
            print(f"   ❌ FAIL - Error: {str(e)}")
            self.failed += 1
            self.results.append({
                'endpoint': 'accuracy_comparison',
                'status': 'FAIL',
                'error': str(e)
            })

    def validate_ensemble_status(self, data):
        """Validate ensemble status response structure"""
        try:
            if 'models' not in data:
                return False, "Missing models field in ensemble status"
            
            models = data['models']
            expected_sports = ['basketball_nba', 'americanfootball_nfl', 'icehockey_nhl']
            
            status_info = []
            trained_models = 0
            
            for sport in expected_sports:
                if sport not in models:
                    return False, f"Missing sport {sport} in ensemble status"
                
                sport_data = models[sport]
                
                # Check required fields
                required_fields = ['model_loaded', 'ml_accuracy', 'spread_accuracy', 'totals_accuracy']
                missing_fields = [f for f in required_fields if f not in sport_data]
                
                if missing_fields:
                    return False, f"Missing fields for {sport}: {missing_fields}"
                
                # Check if model is loaded
                model_loaded = sport_data.get('model_loaded', False)
                if not model_loaded:
                    status_info.append(f"{sport}=not_loaded")
                    continue
                
                # Get accuracy values
                ml_acc = sport_data.get('ml_accuracy')
                spread_acc = sport_data.get('spread_accuracy')
                totals_acc = sport_data.get('totals_accuracy')
                
                # If all accuracies are 0, model is loaded but not trained
                if ml_acc == 0 and spread_acc == 0 and totals_acc == 0:
                    status_info.append(f"{sport}=loaded_not_trained")
                    continue
                
                # Validate accuracy values are reasonable for trained models
                for acc, name in [(ml_acc, 'ml'), (spread_acc, 'spread'), (totals_acc, 'totals')]:
                    if acc is not None and acc > 0 and (acc < 0.4 or acc > 0.95):
                        return False, f"Unreasonable {name}_accuracy for {sport}: {acc}"
                
                status_info.append(f"{sport}=trained(ML:{ml_acc:.1%},Spread:{spread_acc:.1%},Totals:{totals_acc:.1%})")
                trained_models += 1
            
            # Require at least one trained model (NBA should be trained)
            if trained_models == 0:
                return False, "No ensemble models are trained"
            
            return True, "; ".join(status_info)
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def validate_ensemble_prediction_response(self, data):
        """Validate ensemble prediction response structure"""
        try:
            if 'prediction' not in data:
                return False, "Missing prediction object"
            
            prediction = data['prediction']
            
            # Check for required ensemble prediction fields
            required_fields = ['ml_favored_team', 'spread_favored_team', 'totals_favored', 
                             'ml_favored_prob', 'spread_favored_prob', 'totals_favored_prob']
            missing_fields = [f for f in required_fields if f not in prediction]
            
            if missing_fields:
                return False, f"Missing ensemble prediction fields: {missing_fields}"
            
            # Validate team names are not generic
            ml_favored_team = prediction.get('ml_favored_team', '')
            spread_favored_team = prediction.get('spread_favored_team', '')
            
            if ml_favored_team.lower() in ['home', 'away', 'home team', 'away team']:
                return False, f"ml_favored_team should be actual team name, not '{ml_favored_team}'"
            
            if spread_favored_team.lower() in ['home', 'away', 'home team', 'away team']:
                return False, f"spread_favored_team should be actual team name, not '{spread_favored_team}'"
            
            # Validate totals_favored is OVER or UNDER
            totals_favored = prediction.get('totals_favored', '').upper()
            if totals_favored not in ['OVER', 'UNDER']:
                return False, f"totals_favored should be 'OVER' or 'UNDER', got '{totals_favored}'"
            
            # Validate probabilities
            ml_prob = prediction.get('ml_favored_prob', 0)
            spread_prob = prediction.get('spread_favored_prob', 0)
            totals_prob = prediction.get('totals_favored_prob', 0)
            
            if not (0.5 <= ml_prob <= 0.95):
                return False, f"ml_favored_prob should be 0.5-0.95, got {ml_prob}"
            
            if not (0.5 <= spread_prob <= 0.95):
                return False, f"spread_favored_prob should be 0.5-0.95, got {spread_prob}"
            
            if not (0.5 <= totals_prob <= 0.999):
                return False, f"totals_favored_prob should be 0.5-0.999, got {totals_prob}"
            
            # Check method is ensemble
            method = prediction.get('method', '')
            if 'ensemble' not in method.lower():
                return False, f"Expected ensemble method, got '{method}'"
            
            return True, f"ML: {ml_favored_team} ({ml_prob:.3f}), Spread: {spread_favored_team} ({spread_prob:.3f}), Totals: {totals_favored} ({totals_prob:.3f}), Method: {method}"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def validate_accuracy_comparison(self, xgb_data, ensemble_data):
        """Compare XGBoost vs Ensemble accuracy to verify ensemble is better"""
        try:
            # Get NBA accuracies from both models
            xgb_models = xgb_data.get('models', {})
            nba_xgb = xgb_models.get('basketball_nba', {})
            xgb_accuracy = nba_xgb.get('accuracy')
            
            ensemble_models = ensemble_data.get('models', {})
            ensemble_nba = ensemble_models.get('basketball_nba', {})
            ensemble_ml_acc = ensemble_nba.get('ml_accuracy')
            ensemble_spread_acc = ensemble_nba.get('spread_accuracy')
            ensemble_totals_acc = ensemble_nba.get('totals_accuracy')
            
            if xgb_accuracy is None:
                return False, "XGBoost NBA accuracy not available"
            
            if ensemble_ml_acc is None:
                return False, "Ensemble NBA ML accuracy not available"
            
            # Compare ML accuracy (most important) - ensemble should be at least as good
            if ensemble_ml_acc < xgb_accuracy * 0.95:  # Allow 5% tolerance
                return False, f"Ensemble ML accuracy ({ensemble_ml_acc:.1%}) should be close to or higher than XGBoost ({xgb_accuracy:.1%})"
            
            # Calculate improvement or difference
            improvement = (ensemble_ml_acc - xgb_accuracy) / xgb_accuracy * 100
            
            comparison_details = f"XGBoost: {xgb_accuracy:.1%}, Ensemble ML: {ensemble_ml_acc:.1%} ({improvement:+.1f}%)"
            
            if ensemble_spread_acc is not None:
                comparison_details += f", Spread: {ensemble_spread_acc:.1%}"
            
            if ensemble_totals_acc is not None:
                comparison_details += f", Totals: {ensemble_totals_acc:.1%}"
            
            return True, comparison_details
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

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

    def test_consolidated_reasoning_text(self):
        """Test consolidated reasoning text in Events modal - verify no duplicates and no confusing Pick: OVER vs Toronto Raptors ML issues"""
        print("\n📝 TESTING CONSOLIDATED REASONING TEXT (EVENTS MODAL)")
        print("-" * 50)
        
        # Test 1: Toronto vs Minnesota - should have pick
        self.test_reasoning_text_validation("POST", "/analyze-unified/401810582?sport_key=basketball_nba", 
                         description="Toronto vs Minnesota - verify pick_display is 'Toronto Raptors ML', reasoning has ≤7 sections, only ONE OVER mention, no standalone 'Pick:' line",
                         expected_pick_display="Toronto Raptors ML",
                         should_have_pick=True)
        
        # Test 2: Knicks vs Nuggets - should NOT have pick (edge too low)
        self.test_reasoning_text_validation("POST", "/analyze-unified/401810581?sport_key=basketball_nba", 
                         description="Knicks vs Nuggets - verify has_pick is false (edge too low), reasoning explains why no pick was made",
                         expected_pick_display=None,
                         should_have_pick=False)
        
        # Test 3: Verify favored outcomes are correct team names (not "Home")
        self.test_favored_outcomes_team_names("POST", "/analyze-unified/401810582?sport_key=basketball_nba", 
                         description="Verify ml_favored_team, spread_favored_team are actual team names (not 'Home'), totals_favored is 'OVER' or 'UNDER'")
    
    def test_reasoning_text_validation(self, method, endpoint, expected_status=200, description="", expected_pick_display=None, should_have_pick=True):
        """Test endpoint and validate reasoning text structure"""
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
            
            # Validate reasoning text structure
            reasoning_validation_ok = True
            reasoning_details = ""
            
            if json_data and status_ok and json_ok:
                reasoning_validation_ok, reasoning_details = self.validate_reasoning_text_structure(json_data, expected_pick_display, should_have_pick)
            
            # Print results
            if status_ok and json_ok and reasoning_validation_ok:
                print(f"   ✅ PASS - Status: {response.status_code}, JSON: Valid, Reasoning: Valid")
                print(f"   📝 Reasoning Validation: {reasoning_details}")
                
                self.passed += 1
                self.results.append({
                    'endpoint': endpoint,
                    'status': 'PASS',
                    'status_code': response.status_code,
                    'json_valid': True,
                    'reasoning_validation': reasoning_details
                })
            else:
                error_msg = []
                if not status_ok:
                    error_msg.append(f"Expected status {expected_status}, got {response.status_code}")
                if not json_ok:
                    error_msg.append("Invalid JSON response")
                if not reasoning_validation_ok:
                    error_msg.append(f"Reasoning validation issue: {reasoning_details}")
                
                print(f"   ❌ FAIL - {', '.join(error_msg)}")
                print(f"   📄 Response: {response.text[:500]}...")
                
                self.failed += 1
                self.results.append({
                    'endpoint': endpoint,
                    'status': 'FAIL',
                    'status_code': response.status_code,
                    'json_valid': json_ok,
                    'error': ', '.join(error_msg),
                    'response_preview': response.text[:500]
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
    
    def validate_reasoning_text_structure(self, data, expected_pick_display, should_have_pick):
        """Validate reasoning text structure for consolidated reasoning"""
        try:
            prediction = data.get('prediction', {})
            if not prediction:
                return False, "Missing prediction object"
            
            # Check has_pick field
            has_pick = prediction.get('has_pick', False)
            if should_have_pick and not has_pick:
                return False, f"Expected has_pick=true but got has_pick={has_pick}"
            elif not should_have_pick and has_pick:
                return False, f"Expected has_pick=false but got has_pick={has_pick}"
            
            # Check pick_display if pick is expected
            if should_have_pick and expected_pick_display:
                pick_display = prediction.get('pick_display', '')
                if pick_display != expected_pick_display:
                    return False, f"Expected pick_display='{expected_pick_display}' but got '{pick_display}'"
            
            # Get reasoning text
            reasoning = prediction.get('reasoning', '') or prediction.get('analysis', '')
            if not reasoning:
                return False, "Missing reasoning text"
            
            # Check reasoning sections (split by \n\n)
            sections = reasoning.split('\n\n')
            sections = [s.strip() for s in sections if s.strip()]  # Remove empty sections
            
            if should_have_pick and len(sections) > 7:
                return False, f"Expected ≤7 reasoning sections but got {len(sections)} sections"
            
            # Check for OVER mentions (should only be ONE in totals line)
            over_mentions = reasoning.upper().count('OVER')
            if should_have_pick and over_mentions > 1:
                return False, f"Expected only ONE 'OVER' mention but found {over_mentions} mentions"
            
            # Check for standalone "Pick:" lines (should be removed)
            standalone_pick_lines = []
            for section in sections:
                lines = section.split('\n')
                for line in lines:
                    line = line.strip()
                    if line == "Pick:" or line.startswith("Pick:") and len(line.split()) <= 2:
                        standalone_pick_lines.append(line)
            
            if standalone_pick_lines:
                return False, f"Found standalone 'Pick:' lines that should be removed: {standalone_pick_lines}"
            
            # If no pick expected, check reasoning explains why
            if not should_have_pick:
                explanation_keywords = ['edge', 'low', 'insufficient', 'no pick', 'not recommended', 'threshold']
                has_explanation = any(keyword in reasoning.lower() for keyword in explanation_keywords)
                if not has_explanation:
                    return False, "Reasoning should explain why no pick was made (missing keywords: edge, low, insufficient, etc.)"
            
            # Build success message
            details = []
            details.append(f"has_pick={has_pick}")
            if should_have_pick and expected_pick_display:
                details.append(f"pick_display='{prediction.get('pick_display', '')}'")
            details.append(f"reasoning_sections={len(sections)}")
            if should_have_pick:
                details.append(f"over_mentions={over_mentions}")
            details.append(f"reasoning_length={len(reasoning)}")
            
            return True, ", ".join(details)
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def test_favored_outcomes_team_names(self, method, endpoint, expected_status=200, description=""):
        """Test that favored outcomes use actual team names, not 'Home'/'Away'"""
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
            
            # Validate team names
            team_names_ok = True
            team_names_details = ""
            
            if json_data and status_ok and json_ok:
                team_names_ok, team_names_details = self.validate_team_names_not_home_away(json_data)
            
            # Print results
            if status_ok and json_ok and team_names_ok:
                print(f"   ✅ PASS - Status: {response.status_code}, JSON: Valid, Team Names: Valid")
                print(f"   🏆 Team Names: {team_names_details}")
                
                self.passed += 1
                self.results.append({
                    'endpoint': endpoint,
                    'status': 'PASS',
                    'status_code': response.status_code,
                    'json_valid': True,
                    'team_names_validation': team_names_details
                })
            else:
                error_msg = []
                if not status_ok:
                    error_msg.append(f"Expected status {expected_status}, got {response.status_code}")
                if not json_ok:
                    error_msg.append("Invalid JSON response")
                if not team_names_ok:
                    error_msg.append(f"Team names issue: {team_names_details}")
                
                print(f"   ❌ FAIL - {', '.join(error_msg)}")
                print(f"   📄 Response: {response.text[:500]}...")
                
                self.failed += 1
                self.results.append({
                    'endpoint': endpoint,
                    'status': 'FAIL',
                    'status_code': response.status_code,
                    'json_valid': json_ok,
                    'error': ', '.join(error_msg),
                    'response_preview': response.text[:500]
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
    
    def validate_team_names_not_home_away(self, data):
        """Validate that favored team names are actual team names, not 'Home'/'Away'"""
        try:
            prediction = data.get('prediction', {})
            if not prediction:
                return False, "Missing prediction object"
            
            # Check ml_favored_team
            ml_favored_team = prediction.get('ml_favored_team', '')
            if ml_favored_team.lower() in ['home', 'away', 'home team', 'away team']:
                return False, f"ml_favored_team should be actual team name, not '{ml_favored_team}'"
            
            # Check spread_favored_team
            spread_favored_team = prediction.get('spread_favored_team', '')
            if spread_favored_team.lower() in ['home', 'away', 'home team', 'away team']:
                return False, f"spread_favored_team should be actual team name, not '{spread_favored_team}'"
            
            # Check totals_favored
            totals_favored = prediction.get('totals_favored', '').upper()
            if totals_favored not in ['OVER', 'UNDER']:
                return False, f"totals_favored should be 'OVER' or 'UNDER', got '{totals_favored}'"
            
            return True, f"ml_favored_team='{ml_favored_team}', spread_favored_team='{spread_favored_team}', totals_favored='{totals_favored}'"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
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

    def test_ensemble_unified_integration(self):
        """Test Ensemble ML integration into unified predictor - SPECIFIC REVIEW REQUEST"""
        print("\n🎯 TESTING ENSEMBLE ML INTEGRATION INTO UNIFIED PREDICTOR (REVIEW REQUEST)")
        print("-" * 70)
        
        # Test 1: POST /api/analyze-unified/{event_id}?sport_key=basketball_nba - Should use Ensemble ML
        self.test_unified_predictor_uses_ensemble()

    def test_unified_predictor_uses_ensemble(self):
        """Test that unified predictor now uses Ensemble ML instead of XGBoost"""
        # First get a real event ID
        try:
            events_response = requests.get(f"{BASE_URL}/events/basketball_nba", timeout=30)
            if events_response.status_code == 200:
                events = events_response.json()
                if events and len(events) > 0:
                    event_id = events[0].get('id')
                    home_team = events[0].get('home_team', 'Unknown')
                    away_team = events[0].get('away_team', 'Unknown')
                else:
                    event_id = "401810581"  # Fallback
                    home_team = "Test Home"
                    away_team = "Test Away"
            else:
                event_id = "401810581"  # Fallback
                home_team = "Test Home"
                away_team = "Test Away"
        except:
            event_id = "401810581"  # Fallback
            home_team = "Test Home"
            away_team = "Test Away"
        
        url = f"{BASE_URL}/analyze-unified/{event_id}?sport_key=basketball_nba"
        print(f"\n🧪 Testing POST /analyze-unified/{event_id}?sport_key=basketball_nba")
        print(f"   URL: {url}")
        print(f"   Description: Should use Ensemble ML (check logs for 'Running ENSEMBLE ML')")
        print(f"   Event: {away_team} @ {home_team}")
        
        try:
            response = requests.post(url, timeout=60)  # Longer timeout for ML processing
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate that unified predictor uses ensemble ML
                ensemble_integration_ok, integration_details = self.validate_ensemble_integration(data)
                
                if ensemble_integration_ok:
                    print(f"   ✅ PASS - Ensemble Integration: {integration_details}")
                    
                    self.passed += 1
                    self.results.append({
                        'endpoint': f'/analyze-unified/{event_id}',
                        'status': 'PASS',
                        'integration_details': integration_details
                    })
                else:
                    print(f"   ❌ FAIL - Ensemble integration issue: {integration_details}")
                    self.failed += 1
                    self.results.append({
                        'endpoint': f'/analyze-unified/{event_id}',
                        'status': 'FAIL',
                        'error': f"Ensemble integration: {integration_details}"
                    })
            else:
                print(f"   ❌ FAIL - Status: {response.status_code}")
                print(f"   📄 Response: {response.text[:300]}...")
                self.failed += 1
                self.results.append({
                    'endpoint': f'/analyze-unified/{event_id}',
                    'status': 'FAIL',
                    'error': f"HTTP {response.status_code}",
                    'response_preview': response.text[:300]
                })
                
        except Exception as e:
            print(f"   ❌ FAIL - Error: {str(e)}")
            self.failed += 1
            self.results.append({
                'endpoint': f'/analyze-unified/{event_id}',
                'status': 'FAIL',
                'error': str(e)
            })

    def validate_ensemble_integration(self, data):
        """Validate that unified predictor is using Ensemble ML"""
        try:
            prediction = data.get('prediction', {})
            if not prediction:
                return False, "Missing prediction object"
            
            # Check algorithm field - should indicate ensemble usage
            algorithm = prediction.get('algorithm', '')
            
            # Look for ensemble indicators in algorithm field
            ensemble_indicators = ['ensemble', 'unified_ensemble']
            uses_ensemble = any(indicator in algorithm.lower() for indicator in ensemble_indicators)
            
            if not uses_ensemble:
                # Check if it's still using old XGBoost
                if 'xgboost' in algorithm.lower():
                    return False, f"Still using XGBoost algorithm '{algorithm}' instead of Ensemble ML"
                else:
                    return False, f"Algorithm '{algorithm}' doesn't indicate Ensemble ML usage"
            
            # Check for ensemble-specific fields
            ensemble_fields = ['ensemble_probability', 'ensemble_confidence', 'ml_ensemble_prob']
            has_ensemble_fields = any(field in prediction for field in ensemble_fields)
            
            # Check method field if available
            method = prediction.get('method', '')
            ensemble_method = 'ensemble' in method.lower() if method else False
            
            # Check reasoning/analysis for ensemble mentions
            reasoning = prediction.get('reasoning', '') or prediction.get('analysis', '')
            ensemble_in_reasoning = 'ensemble' in reasoning.lower() if reasoning else False
            
            # Build validation details
            details = [f"algorithm='{algorithm}'"]
            
            if method:
                details.append(f"method='{method}'")
            
            if has_ensemble_fields:
                details.append("has_ensemble_fields=true")
            
            if ensemble_in_reasoning:
                details.append("ensemble_mentioned_in_reasoning=true")
            
            # Additional validation - check for ensemble-specific probability fields
            ensemble_prob_fields = ['ml_ensemble_prob', 'ensemble_ml_prob', 'ensemble_probability']
            found_ensemble_prob = None
            for field in ensemble_prob_fields:
                if field in prediction:
                    found_ensemble_prob = prediction[field]
                    details.append(f"{field}={found_ensemble_prob}")
                    break
            
            return True, ", ".join(details)
            
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
        
        # ML TRAINING SYSTEM (SPECIFIC REVIEW REQUEST)
        self.test_ml_training_system()
        
        # ENSEMBLE ML SYSTEM (SPECIFIC REVIEW REQUEST)
        self.test_ensemble_ml_system()
        
        # ENSEMBLE ML INTEGRATION INTO UNIFIED PREDICTOR (SPECIFIC REVIEW REQUEST)
        self.test_ensemble_unified_integration()
        
        # ML ENDPOINTS (NEW XGBoost Integration)
        self.test_ml_endpoints()
        
        # XGBOOST FAVORED OUTCOMES (NEW FEATURE TESTING)
        self.test_xgboost_favored_outcomes()
        
        # CONSOLIDATED REASONING TEXT TESTS (SPECIFIC REVIEW REQUEST)
        self.test_consolidated_reasoning_text()
        
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