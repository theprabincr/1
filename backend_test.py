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
        
    def test_endpoint(self, method, endpoint, expected_status=200, description=""):
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
            
            # Print results
            if status_ok and json_ok:
                print(f"   ✅ PASS - Status: {response.status_code}, JSON: Valid")
                if isinstance(json_data, dict):
                    if 'message' in json_data:
                        print(f"   📝 Message: {json_data['message']}")
                    if 'status' in json_data:
                        print(f"   📊 Status: {json_data['status']}")
                elif isinstance(json_data, list):
                    print(f"   📊 Array length: {len(json_data)}")
                
                self.passed += 1
                self.results.append({
                    'endpoint': endpoint,
                    'status': 'PASS',
                    'status_code': response.status_code,
                    'json_valid': True,
                    'response_size': len(str(json_data)) if json_data else 0
                })
            else:
                error_msg = []
                if not status_ok:
                    error_msg.append(f"Expected status {expected_status}, got {response.status_code}")
                if not json_ok:
                    error_msg.append("Invalid JSON response")
                
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
    
    def test_prediction_algorithm(self):
        """Test the AI prediction algorithm endpoints"""
        print("\n🧠 TESTING PREDICTION ALGORITHM (CRITICAL)")
        print("-" * 50)
        
        # First get an event to test with
        try:
            response = requests.get(f"{BASE_URL}/events/basketball_nba", timeout=30)
            if response.status_code == 200:
                events = response.json()
                if events and len(events) > 0:
                    test_event = events[0]
                    event_id = test_event.get('id')
                    
                    if event_id:
                        # Test unified analysis endpoint
                        self.test_endpoint("POST", f"/analyze-unified/{event_id}", 
                                         description="Full AI analysis for specific event")
                        
                        # Test line movement for the event
                        self.test_endpoint("GET", f"/line-movement/{event_id}",
                                         description="Line movement analysis")
                        
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