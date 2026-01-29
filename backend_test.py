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
    
    def run_all_tests(self):
        """Run all API endpoint tests"""
        print("=" * 60)
        print("🚀 BetPredictor Backend API Test Suite")
        print("=" * 60)
        
        # Test 1: Health check
        self.test_endpoint("GET", "/", description="Health check endpoint")
        
        # Test 2: Sports list
        self.test_endpoint("GET", "/sports", description="List available sports")
        
        # Test 3: NBA events
        self.test_endpoint("GET", "/events/basketball_nba", description="Get NBA events with odds")
        
        # Test 4: AI recommendations
        self.test_endpoint("GET", "/recommendations", description="Get AI recommendations")
        
        # Test 5: Performance stats
        self.test_endpoint("GET", "/performance", description="Get performance statistics")
        
        # Test 6: Notifications
        self.test_endpoint("GET", "/notifications", description="Get notifications list")
        
        # Test 7: Data source status
        self.test_endpoint("GET", "/data-source-status", description="Get ESPN data source status")
        
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