#!/usr/bin/env python3
"""
Backend Test Suite for BetPredictor Adaptive Learning System
Tests the new adaptive learning endpoints that allow ML models to learn and improve from past results.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, Any, List

# Backend URL from frontend .env
BASE_URL = "https://project-navigator-11.preview.emergentagent.com/api"

class AdaptiveLearningTester:
    def __init__(self):
        self.session = None
        self.test_results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_test(self, test_name: str, status: str, details: str = "", response_data: Any = None):
        """Log test result"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "response_data": response_data
        }
        self.test_results.append(result)
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")
        if response_data and status == "FAIL":
            print(f"   Response: {json.dumps(response_data, indent=2)}")
        print()
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> tuple[int, Dict]:
        """Make HTTP request and return status code and response data"""
        url = f"{BASE_URL}{endpoint}"
        try:
            async with self.session.request(method, url, **kwargs) as response:
                status_code = response.status
                try:
                    data = await response.json()
                except:
                    data = {"error": "Invalid JSON response", "text": await response.text()}
                return status_code, data
        except Exception as e:
            return 0, {"error": str(e)}
    
    async def test_adaptive_learning_status(self):
        """Test GET /api/adaptive-learning/status"""
        status_code, data = await self.make_request("GET", "/adaptive-learning/status")
        
        if status_code != 200:
            self.log_test("Adaptive Learning Status", "FAIL", 
                         f"Expected 200, got {status_code}", data)
            return
        
        # Check required fields
        required_fields = ["status", "model_performance", "current_weights_by_sport"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("Adaptive Learning Status", "FAIL", 
                         f"Missing required fields: {missing_fields}", data)
            return
        
        # Check status is "active"
        if data.get("status") != "active":
            self.log_test("Adaptive Learning Status", "FAIL", 
                         f"Expected status 'active', got '{data.get('status')}'", data)
            return
        
        # Check sports weights structure
        weights_by_sport = data.get("current_weights_by_sport", {})
        expected_sports = ["basketball_nba", "americanfootball_nfl", "icehockey_nhl"]
        expected_models = ["elo_model", "context_model", "line_movement_model", "statistical_model", "psychology_model"]
        
        for sport in expected_sports:
            if sport not in weights_by_sport:
                self.log_test("Adaptive Learning Status", "FAIL", 
                             f"Missing sport weights for {sport}", data)
                return
            
            sport_weights = weights_by_sport[sport]
            for model in expected_models:
                if model not in sport_weights:
                    self.log_test("Adaptive Learning Status", "FAIL", 
                                 f"Missing {model} weight for {sport}", data)
                    return
        
        self.log_test("Adaptive Learning Status", "PASS", 
                     f"Status: {data.get('status')}, Sports: {list(weights_by_sport.keys())}")
    
    async def test_model_stats_by_sport(self):
        """Test GET /api/adaptive-learning/model-stats/basketball_nba"""
        status_code, data = await self.make_request("GET", "/adaptive-learning/model-stats/basketball_nba")
        
        if status_code != 200:
            self.log_test("Model Stats by Sport", "FAIL", 
                         f"Expected 200, got {status_code}", data)
            return
        
        # Check required fields
        required_fields = ["sport_key", "model_stats", "current_weights", "best_performer", "worst_performer"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("Model Stats by Sport", "FAIL", 
                         f"Missing required fields: {missing_fields}", data)
            return
        
        # Check sport_key matches
        if data.get("sport_key") != "basketball_nba":
            self.log_test("Model Stats by Sport", "FAIL", 
                         f"Expected sport_key 'basketball_nba', got '{data.get('sport_key')}'", data)
            return
        
        # Check current_weights has all 5 models
        current_weights = data.get("current_weights", {})
        expected_models = ["elo_model", "context_model", "line_movement_model", "statistical_model", "psychology_model"]
        
        for model in expected_models:
            if model not in current_weights:
                self.log_test("Model Stats by Sport", "FAIL", 
                             f"Missing {model} in current_weights", data)
                return
        
        self.log_test("Model Stats by Sport", "PASS", 
                     f"Sport: {data.get('sport_key')}, Models: {list(current_weights.keys())}")
    
    async def test_rolling_performance(self):
        """Test GET /api/adaptive-learning/rolling-performance/elo_model?sport_key=basketball_nba&days=30"""
        status_code, data = await self.make_request("GET", "/adaptive-learning/rolling-performance/elo_model?sport_key=basketball_nba&days=30")
        
        if status_code != 200:
            self.log_test("Rolling Performance", "FAIL", 
                         f"Expected 200, got {status_code}", data)
            return
        
        # Handle case where no data exists yet (newly initialized system)
        if "error" in data and data.get("error") == "No data found":
            self.log_test("Rolling Performance", "PASS", 
                         "No historical data yet (newly initialized system)", data)
            return
        
        # Check required fields for normal response
        required_fields = ["model_name", "sport_key", "window_days"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("Rolling Performance", "FAIL", 
                         f"Missing required fields: {missing_fields}", data)
            return
        
        # Check values match request
        if data.get("model_name") != "elo_model":
            self.log_test("Rolling Performance", "FAIL", 
                         f"Expected model_name 'elo_model', got '{data.get('model_name')}'", data)
            return
        
        if data.get("sport_key") != "basketball_nba":
            self.log_test("Rolling Performance", "FAIL", 
                         f"Expected sport_key 'basketball_nba', got '{data.get('sport_key')}'", data)
            return
        
        if data.get("window_days") != 30:
            self.log_test("Rolling Performance", "FAIL", 
                         f"Expected window_days 30, got {data.get('window_days')}", data)
            return
        
        # Should have either data or "No results in window" message
        has_data = "total_in_window" in data and data.get("total_in_window", 0) > 0
        has_message = "message" in data and "No results in window" in data.get("message", "")
        
        if not (has_data or has_message):
            self.log_test("Rolling Performance", "FAIL", 
                         "Expected either data with total_in_window > 0 or 'No results in window' message", data)
            return
        
        result_type = "with data" if has_data else "no results in window"
        self.log_test("Rolling Performance", "PASS", 
                     f"Model: {data.get('model_name')}, Sport: {data.get('sport_key')}, Days: {data.get('window_days')}, Result: {result_type}")
    
    async def test_calibration_report(self):
        """Test GET /api/adaptive-learning/calibration"""
        status_code, data = await self.make_request("GET", "/adaptive-learning/calibration")
        
        if status_code != 200:
            self.log_test("Calibration Report", "FAIL", 
                         f"Expected 200, got {status_code}", data)
            return
        
        # Check required fields
        required_fields = ["sport_key", "calibration"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("Calibration Report", "FAIL", 
                         f"Missing required fields: {missing_fields}", data)
            return
        
        # Check sport_key is "all"
        if data.get("sport_key") != "all":
            self.log_test("Calibration Report", "FAIL", 
                         f"Expected sport_key 'all', got '{data.get('sport_key')}'", data)
            return
        
        # Check calibration structure
        calibration = data.get("calibration", {})
        expected_cal_fields = ["buckets", "brier_score", "interpretation", "total_samples"]
        
        for field in expected_cal_fields:
            if field not in calibration:
                self.log_test("Calibration Report", "FAIL", 
                             f"Missing {field} in calibration object", data)
                return
        
        # Check buckets is an array
        if not isinstance(calibration.get("buckets"), list):
            self.log_test("Calibration Report", "FAIL", 
                         "Expected 'buckets' to be an array", data)
            return
        
        self.log_test("Calibration Report", "PASS", 
                     f"Sport: {data.get('sport_key')}, Buckets: {len(calibration.get('buckets', []))}, Total samples: {calibration.get('total_samples', 0)}")
    
    async def test_lr_weights(self):
        """Test GET /api/adaptive-learning/lr-weights/basketball_nba"""
        status_code, data = await self.make_request("GET", "/adaptive-learning/lr-weights/basketball_nba")
        
        if status_code != 200:
            self.log_test("LR Weights", "FAIL", 
                         f"Expected 200, got {status_code}", data)
            return
        
        # Check required fields
        required_fields = ["sport_key", "weights", "updates_count", "status", "note"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("LR Weights", "FAIL", 
                         f"Missing required fields: {missing_fields}", data)
            return
        
        # Check sport_key matches
        if data.get("sport_key") != "basketball_nba":
            self.log_test("LR Weights", "FAIL", 
                         f"Expected sport_key 'basketball_nba', got '{data.get('sport_key')}'", data)
            return
        
        # Check weights structure
        weights = data.get("weights", {})
        expected_weight_fields = ["elo_diff", "form_diff", "margin_diff", "rest_advantage", 
                                "home_advantage", "injury_impact", "line_movement", 
                                "four_factors", "intercept"]
        
        for field in expected_weight_fields:
            if field not in weights:
                self.log_test("LR Weights", "FAIL", 
                             f"Missing {field} in weights object", data)
                return
        
        # Check updates_count is a number
        updates_count = data.get("updates_count")
        if not isinstance(updates_count, (int, float)):
            self.log_test("LR Weights", "FAIL", 
                         f"Expected updates_count to be a number, got {type(updates_count)}", data)
            return
        
        # Check status is valid
        status = data.get("status")
        if status not in ["using_defaults", "learned"]:
            self.log_test("LR Weights", "FAIL", 
                         f"Expected status 'using_defaults' or 'learned', got '{status}'", data)
            return
        
        # Check note mentions online learning
        note = data.get("note", "")
        if "online learning" not in note.lower():
            self.log_test("LR Weights", "FAIL", 
                         f"Expected note to mention 'online learning', got '{note}'", data)
            return
        
        self.log_test("LR Weights", "PASS", 
                     f"Sport: {data.get('sport_key')}, Updates: {updates_count}, Status: {status}")
    
    async def run_all_tests(self):
        """Run all adaptive learning tests"""
        print("🧠 TESTING ADAPTIVE LEARNING SYSTEM ENDPOINTS")
        print("=" * 60)
        print()
        
        # Test all endpoints
        await self.test_adaptive_learning_status()
        await self.test_model_stats_by_sport()
        await self.test_rolling_performance()
        await self.test_calibration_report()
        await self.test_lr_weights()
        
        # Summary
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = len([r for r in self.test_results if r["status"] == "PASS"])
        failed = len([r for r in self.test_results if r["status"] == "FAIL"])
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        print()
        
        if failed > 0:
            print("FAILED TESTS:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"❌ {result['test']}: {result['details']}")
            print()
        
        return passed, failed, total

async def main():
    """Main test runner"""
    async with AdaptiveLearningTester() as tester:
        passed, failed, total = await tester.run_all_tests()
        
        if failed == 0:
            print("🎉 ALL ADAPTIVE LEARNING TESTS PASSED!")
        else:
            print(f"⚠️  {failed} out of {total} tests failed")
        
        return failed == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)