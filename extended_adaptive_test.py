#!/usr/bin/env python3
"""
Extended Backend Test Suite for BetPredictor Adaptive Learning System
Additional tests for edge cases and different parameters.
"""

import asyncio
import aiohttp
import json
from datetime import datetime

# Backend URL from frontend .env
BASE_URL = "https://project-navigator-11.preview.emergentagent.com/api"

class ExtendedAdaptiveLearningTester:
    def __init__(self):
        self.session = None
        self.test_results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test result"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")
        print()
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> tuple[int, dict]:
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
    
    async def test_different_sports(self):
        """Test model stats for different sports"""
        sports = ["basketball_nba", "americanfootball_nfl", "icehockey_nhl"]
        
        for sport in sports:
            status_code, data = await self.make_request("GET", f"/adaptive-learning/model-stats/{sport}")
            
            if status_code == 200 and data.get("sport_key") == sport:
                self.log_test(f"Model Stats - {sport}", "PASS", 
                             f"Successfully retrieved stats for {sport}")
            else:
                self.log_test(f"Model Stats - {sport}", "FAIL", 
                             f"Failed to get stats for {sport}: {status_code}")
    
    async def test_different_models(self):
        """Test rolling performance for different models"""
        models = ["elo_model", "context_model", "line_movement_model", "statistical_model", "psychology_model"]
        
        for model in models:
            status_code, data = await self.make_request("GET", 
                f"/adaptive-learning/rolling-performance/{model}?sport_key=basketball_nba&days=7")
            
            if status_code == 200:
                # Check if it's the "no data" case or proper response
                if "error" in data and data.get("error") == "No data found":
                    self.log_test(f"Rolling Performance - {model}", "PASS", 
                                 f"No historical data for {model} (expected for new system)")
                elif data.get("model_name") == model:
                    self.log_test(f"Rolling Performance - {model}", "PASS", 
                                 f"Retrieved performance data for {model}")
                else:
                    self.log_test(f"Rolling Performance - {model}", "FAIL", 
                                 f"Unexpected response structure for {model}")
            else:
                self.log_test(f"Rolling Performance - {model}", "FAIL", 
                             f"Failed to get performance for {model}: {status_code}")
    
    async def test_different_time_windows(self):
        """Test rolling performance with different time windows"""
        windows = [7, 14, 30, 60]
        
        for days in windows:
            status_code, data = await self.make_request("GET", 
                f"/adaptive-learning/rolling-performance/elo_model?sport_key=basketball_nba&days={days}")
            
            if status_code == 200:
                # Check if window_days matches (if data structure is returned)
                if "window_days" in data and data.get("window_days") == days:
                    self.log_test(f"Time Window - {days} days", "PASS", 
                                 f"Correctly handled {days}-day window")
                elif "error" in data and data.get("error") == "No data found":
                    self.log_test(f"Time Window - {days} days", "PASS", 
                                 f"No data for {days}-day window (expected for new system)")
                else:
                    self.log_test(f"Time Window - {days} days", "FAIL", 
                                 f"Unexpected response for {days}-day window")
            else:
                self.log_test(f"Time Window - {days} days", "FAIL", 
                             f"Failed to get {days}-day window: {status_code}")
    
    async def test_lr_weights_different_sports(self):
        """Test LR weights for different sports"""
        sports = ["basketball_nba", "americanfootball_nfl", "icehockey_nhl"]
        
        for sport in sports:
            status_code, data = await self.make_request("GET", f"/adaptive-learning/lr-weights/{sport}")
            
            if status_code == 200 and data.get("sport_key") == sport:
                weights = data.get("weights", {})
                if len(weights) >= 8:  # Should have at least 8 weight parameters
                    self.log_test(f"LR Weights - {sport}", "PASS", 
                                 f"Retrieved {len(weights)} weight parameters for {sport}")
                else:
                    self.log_test(f"LR Weights - {sport}", "FAIL", 
                                 f"Insufficient weight parameters for {sport}: {len(weights)}")
            else:
                self.log_test(f"LR Weights - {sport}", "FAIL", 
                             f"Failed to get LR weights for {sport}: {status_code}")
    
    async def test_calibration_with_sport_filter(self):
        """Test calibration report with sport-specific filter"""
        sports = ["basketball_nba", "americanfootball_nfl", None]  # None for "all"
        
        for sport in sports:
            endpoint = "/adaptive-learning/calibration"
            if sport:
                endpoint += f"?sport_key={sport}"
            
            status_code, data = await self.make_request("GET", endpoint)
            
            if status_code == 200:
                expected_sport = sport or "all"
                if data.get("sport_key") == expected_sport:
                    calibration = data.get("calibration", {})
                    if "brier_score" in calibration and "total_samples" in calibration:
                        self.log_test(f"Calibration - {expected_sport}", "PASS", 
                                     f"Retrieved calibration for {expected_sport}")
                    else:
                        self.log_test(f"Calibration - {expected_sport}", "FAIL", 
                                     f"Missing calibration fields for {expected_sport}")
                else:
                    self.log_test(f"Calibration - {expected_sport}", "FAIL", 
                                 f"Wrong sport_key in response for {expected_sport}")
            else:
                self.log_test(f"Calibration - {expected_sport}", "FAIL", 
                             f"Failed to get calibration for {expected_sport}: {status_code}")
    
    async def run_extended_tests(self):
        """Run all extended tests"""
        print("🧠 EXTENDED ADAPTIVE LEARNING SYSTEM TESTS")
        print("=" * 60)
        print()
        
        await self.test_different_sports()
        await self.test_different_models()
        await self.test_different_time_windows()
        await self.test_lr_weights_different_sports()
        await self.test_calibration_with_sport_filter()
        
        # Summary
        print("=" * 60)
        print("EXTENDED TEST SUMMARY")
        print("=" * 60)
        
        passed = len([r for r in self.test_results if r["status"] == "PASS"])
        failed = len([r for r in self.test_results if r["status"] == "FAIL"])
        total = len(self.test_results)
        
        print(f"Total Extended Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        print()
        
        if failed > 0:
            print("FAILED EXTENDED TESTS:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"❌ {result['test']}: {result['details']}")
            print()
        
        return passed, failed, total

async def main():
    """Main extended test runner"""
    async with ExtendedAdaptiveLearningTester() as tester:
        passed, failed, total = await tester.run_extended_tests()
        
        if failed == 0:
            print("🎉 ALL EXTENDED ADAPTIVE LEARNING TESTS PASSED!")
        else:
            print(f"⚠️  {failed} out of {total} extended tests failed")
        
        return failed == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)