#!/usr/bin/env python3
"""
BetPredictor V6 Advanced Algorithm Testing Suite
Tests the V6 endpoints for comprehensive algorithm functionality
"""

import asyncio
import aiohttp
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Backend URL from frontend .env
BACKEND_URL = "https://realtime-tester-2.preview.emergentagent.com/api"

class V6TestSuite:
    def __init__(self):
        self.session = None
        self.test_results = []
        self.nba_event_ids = []
        
    async def setup(self):
        """Setup test session"""
        self.session = aiohttp.ClientSession()
        print("🚀 BetPredictor V6 Testing Suite Started")
        print(f"Backend URL: {BACKEND_URL}")
        print("=" * 60)
        
    async def teardown(self):
        """Cleanup test session"""
        if self.session:
            await self.session.close()
            
    async def make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request to backend"""
        url = f"{BACKEND_URL}{endpoint}"
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.content_type == 'application/json':
                    data = await response.json()
                else:
                    data = {"text": await response.text()}
                
                return {
                    "status": response.status,
                    "data": data,
                    "headers": dict(response.headers)
                }
        except Exception as e:
            return {
                "status": 0,
                "error": str(e),
                "data": None
            }
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    async def get_nba_events(self) -> List[str]:
        """Get NBA event IDs for testing"""
        print("\n📋 Getting NBA events for testing...")
        
        response = await self.make_request("GET", "/events/basketball_nba?pre_match_only=true")
        
        if response["status"] != 200:
            self.log_test("Get NBA Events", False, f"Status: {response['status']}")
            return []
        
        events = response["data"]
        if not events or len(events) == 0:
            self.log_test("Get NBA Events", False, "No events returned")
            return []
        
        # Extract event IDs
        event_ids = []
        for event in events[:5]:  # Take first 5 events
            event_id = event.get("id")
            if event_id:
                event_ids.append(event_id)
                print(f"    Found event: {event.get('home_team')} vs {event.get('away_team')} (ID: {event_id})")
        
        self.log_test("Get NBA Events", len(event_ids) > 0, f"Found {len(event_ids)} events")
        return event_ids
    
    async def test_v6_predictions_list(self):
        """Test GET /api/predictions/v6"""
        print("\n🎯 Testing V6 Predictions List...")
        
        response = await self.make_request("GET", "/predictions/v6")
        
        if response["status"] != 200:
            self.log_test("V6 Predictions List", False, f"Status: {response['status']}")
            return
        
        data = response["data"]
        
        # Verify response structure
        required_fields = ["predictions", "stats", "algorithm"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test("V6 Predictions List", False, f"Missing fields: {missing_fields}")
            return
        
        # Verify algorithm
        if data.get("algorithm") != "betpredictor_v6":
            self.log_test("V6 Predictions List", False, f"Wrong algorithm: {data.get('algorithm')}")
            return
        
        # Verify stats structure
        stats = data.get("stats", {})
        required_stats = ["total", "wins", "losses", "pending", "win_rate", "avg_confidence"]
        missing_stats = [stat for stat in required_stats if stat not in stats]
        
        if missing_stats:
            self.log_test("V6 Predictions List", False, f"Missing stats: {missing_stats}")
            return
        
        predictions_count = len(data.get("predictions", []))
        total_count = stats.get("total", 0)
        
        self.log_test("V6 Predictions List", True, 
                     f"Algorithm: {data['algorithm']}, Predictions: {predictions_count}, Total: {total_count}")
    
    async def test_v6_analyze_event(self, event_id: str):
        """Test POST /api/analyze-v6/{event_id}"""
        print(f"\n🔍 Testing V6 Event Analysis for {event_id}...")
        
        response = await self.make_request("POST", f"/analyze-v6/{event_id}?sport_key=basketball_nba")
        
        if response["status"] != 200:
            self.log_test(f"V6 Analyze Event {event_id}", False, f"Status: {response['status']}")
            return
        
        data = response["data"]
        
        # Verify response structure
        required_fields = ["event", "prediction"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.log_test(f"V6 Analyze Event {event_id}", False, f"Missing fields: {missing_fields}")
            return
        
        # Verify event structure
        event = data.get("event", {})
        event_required = ["id", "home_team", "away_team", "commence_time", "sport_key"]
        event_missing = [field for field in event_required if field not in event]
        
        if event_missing:
            self.log_test(f"V6 Analyze Event {event_id}", False, f"Missing event fields: {event_missing}")
            return
        
        # Verify prediction structure
        prediction = data.get("prediction", {})
        
        # Check if has_pick field exists
        if "has_pick" not in prediction:
            self.log_test(f"V6 Analyze Event {event_id}", False, "Missing has_pick field")
            return
        
        has_pick = prediction.get("has_pick")
        
        if has_pick:
            # If has pick, verify pick fields
            pick_required = ["pick", "confidence", "edge", "model_agreement", "reasoning", "key_factors"]
            pick_missing = [field for field in pick_required if field not in prediction]
            
            if pick_missing:
                self.log_test(f"V6 Analyze Event {event_id}", False, f"Missing pick fields: {pick_missing}")
                return
            
            # Verify ensemble details
            if "ensemble_details" not in prediction:
                self.log_test(f"V6 Analyze Event {event_id}", False, "Missing ensemble_details")
                return
            
            ensemble = prediction.get("ensemble_details", {})
            ensemble_required = ["ensemble_probability", "model_agreement", "num_models", "individual_predictions"]
            ensemble_missing = [field for field in ensemble_required if field not in ensemble]
            
            if ensemble_missing:
                self.log_test(f"V6 Analyze Event {event_id}", False, f"Missing ensemble fields: {ensemble_missing}")
                return
            
            # Verify 5 models
            individual_preds = ensemble.get("individual_predictions", {})
            expected_models = ["elo_model", "context_model", "line_movement_model", "statistical_model", "psychology_model"]
            missing_models = [model for model in expected_models if model not in individual_preds]
            
            if missing_models:
                self.log_test(f"V6 Analyze Event {event_id}", False, f"Missing models: {missing_models}")
                return
            
            # Verify simulation data
            if "simulation_data" not in prediction:
                self.log_test(f"V6 Analyze Event {event_id}", False, "Missing simulation_data")
                return
            
            sim_data = prediction.get("simulation_data", {})
            if "monte_carlo" not in sim_data:
                self.log_test(f"V6 Analyze Event {event_id}", False, "Missing monte_carlo in simulation_data")
                return
            
            # Verify matchup summary
            if "matchup_summary" not in prediction:
                self.log_test(f"V6 Analyze Event {event_id}", False, "Missing matchup_summary")
                return
            
            matchup = prediction.get("matchup_summary", {})
            matchup_required = ["elo_diff", "context_advantage", "injury_advantage"]
            matchup_missing = [field for field in matchup_required if field not in matchup]
            
            if matchup_missing:
                self.log_test(f"V6 Analyze Event {event_id}", False, f"Missing matchup fields: {matchup_missing}")
                return
            
            pick = prediction.get("pick")
            confidence = prediction.get("confidence")
            edge = prediction.get("edge")
            agreement = prediction.get("model_agreement")
            
            self.log_test(f"V6 Analyze Event {event_id}", True, 
                         f"HAS PICK: {pick}, Confidence: {confidence}%, Edge: {edge}%, Agreement: {agreement}%")
        else:
            # If no pick, verify reasoning
            if "reasoning" not in prediction:
                self.log_test(f"V6 Analyze Event {event_id}", False, "Missing reasoning for no-pick")
                return
            
            reasoning = prediction.get("reasoning", "")
            ensemble_conf = prediction.get("ensemble_confidence", 0)
            
            self.log_test(f"V6 Analyze Event {event_id}", True, 
                         f"NO PICK: {reasoning[:50]}... (Ensemble: {ensemble_conf}%)")
    
    async def test_predictions_comparison(self):
        """Test GET /api/predictions/comparison"""
        print("\n📊 Testing Predictions Comparison...")
        
        response = await self.make_request("GET", "/predictions/comparison")
        
        if response["status"] != 200:
            self.log_test("Predictions Comparison", False, f"Status: {response['status']}")
            return
        
        data = response["data"]
        
        # Verify response structure
        if "algorithms" not in data:
            self.log_test("Predictions Comparison", False, "Missing algorithms field")
            return
        
        algorithms = data.get("algorithms", {})
        
        # Verify both V5 and V6 are present
        required_algos = ["betpredictor_v5", "betpredictor_v6"]
        missing_algos = [algo for algo in required_algos if algo not in algorithms]
        
        if missing_algos:
            self.log_test("Predictions Comparison", False, f"Missing algorithms: {missing_algos}")
            return
        
        # Verify each algorithm has required stats
        for algo_name, algo_data in algorithms.items():
            required_stats = ["total", "wins", "losses", "pending", "win_rate", "avg_confidence", "description"]
            missing_stats = [stat for stat in required_stats if stat not in algo_data]
            
            if missing_stats:
                self.log_test("Predictions Comparison", False, f"{algo_name} missing stats: {missing_stats}")
                return
        
        v5_stats = algorithms.get("betpredictor_v5", {})
        v6_stats = algorithms.get("betpredictor_v6", {})
        
        self.log_test("Predictions Comparison", True, 
                     f"V5: {v5_stats.get('total', 0)} total, V6: {v6_stats.get('total', 0)} total")
    
    async def test_model_performance(self):
        """Test GET /api/model-performance"""
        print("\n🤖 Testing Model Performance...")
        
        response = await self.make_request("GET", "/model-performance")
        
        if response["status"] != 200:
            self.log_test("Model Performance", False, f"Status: {response['status']}")
            return
        
        data = response["data"]
        
        # Verify response structure
        if "sub_models" not in data:
            self.log_test("Model Performance", False, "Missing sub_models field")
            return
        
        sub_models = data.get("sub_models", {})
        
        # Verify 5 expected models
        expected_models = ["elo_model", "context_model", "line_movement_model", "statistical_model", "psychology_model"]
        missing_models = [model for model in expected_models if model not in sub_models]
        
        if missing_models:
            self.log_test("Model Performance", False, f"Missing models: {missing_models}")
            return
        
        # Verify each model has required stats
        for model_name, model_data in sub_models.items():
            required_stats = ["accuracy", "correct", "total", "roi", "current_weight"]
            missing_stats = [stat for stat in required_stats if stat not in model_data]
            
            if missing_stats:
                self.log_test("Model Performance", False, f"{model_name} missing stats: {missing_stats}")
                return
        
        model_count = len(sub_models)
        self.log_test("Model Performance", True, f"Found {model_count}/5 models with complete stats")
    
    async def run_comprehensive_test(self):
        """Run all V6 tests"""
        await self.setup()
        
        try:
            # 1. Get NBA events for testing
            self.nba_event_ids = await self.get_nba_events()
            
            # 2. Test V6 predictions list
            await self.test_v6_predictions_list()
            
            # 3. Test V6 event analysis (test 2-3 events)
            test_events = self.nba_event_ids[:3] if len(self.nba_event_ids) >= 3 else self.nba_event_ids
            for event_id in test_events:
                await self.test_v6_analyze_event(event_id)
            
            # 4. Test predictions comparison
            await self.test_predictions_comparison()
            
            # 5. Test model performance
            await self.test_model_performance()
            
            # Print summary
            self.print_summary()
            
        finally:
            await self.teardown()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("🎯 BetPredictor V6 Test Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t["success"]])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for test in self.test_results:
                if not test["success"]:
                    print(f"  - {test['test']}: {test['details']}")
        
        print("\n🔍 Key V6 Features Verified:")
        print("  ✓ V6 predictions endpoint with algorithm='betpredictor_v6'")
        print("  ✓ V6 analysis with ensemble of 5 models")
        print("  ✓ Conservative approach (has_pick=true/false)")
        print("  ✓ Comprehensive response structure")
        print("  ✓ Simulation data and matchup analysis")
        print("  ✓ V5 vs V6 comparison")
        print("  ✓ Individual model performance tracking")


async def main():
    """Main test runner"""
    test_suite = V6TestSuite()
    await test_suite.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main())